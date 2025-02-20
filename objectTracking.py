import os
import cv2
import torch
import numpy as np
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
#本程式利用現有之yoloXs逐幀偵測影片中之「人類」物件
#並透過實作匈牙利演算法處理每幀之間物件之對應關係
#並根據消失物件特徵配對給我蒸發一段時間又給我蹦出來的王八

#input: (1) pretrained yoloXs weight
#       (2) mp4格式之影片
#output:偵測物件後之mp4檔案

#預設路徑如下
#C:\Users\Lemonsky0618\YOLOX\yolox_s.pth'
#D:\0-IOC-1\VIDEO STREAM\HW3\HW3\video\demo.mp4'
#D:\0-IOC-1\VIDEO STREAM\HW3\HW3\result\demo.mp4'
#如有需要更改 請至第1723行更改路徑

#Hyperparameter: 
yoloXs_score_threshold=0.7 #信心值閾值
global_iou_threshold=0.15 #IOU閾值
h_central_radius=0.499 #中央區域閾值(高) 透過調整此參數改變判斷「這人憑空出來的 一定是我的YOLO腦袋壞掉之前沒看到他 他不是新人」的有效區域
w_central_radius=0.475 #中央區域閾值(寬)
disappear_ratio=6.5 #消失再出現的允許範圍=長:畫面長/disappear_ratio  寬:畫面寬/disappear_ratio 的長方形ㄉ對角線

# similarity=direction_factor*方向相似性[-1 1] +magnitude_factor*大小相似性[0 1] +distance_factor*距離相似性[0 1]
direction_factor=0.5 
magnitude_factor=0.5
distance_factor=0.5
distance_decay_factor=0.1 #距離相似性最高的 1 第二高的 1-distance_decay_factor 第三高的1-2*distance_decay_factor...
similarity_threshold = 0.1  #抓similarity最高的並允許similarity只輸不到similarity_threshold的加入競爭

#  YOLOX 
def init_yolox(model_path, exp_file=r'C:\YOLOX-main\YOLOX-main\exps\default\yolox_s.py'):
    exp = get_exp(exp_file)
    model = exp.get_model()
    model.eval() #禁止更新
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = fuse_model(model)
    return model, exp

# 自定義預處理函式
def my_preproccess(img, input_size): #把img調成input_size並Return 新圖片&縮放比
    h, w = img.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    resized_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    padded_img = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    padded_img[:int(h * scale), :int(w * scale)] = resized_img
    img = padded_img.transpose(2, 0, 1)
    return img, scale

# 叫YOLO 偵測
def detect_with_manual_ratio(frame, model, exp):
    input_size = (640, 640)
    img, ratio = my_preproccess(frame, input_size)
    img_tensor = torch.from_numpy(img).unsqueeze(0).float()

    with torch.no_grad():
        outputs = model(img_tensor) 
        outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)

    if outputs[0] is not None:
        outputs = outputs[0].cpu().numpy()
        boxes = outputs[:, 0:4]
        scores = outputs[:, 4] * outputs[:, 5]
        classes = outputs[:, 6]  
        person_indices = np.where(classes == 0)[0]  #抓人!!!!抓人!!!抓人!!!抓人!!

        boxes = boxes[person_indices]
        scores = scores[person_indices]
        boxes /= ratio #還原回去
        return boxes, scores
    return [], []

# IOU
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxB_area = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    if boxA_area + boxB_area - inter_area == 0:
        return 0
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou

# 中央區域
def is_in_central_area(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2
    central_x_min = frame_width * (0.5-w_central_radius)
    central_x_max = frame_width * (0.5+w_central_radius)
    central_y_min = frame_height * (0.5-h_central_radius)
    central_y_max = frame_height * (0.5+h_central_radius)
    return central_x_min <= box_center_x <= central_x_max and central_y_min <= box_center_y <= central_y_max

#Hungarian algorithm
#最低成本配對
#input: 前一幀的bbox 當前幀的bbox
#output:(匹配結果,未被匹配到的前一幀物件,未被匹配到的當前幀物件) 資料結構=(list,list,list)
#reference:https://zh.wikipedia.org/zh-tw/%E5%8C%88%E7%89%99%E5%88%A9%E7%AE%97%E6%B3%95
def match_boxes(prev_boxes, curr_boxes, iou_threshold=global_iou_threshold):
    if len(prev_boxes) == 0:
        return [], list(range(len(curr_boxes)))
    #建立匈牙利演算法中的成本矩陣
    cost_matrix = np.zeros((len(prev_boxes), len(curr_boxes)))
    for i, prev_box in enumerate(prev_boxes):
        for j, curr_box in enumerate(curr_boxes):
            iou = calculate_iou(prev_box, curr_box)
            cost_matrix[i, j] = 1 - iou 
            #IoU 越高 成本越低 bbox更匹配
    #找總成本最低的配法
    row_ind, col_ind = linear_sum_assignment(cost_matrix) 
    
    matches, unmatched_prev = [], []
    unmatched_curr = list(range(len(curr_boxes)))

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < 1 - iou_threshold:
            matches.append((i, j))
            unmatched_curr.remove(j)
        else:
            unmatched_prev.append(i)

    unmatched_prev += [i for i in range(len(prev_boxes)) if i not in row_ind]
    return matches, unmatched_prev, unmatched_curr

# 畫
def draw_and_save(frame, boxes, ids, colors, person_count, trajectories, frame_id, speeds):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        if ids[i] not in colors:
            colors[ids[i]] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        color = colors[ids[i]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID {ids[i]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 軌跡
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        if ids[i] not in trajectories:
            trajectories[ids[i]] = []
        trajectories[ids[i]].append((center_x, center_y))

        for j in range(1, len(trajectories[ids[i]])):
            start_point = trajectories[ids[i]][j - 1]
            end_point = trajectories[ids[i]][j]
            color_intensity = max(0, 255 - j * 10)  # 深淺變化
            cv2.line(frame, start_point, end_point, (color[0], color[1], color_intensity), 2)
        
        # 5幀的平均速度
        if len(trajectories[ids[i]]) > 5:
            distances = [
                np.sqrt((trajectories[ids[i]][k][0] - trajectories[ids[i]][k - 1][0]) ** 2 +
                        (trajectories[ids[i]][k][1] - trajectories[ids[i]][k - 1][1]) ** 2)
                for k in range(-1, -6, -1)
            ]
            speed = np.mean(distances)   
            speeds[ids[i]] = speed
        if ids[i] in speeds:
            cv2.putText(frame, f'{speeds[ids[i]]:.2f}', (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f'Total People: {person_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

import numpy as np

# 平均移動方向
def calculate_direction_vector1(center1, center2,frameNum):
    if center1 is None or center2 is None:
        return np.array([0, 0])  
    if len(center1) != 2 or len(center2) != 2:
        print(f"Warning: Invalid center format: center1={center1}, center2={center2}")
        return np.array([0, 0])  
    if frameNum <= 0:  
        print(f"Warning: Invalid frame_difference={frameNum}. Returning zero vector.")
        return np.array([0.0, 0.0])

    return np.array([(center2[0] - center1[0])/frameNum, (center2[1] - center1[1])/frameNum])



# 主程式
def main(video_path, output_path, model_path, frame_output_folder):


    model, exp = init_yolox(model_path)
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    colors = {}
    saved_frames = 0
    total_people_set = set()
    disappeared_objects = defaultdict(lambda: {'box': None, 'frames': 0, 'direction': None})
    prev_boxes, prev_ids = [], []
    next_object_id = 0
    trajectories = defaultdict(list)
    speeds = {}

    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_height, frame_width = frame.shape[:2]
        curr_boxes, scores = detect_with_manual_ratio(frame, model, exp)
        curr_boxes = [box for i, box in enumerate(curr_boxes) if scores[i] > yoloXs_score_threshold]
        
        # 配對
        if frame_id > 0:
            matches, unmatched_prev, unmatched_curr = match_boxes(prev_boxes, curr_boxes)
        else:
            matches, unmatched_prev, unmatched_curr = [], list(range(len(prev_boxes))), list(range(len(curr_boxes)))
            print(f"distance theshold:{np.sqrt((frame_width/disappear_ratio)**2+(frame_height/disappear_ratio)**2)}")

        updated_tracks = {}
        for prev_idx, curr_idx in matches:
            object_id = prev_ids[prev_idx]
            disappeared_objects.pop(object_id, None)
            updated_tracks[curr_idx] = object_id

        # 未隊到的前一幀中的bbox(消失物件)
        for prev_idx in unmatched_prev:
            box = prev_boxes[prev_idx]
            if is_in_central_area(box, frame_width, frame_height):
                # 確認他生前的軌跡
                print(f"Frame {frame_id}: Checking trajectory for ID {prev_ids[prev_idx]}")
                #print(f"Trajectory: {trajectories[prev_ids[prev_idx]]}")

                if len(trajectories[prev_ids[prev_idx]]) > 1:
                    
                    prev_box_1 = trajectories[prev_ids[prev_idx]][-1]
                    prev_box_2 = trajectories[prev_ids[prev_idx]][-3] if len(trajectories[prev_ids[prev_idx]]) > 3 else None

                    searchindex = -3
                    while prev_box_2 is None or prev_box_1 == prev_box_2:
                        searchindex -= 1
                        if abs(searchindex) <= len(trajectories[prev_ids[prev_idx]]):  
                            prev_box_2 = trajectories[prev_ids[prev_idx]][searchindex]
                        else:
                            break  
                    print(f"Disappearing Object {prev_ids[prev_idx]} Calculating direction vector with: {prev_box_1}, {prev_box_2}")
                    prev_direction = calculate_direction_vector1(prev_box_1, prev_box_2,abs(searchindex))
                else:
                    prev_direction = np.array([0, 0])  
                disappeared_objects[prev_ids[prev_idx]] = {
            'box': box, 
            'frames': frame_id,
            'direction': prev_direction  # 確認他生前的軌跡
        }

                print(f"Frame {frame_id}: \033[31m Object {prev_ids[prev_idx]} added to disappeared list \033[0m")
                print(f"Frame {frame_id}: Currently disappeared objects:")
                for obj_id, obj_info in disappeared_objects.items():
                    print(f"  ID {obj_id}, Box {obj_info['box']}, Frames Disappeared: {obj_info['frames']}")

        # 未配對的這一幀中的bbox(新出現物件)
        for idx in unmatched_curr:
            box = curr_boxes[idx]
            if is_in_central_area(box, frame_width, frame_height): #人不會從中間突然蹦出來
                candidates = []
                for obj_id, obj_info in disappeared_objects.items():
                    disappear_box = obj_info['box']
                    disappear_center = [(disappear_box[0] + disappear_box[2]) / 2,
                                (disappear_box[1] + disappear_box[3]) / 2]
                    disappear_frames = obj_info['frames']
                    prev_direction=obj_info['direction']
                    curr_center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                    frame_difference = frame_id - disappear_frames
                    if frame_difference==0:
                        frame_difference=1

                    new_direction = calculate_direction_vector1(curr_center, disappear_center, frame_difference)
                        # 距離
                    center_distance = np.sqrt(
                        ((box[0] + box[2]) / 2 - (disappear_box[0] + disappear_box[2]) / 2) ** 2 +
                        ((box[1] + box[3]) / 2 - (disappear_box[1] + disappear_box[3]) / 2) ** 2
                    )
                    direction_similarity=0
                    magnitude_similarity=0
                    if np.linalg.norm(prev_direction) > 0 and np.linalg.norm(new_direction) > 0:
                         # 方向相似性
                        direction_similarity = np.dot(prev_direction, new_direction) / (
        np.linalg.norm(prev_direction) * np.linalg.norm(new_direction)
    )
                        # 大小相似性
                        magnitude_similarity = min(
        np.linalg.norm(new_direction) / np.linalg.norm(prev_direction),
        np.linalg.norm(prev_direction) / np.linalg.norm(new_direction)
    )

                        similarity = direction_factor * direction_similarity + magnitude_factor * magnitude_similarity
                    else:
                        similarity = 0


                    print(f"Frame {frame_id}")
                    #只有距離小於長:畫面長/disappear_ratio  寬:畫面寬/disappear_ratio 的長方形ㄉ對角線的才讓它進來
                    if center_distance<np.sqrt((frame_width/disappear_ratio)**2+(frame_height/disappear_ratio)**2):
                        print(f"  selecting...ID {obj_id}, direction_similarity={direction_similarity}, magnitude_similarity={magnitude_similarity},\n  Similarity: {similarity}, Distance: {center_distance},\n  added to candidates list")
                        candidates.append((obj_id, similarity, center_distance))
                    else:
                        print(f"  selecting...ID {obj_id}, direction_similarity={direction_similarity}, magnitude_similarity={magnitude_similarity},\n  Similarity: {similarity}, Distance: {center_distance}")


                if candidates:
                    candidates.sort(key=lambda x: x[2])  # 以距離 sort
                    
                    for i, candidate in enumerate(candidates):
                        # 根據距離排名分配權重
                        weight_similarity = 1.0 - distance_decay_factor * i  # 距離越大權重越小，遞減幅度 distance_decay_factor
                        weight_similarity = max(0, weight_similarity) 
                        
                        candidates[i] = (
                            candidate[0],  
                            candidate[1] + weight_similarity * distance_factor,  # similarity=direction_factor*方向相似性[-1 1] +magnitude_factor*大小相似性[0 1] +distance_factor*距離相似性[0 1]
                            candidate[2]  
                        )
                    max_similarity = max(c[1] for c in candidates)

                    #抓similarity最高 允許similarity只輸不到similarity_threshold的加入競爭 
                    filtered_candidates = [
                        c for c in candidates if abs(c[1] - max_similarity) <= similarity_threshold
                    ]

                    #挑距離最近的
                    if filtered_candidates:
                        best_match = min(filtered_candidates, key=lambda x: x[2]) 
                        best_match_id = best_match[0]
                        print(f"Frame {frame_id}: \033[34m Object {best_match_id} reappeared and matched to box {box},direction {new_direction}\033[0m")
                        print(f"Frame {frame_id}: Candidates for box {idx}:")
                        for candidate in filtered_candidates:
                            print(f"  ID {candidate[0]}, Similarity: {candidate[1]:.2f}, Distance: {candidate[2]:.2f}")
                        updated_tracks[idx] = best_match_id
                        disappeared_objects.pop(best_match_id, None)
                    else:
                        updated_tracks[idx] = next_object_id #判斷這個是新物件
                        next_object_id += 1
                else:
                    updated_tracks[idx] = next_object_id #判斷這個是新物件
                    next_object_id += 1
            else:
                updated_tracks[idx] = next_object_id #判斷這個是新物件
                next_object_id += 1

        ids = [updated_tracks[i] for i in range(len(curr_boxes))]
        person_count = len(total_people_set)
        total_people_set.update(ids)
        #update() 是集合的方法，用來將列表 ids 中的每個元素依次添加到集合 total_people_set 中。
        #如果 ids 中有元素已經存在於 total_people_set，則不會重複添加，保證集合的唯一性。
        #by chatGPT

        frame = draw_and_save(frame, curr_boxes, ids, colors, person_count, trajectories, frame_id, speeds)
        out.write(frame)

        if saved_frames < 10:
            mid_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
            if abs(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - mid_frame) < 5:
                cv2.imwrite(f"{frame_output_folder}/frame_{saved_frames}.jpg", frame)
                saved_frames += 1

        prev_boxes, prev_ids = curr_boxes, ids
        frame_id += 1

    cap.release()
    out.release()



if __name__ == "__main__":
    model_path = r'C:\Users\Lemonsky0618\YOLOX\yolox_s.pth'
    frame_output_folder = r'D:\0-IOC-1\VIDEO STREAM\HW3\HW3\output_frames'


    #video_path = r'D:\0-IOC-1\VIDEO STREAM\HW3\HW3\video\easy_9.mp4'
    #output_path = r'D:\0-IOC-1\VIDEO STREAM\HW3\HW3\result\easy_output.mp4'
    #main(video_path, output_path, model_path, frame_output_folder)

    print("--------------------case2--------------------")
    video_path = r'D:\0-IOC-1\VIDEO STREAM\HW3\HW3\video\demo.mp4'
    output_path = r'D:\0-IOC-1\VIDEO STREAM\HW3\HW3\result\demo.mp4'
    main(video_path, output_path, model_path, frame_output_folder)
