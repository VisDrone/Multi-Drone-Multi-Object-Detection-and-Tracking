"""
@ author: Guanlin Chen
@ version_v2: 2022/07/11 22:12:00
@ description:   
    评估多机系统关联性能, AAS和IAS.
    1. For AAS:
        1) Input: 
            result.json:
                {"frame=0":[[id,x1,y1,x2,y2],[...]],
                "frame=1":[[...],[...]],...]]...}
            gt.txt:
                frame+1, id, x1, y1, w, h, _, _, _
                ...
        2) Output: AAS:  0<= float.64 <= 1
    2. For IAS:

"""

import os
import json
import csv
import pandas as pd

# 存储每一帧的所有objects
class frameObject():
    def __init__(self, frame, resultA, resultB):
        self.frame = int(frame)
        self.resultA = resultA
        self.resultB = resultB
        self.gtA = []
        self.gtB = []


# groundtruth 对象类
class GTTarget():
    def __init__(self, lineid, frame, camera, ID, x1, y1, x2, y2):
        self.lineid = int(lineid)
        self.frame = int(frame)
        self.camera = int(camera)
        self.ID = int(ID)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.interIndex = -1   # 类内关联


# 双机追踪结果对象类
class MangoTarget(GTTarget): 
    def __init__(self, lineid, frame, camera, ID, x1, y1, x2, y2):
        super().__init__(lineid, frame, camera, ID, x1, y1, x2, y2)
        self.corssIndex = -1            # 类间关联，未关联为-1


# 读取gt文件构建gt对象
def read_gt(gt_path, camera):
    sequence_gt_targets = []
    with open(gt_path,"r") as f:
        tmplines = f.readlines()
    lines = sorted(tmplines, key=lambda x: (x[1], x[0]))      # 按照frame排升序
    for i, line in enumerate(lines):
        attributes = line.split(",")
        targetObject = GTTarget(i, int(attributes[0])-1, camera, attributes[1], attributes[2], attributes[3], int(attributes[2])+int(attributes[4]), int(attributes[3])+int(attributes[5]))
        sequence_gt_targets.append(targetObject)

    return sequence_gt_targets

# result按照以frame为key读取
def read_result(result_path_A, result_path_B):
    sequence_result_frames = []
    with open(result_path_A, 'r') as f_a:
        frames_A = json.load(f_a)
    with open(result_path_B, 'r') as f_b:
        frames_B = json.load(f_b)
    for i, (frame_A, frame_B) in enumerate(zip(frames_A, frames_B)):
        list_A = []
        list_B = []
        for j, target in enumerate(frames_A[frame_A]):
            tempTarget = MangoTarget(i*100+10+j, i, 1, int(target[0]), float(target[1]), float(target[2]), float(target[3]), float(target[4]))
            list_A.append(tempTarget)
        for j, target in enumerate(frames_B[frame_B]):
            tempTarget = MangoTarget(i*100+20+j, i, 2, target[0], target[1], target[2], target[3], target[4])
            list_B.append(tempTarget)

        currentFrame = frameObject(i, list_A, list_B)
        sequence_result_frames.append(currentFrame)

    return sequence_result_frames

# 将gt匹配到frame中
def assign_gt_to_frames(frame_objects, gt_targets_A, gt_targets_B):
    for i, target_A in enumerate(gt_targets_A):
        frameid = target_A.frame
        frame_objects[frameid].gtA.append(target_A)
    for i, target_B in enumerate(gt_targets_B):
        frameid = target_B.frame
        frame_objects[frameid].gtB.append(target_B)

    return frame_objects

# 计算IOU
def calIoU(x1, y1, x2, y2, x_1, y_1, x_2, y_2):

    endx = max(x2, x_2)
    startx = min(x1, x_1)
    width = x2-x1 + x_2-x_1 - (endx - startx)

    endy = max(y2, y_2)
    starty = min(y1, y_1)
    height = y2-y1 + y_2-y_1 - (endy - starty)

    # 如果重叠部分为负, 即不重叠
    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = (x2-x1) * (y2-y1)
        Area2 = (x_2-x_1) * (y_2-y_1)
        ratio = Area * 1.0 / (Area1 + Area2 - Area)

    return ratio


# 计算多机指标AAS
def calAAS(frame_objects):
    frame_AAS = 0
    se_MDA = []
    se_accu_MDA = []
    # 逐帧计算关联序列对
    for i, frame in enumerate(frame_objects):
        count_GA = 0
        count_RA = 0
        count_TA = 0
        count_MA = 0
        current_resultA = frame_objects[i].resultA
        current_resultB = frame_objects[i].resultB
        current_gtA = frame_objects[i].gtA
        current_gtB = frame_objects[i].gtB

        
        for j, targetA in enumerate(current_resultA):    # result类内关联
            for k, targetB in enumerate(current_resultB):
                if targetA.ID == targetB.ID:
                    targetA.interIndex = k
                    targetB.interIndex = j
                    count_RA += 1

        for j, gtA in enumerate(current_gtA):      # gt类内关联
            for k, gtB in enumerate(current_gtB):
                if gtA.ID == gtB.ID:
                    gtA.interIndex = k
                    gtB.interIndex = j
                    count_GA += 1


        for j, targetA in enumerate(current_resultA):   # 类间关联
            for k, gtA in enumerate(current_gtA):
                IoU = calIoU(targetA.x1, targetA.y1,targetA.x2,targetA.y2,gtA.x1,gtA.y1,gtA.x2,gtA.y2)
                if IoU >=0.5 and (targetA.interIndex != -1):    # 是比配对，且与gt匹配
                    targetA.corssIndex = k
                    targetB = current_resultB[targetA.interIndex]
                    gtB = current_gtB[gtA.interIndex]
                    IoU_pair = calIoU(targetB.x1, targetB.y1,targetB.x2,targetB.y2,gtB.x1,gtB.y1,gtB.x2,gtB.y2)
                    if IoU_pair >= 0.5:
                        count_TA += 1
                    
        count_FA = count_RA - count_TA
        count_MA = count_GA - count_TA
        image_MDA = count_TA/(count_GA + count_FA + count_MA)
        if image_MDA > 1:
            image_MDA = 1

        # print(image_MDA)
        se_MDA.append(image_MDA)
        frame_AAS += count_TA/(count_GA + count_FA + count_MA)
        se_accu_MDA.append(frame_AAS/(i+1))
    
    out_AAS = frame_AAS/len(frame_objects)
    print("AAS: ",out_AAS)

    return out_AAS, se_MDA, se_accu_MDA


if __name__ == "__main__":
    json_root_dir = "/root/mmtracking-master/reid_json_resultfiles/carafe-sbs_R50-ibn-model_final(thres-45)/"

    sequences_result = os.listdir(json_root_dir)
    sequences_result.sort()
    if '.ipynb_checkpoints' in sequences_result:
        sequences_result.remove('.ipynb_checkpoints')
    
    sequences_gt = os.listdir("./demo/eval/test/")
    sequences_gt.remove('.ipynb_checkpoints')
    sequences_gt.sort()
    
    result_dict = {}
    out_AAS = 0
    sequences = 0
    i = 0
    while(i<len(sequences_result)):
        # print(sequences_result)
        # print(i, sequences_result)
        if ".ipynb_checkpoints" == str(sequences_result[i]):
            print(i, sequences_result)
            i = i + 1
            continue
        
        if ".txt" in str(sequences_result[i]):
            i = i+1
            continue
        result_path_A = json_root_dir + sequences_result[i]
        result_path_B = json_root_dir + sequences_result[i+1]
        gt_path_A = "./demo/eval/test/" + sequences_result[i].split(".")[0] + ".txt"
        gt_path_B = "./demo/eval/test/" + sequences_result[i+1].split(".")[0] + ".txt"

        frame_objects = read_result(result_path_A, result_path_B)
        gt_targets_A = read_gt(gt_path_A, 1)
        gt_targets_B = read_gt(gt_path_B, 2)
        
        frame_objects = assign_gt_to_frames(frame_objects, gt_targets_A, gt_targets_B)    # 完成以帧为key的数据结构构建

        print(sequences_result[i], 'and', sequences_result[i+1])
        out_AAS_, se_MDA, se_accu_MDA = calAAS(frame_objects)
        result_dict["{}".format(str("seq_" + sequences_result[i].replace("-1.json", "")))] = se_MDA
        # result_dict["{}".format(str("seq_" + sequences_result[i].replace("-1.json", "")))] = se_accu_MDA
        out_AAS += out_AAS_
        sequences += 1
        i += 2

    print("Total AAS: ", out_AAS/sequences)
    # result_dict = {"frames":[0,1,2,3], "seq_22":[2,3,4,5], "seq_23":[2,3,4,5]}
    max_len = max(len(value) for key, value in result_dict.items())
    for key, value in result_dict.items():
        value.extend(["nan" for _ in range(max_len-len(value))])    #长度必须保持一致，否则报错

    dataframe = pd.DataFrame(result_dict)
    dataframe.to_csv(r"./REID-carafe-sbs_R50-ibn-model_final(thres-45).csv",sep=',') # accuMDA_
#     import pandas as pd
#     #a和b的长度必须保持一致，否则报错
#     a = [x for x in range(5)]
#     b = [x for x in range(5,10)]
#     #字典中的key值即为csv中列名
#     dataframe = pd.DataFrame({'a_name':a,'b_name':b})

#     #将DataFrame存储为csv,index表示是否显示行名，default=True
#     dataframe.to_csv(r"G:\A1大论文内容\实验\test.csv",sep=',')





  