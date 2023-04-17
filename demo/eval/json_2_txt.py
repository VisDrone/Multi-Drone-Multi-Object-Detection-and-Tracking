# _*_ coding: utf-8 _*_
# @Time    :2022/7/22 14:15
# @Author  :LiuZhihao
# @File    :json_2_txt.py


import json
from matplotlib.pyplot import axis

from yaml import load
import os


parser == ArgumentParser()
parser.add_argument('--sequences_result', default="./json_resultfiles/Firstframe_initialized_faster_rcnn_r50_fpn_carafe_1x_full_mdmt/", help="input file directory")
parser.add_argument('--output_dir', default="./demo/txt/MOT/Firstframe_initialized_faster_rcnn_r50_fpn_carafe_1x_full_mdmt/", help="input file directory")
args = parser.parse_args()

input_dir = args.sequences_result
# input_dir = "./json_resultfiles/Firstframe_initialized_bytetrack_autoassign_full_mdmt-private-half/"


json_files_dirs = os.listdir(input_dir)

for file in json_files_dirs:
    print(file)
    if "txt" in file or "ipynb" in file:
        continue
    # if file not in ["26-1.json", "26-2.json"]:
    #     continue
    json_dir = os.path.join(input_dir, file)
    sequence = json_dir.split("/")[-1]
    sequence = sequence.split(".")[0]
    # print(sequence)
    # print(json_dir)
    n = 0
    maxID = 0
    with open(json_dir) as f:
        load_json = json.load(f)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        with open("{0}/{1}.txt".format(args.output_dir, sequence), "w") as f_txt:
            # 两个循环分别遍历字典的键值对
            for (key, values) in load_json.items():
                for value in values:
                    maxID = max(maxID, value[0])
                    print(maxID)
            while n <= maxID:
                for (key, values) in load_json.items():
                    # print(key)
                    frame = key.split("=")[-1]
                    # 先找ID=1的
                    for value in values:
                        if value[0] == n:
                            string_line = "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(int(frame), int(value[0]),
                                                                                         value[1], value[2],
                                                                                         value[3] - value[1],
                                                                                         value[4] - value[2], int(1),
                                                                                         int(1), int(1))
                            f_txt.write(string_line)

                    # if key == "frame={}".format(n):
                    #     print("bigin")

                    # for value in values:
                    #     # print(value)
                    #     frame = key.split("=")[-1]
                    #     string_line = "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(frame, value[0], value[1], value[2], value[3], value[4], int(1), int(3), int(1))
                    #     f_txt.write(string_line)
                n += 1




