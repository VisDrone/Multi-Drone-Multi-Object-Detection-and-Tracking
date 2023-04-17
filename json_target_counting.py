import json
import os

input_file_dir = "./json_resultfiles2/multiDrone_Localmatching-NMS/NMS-bytetrack_autoassign_full_mdmt-private-half"
output_filr_dir = "./target_counting_dir"

num_target = 0
frame = 0
for file in os.listdir(input_file_dir):
    num_target_perjson = 0
    file_dir = os.path.join(input_file_dir, file)
    file_name = str(file_dir)
    print(file_name)
    if "txt" in file_name or "ipynb" in file_name:
        continue
    with open(file_dir) as json_file:
        json_load = json.load(json_file)
        
        frame += len(json_load.keys())
        for key, value in json_load.items():
            # print(key)
            # print(len(value))
            num_target_perjson += len(value)
            num_target += len(value)
            # print(num_target)
    # with open ("{}/target_counting.txt".format(output_filr_dir), "a") as f:
        # f.write(file_name+"\n")
        # f.write(str(num_target_perjson)+"\n")
with open ("{}/target_counting.txt".format(output_filr_dir), "a") as f:
        f.write(file_name+"\n")
        f.write(str(num_target)+"\n")
        f.write("frames" + str(frame)+"\n")