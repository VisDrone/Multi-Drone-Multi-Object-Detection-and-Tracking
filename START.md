# Multi-Drone-Multi-Object-Detection-and-Tracking


## Installation
1.create a conda virtual environment and activate it:
    
    conda create -n mdmt python=3.8
    conda activate mdmt

2.install pytorch and suited mmcv-full, please refer to ![MMtracking](https://github.com/open-mmlab/mmtracking/blob/master/docs/en/install.md).

(if you have no idea which version to install, stay with ours:
torch                          1.10.0+cu113/
torchvision                    0.11.1+cu113/
mmcv-full                      1.5.0/
mmdet                          2.25.1)

3.download the codes and :
    
    cd Multi-Drone-Multi-Object-Detection-and-Tracking-main
    pip install -r requirements.txt
    cd mmdet
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    cd ..
    pip install -r requirements.txt
    pip install -v -e .  # or "python setup.py develop"


## Getting started
### Dataset Structure
> Multi-Drone-Multi-Object-Detection-and-Tracking-main
>> data
>>> MDMT
>>>> train/
>>>> val/
>>>> test/
>>>>> sequences
>>>>>> img1
>>>>>>> ~.jpg
### Inference
1.Inference MIA-Net:

    python ./demo/supplement_MIA.py
import arguments:

`--config` config file

`--input`  input data folder

`--xml_dir`  input xml file of the groundtruth
                    
`--result_dir`  the directory to save results, no "/" in the end

`--method`  the sub-directory used in result_dir, representing different methods

2.Inference MIA-Net(w/o supplementation), MIA-Net(w/ localmatching), MIA-Net(w/ globalmatching), run:
    
    python ./demo/multiDrone_matchingIDallocation-NMS.py
    python ./demo/multiDrone_localmatching-NMS.py
    python ./demo/multiDrone_globalmatching-NMS.py
### Evaluation
1.For tracking perforcement:

    python ./demo/eval/json_2_txt.py [--sequences_result] [--output_dir]
    python ./demo/eval/txttxt_test.py [--test_file_dir]

2.For MDA score:

    python ./demo/eval/mango_eval.py [--sequences_result]


<!-- ## FAQ
Please refer to ![FAQ](https://github.com/Edision-liu/Multi-Drone-Multi-Object-Detection-and-Tracking/edit/main/README.md) for frequently asked questions. -->


## Citation
    @article{liu2023robust,
        title={Robust Multi-Drone Multi-Target Tracking to Resolve Target Occlusion: A Benchmark},
        author={Liu, Zhihao and Shang, Yuanyuan and Li, Timing and Chen, Guanlin and Wang, Yu and Hu, Qinghua and Zhu, Pengfei},
        journal={IEEE Transactions on Multimedia},
        year={2023},
        publisher={IEEE}
    }
