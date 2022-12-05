# Multi-Drone-Multi-Object-Detection-and-Tracking

Robust Multi-Drone Multi-Target Tracking to Resolve Target Occlusion: A Benchmark

## Abstract 

Multi-drone multi-target tracking aims at collaboratively detecting and tracking targets across multiple drones and associating the identities of objects from different drones, which can overcome the shortcomings of single-drone object tracking.
To address the critical challenges of identity association and target occlusion in multi-drone multi-target tracking tasks, we collect an occlusion-aware multi-drone multi-target tracking dataset named MDMT. It contains 88 video sequences with 39,678 frames, including 11,454 different IDs of persons, bicycles, and cars. 
The MDMT dataset comprises 2,204,620 bounding boxes, of which \textcolor{red}{543,444} bounding boxes contain target occlusions. 
We also design a multi-device target association score (MDA) as the evaluation criteria for the ability of cross-view target association in multi-device tracking.
Furthermore, we propose a Multi-matching Identity Authentication network (MIA-Net) for the multi-drone multi-target tracking task.
The local-global matching algorithm in MIA-Net discovers the topological relationship of targets across drones, efficiently solves the problem of cross-drone association, and also effectively complements occluded targets with the advantage of multiple drone view mapping. Extensive experiments on the MDMT dataset validate the effectiveness of our proposed MIA-Net for the task of identity association and multi-object tracking with occlusions.

![VisDrone](https://github.com/VisDrone/Multi-Drone-Multi-Object-Detection-and-Tracking/blob/main/MDMT.png)


## Dataset

To address the critical challenges of identity association and target occlusion in multi-drone multi-target tracking tasks, we collect an occlusion-aware multi-drone multi-target tracking dataset named MDMT. It contains 88 video sequences with 39,678 frames, including 11,454 different IDs of persons, bicycles, and cars.

![VisDrone](https://github.com/VisDrone/Multi-Drone-Multi-Object-Detection-and-Tracking/blob/main/occluded_pic.png)


MDMT-FULL(25.8G): [Baidu Netdisk](https://pan.baidu.com/s/1Zkp9jrGSHxATFstUAkhs-w?pwd=9un6)


## Multi-matching Identity Authentication network (MIA-Net)

MIA-Net (code)


## Citation 
Please cite this paper if you want to use it in your work.



