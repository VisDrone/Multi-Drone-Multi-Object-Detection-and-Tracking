# Multi-UAV Collaborative Detection and Tracking 2.0     

This project is an upgraded iteration of version 1.0, with the following main improvements:     
1. **Added target detection training code** (parameters maximally reference version 1.0)     
2. **Fixed known issues from version 1.0**     
3. **Provided complete environment configuration**     
4. **Optimized system integration process**     

## Environment Preparation     
Please complete the following steps in sequence:     

### Project Cloning     
Download the following repositories to local folder `2025_05_MMDet_MMTrack_MDMT` (can be customized):         

[//]: # (MMDetection)    
- MMDetection（v2.22.0）       
  [![MMDetection](https://img.shields.io/badge/MMDetection-v2.22.0-blue)](https://github.com/open-mmlab/mmdetection/tree/v2.22.0)     

[//]: # (MDMT)    
- MDMT       
  [![MDMT](https://img.shields.io/badge/MDMT-Main-green)](https://github.com/VisDrone/Multi-Drone-Multi-Object-Detection-and-Tracking)     

#### Directory Structure     
```text
├── 2025_05_MMDet_MMTrack_MDMT  # Can be named by yourself  
│   ├── datasets  # Store all datasets (prepare in advance)    
│   │   └── MDMT_dataset      
│   │   │   ├── COCO_1  # COCO-format dataset for View 1      
│   │   │   │   ├── test  # Test set    
│   │   │   │   ├── train  # Training set    
│   │   │   │   ├── trainval  # Training-validation set    
│   │   │   │   ├── val  # Validation set    
│   │   │   │   ├── test.json  # Test set annotations
│   │   │   │   ├── train.json  # Training set annotations
│   │   │   │   ├── trainval.json  # Trainval annotations
│   │   │   │   └── val.json  # Validation set annotations
│   │   │   └── COCO_2  # COCO-format dataset for View 2 (same as View 1)          
│   ├── mmdetection-2.22.0    
│   ├── Multi-Drone-Multi-Object-Detection-and-Tracking-main    
│   ├── mmdetection-2.22.0.zip    
│   └── Multi-Drone-Multi-Object-Detection-and-Tracking-main.zip    
```

## MMDetection Environment Configuration
### NVIDIA GeForce RTX 3090    
```bash
# Create Python 3.9 virtual environment
conda create --name 2025_05_MMDet python=3.9 -y
conda activate 2025_05_MMDet

# Install PyTorch with CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install specific version of setuptools with 50s timeout (for unstable networks)
pip --default-timeout=50 install setuptools==50.3.2
# Install mmcv-full 1.5.0 compatible with PyTorch 1.11 + CUDA 11.3
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html

# Compile MMDetection
cd ./2025_05_MMDet_MMTrack_MDMT/mmdetection-2.22.0/
pip --default-timeout=50 install cython==3.0.12
pip --default-timeout=50 install -r requirements/build.txt
pip --default-timeout=50 install -v -e .

# Install enhancement components
pip --default-timeout=50 install instaboostfast
pip --default-timeout=200 install git+https://github.com/cocodataset/panopticapi.git
pip --default-timeout=300 install git+https://github.com/lvis-dataset/lvis-api.git
pip --default-timeout=200 install -r requirements/albu.txt

# Downgrade numpy library (2.0.2 → 1.26.4)
pip --default-timeout=100 install numpy==1.26.4

[//]: # (conda deactivate)
[//]: # (conda remove -n 2025_05_MMDet --all)
```

## Key Configuration Modifications
### Disable Configuration Validation
Comment line 502 in mmcv/utils/config.py (example path) in your conda:

```python
# text, _ = FormatCode(text, style_config=yapf_style, verify=True)
```

### Class Configuration Update
Modify class definitions in these files:  

mmdet/datasets/coco.py (line 25/40)    

```python
CLASSES = ("person", "car", "bicycle")    
PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142)]    
```

mmdet/core/evaluation/class_names.py (lines 68-83)    

```python
def coco_classes():
    """Class names of COCO."""
    return [
        "person", "car", "bicycle"
    ] 
```


## Model Training Configuration
### Configuration File Adjustments     
#### 1.Object Detection Configuration File Modifications    
**File Locations**:`2025_05_MMDet_MMTrack_MDMT/mmdetection-2.22.0/configs/`  
**Target Files**:`COCO_1.py` and `COCO_2.py`(corresponding to training configurations for view 1/2 object detection respectively)

**Key Configuration Items**(using COCO_1.py as example):
```python
# Data path configuration (recommended to use absolute paths)
data_root = '/root/2025_05_MMDet_MMTrack_MDMT/datasets/MDMT_dataset/COCO_1'  # Line 118: Dataset root directory

# Training set configuration
ann_file=data_root + '/train.json',   # 第162行:标注文件路径
img_prefix=data_root + '/train',      # 第163行:图像存储路径

# Validation set configuration
ann_file=data_root + '/val.json',     # Line 180
img_prefix=data_root + '/val',        # Line 181

# Test set configuration
ann_file=data_root + '/test.json',    # Line 203
img_prefix=data_root + '/test',       # Line 204

# Model save path
work_dir = '/root/2025_05_MMDet_MMTrack_MDMT/mmdetection-2.22.0/wordir/COCO_1'         # Line 244
```

#### 2.Training Script Configuration Adjustments
**Target Files**:`train1.py` and `train2.py`

**Key Parameters**（using train1.py as example）:
```python
from argparse import ArgumentParser
parser = ArgumentParser()
# Training configuration file path
parser.add_argument('--config', default='/root/2025_05_MMDet_MMTrack_MDMT/mmdetection-2.22.0/configs/COCO_1.py')   # Line 24
# Model output directory
parser.add_argument('--work-dir', default='/root/2025_05_MMDet_MMTrack_MDMT/mmdetection-2.22.0/wordir/COCO_1')   # Line 25
# Epoch settings (example shows only 1 epoch)
epochs = 1  # Line 233
```

#### 3.Training Execution Process    
```bash
# Execute view 1/2 training sequentially (parallel execution may cause GPU conflicts)
# Train view 1 object detection model  
python ./tools/train1.py     
# Train view 2 object detection model    
python ./tools/train2.py    
```

### Model Training Results
#### View 1 Detection Performance (1 epoch)
```text
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.184
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.313
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.197
……
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.090
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.333
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.578  
```

#### View 2 Detection Performance (1 epoch)
```text
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.118
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.219
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.120
……
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.085
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.206
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.661
```

### Model File Processing
Locate the view 1 object detection model **epoch_1.pth** in /root/2025_05_MMDet_MMTrack_MDMT/work_dirs/**COCO_1**, rename it to **COCO_1.pth**    
Locate the view 2 object detection model **epoch_1.pth** in /root/2025_05_MMDet_MMTrack_MDMT/work_dirs/**COCO_2**, rename it to **COCO_2.pth**    

--------------------------------------------------------------------------------------------

## MDMT Environment Configuration
### NVIDIA GeForce RTX 1660Ti    
```bash
# Create Python 3.8 virtual environment 
conda create -n 2025_05_MDMT python=3.8 -y
conda activate 2025_05_MDMT

# Install PyTorch suite (CUDA 10.2 version) 
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch

# Enter project directory  
cd ./2025_05_MMDet_MMTrack_MDMT/Multi-Drone-Multi-Object-Detection-and-Tracking-main

# Install full MMCV (must strictly match PyTorch and CUDA versions, here for CUDA10.2+PyTorch1.10)  
pip --default-timeout=100 install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html

# Install MMDetection object detection framework (version must be compatible with MMCV) 
pip --default-timeout=100 install mmdet==2.25.1
# Install data processing library pandas (specify v1.3.5 for compatibility)  
pip --default-timeout=100 install pandas==1.3.5
# Install scientific computing library scipy (large dependency, set longer timeout)  
pip --default-timeout=200 install scipy==1.7.3
# Install documentation testing tool  
pip --default-timeout=100 install xdoctest==0.10.0
# Install MMTracking multi-object tracking framework (ensure compatibility with MMDet version)  
pip --default-timeout=100 install mmtrack==0.14.0

# Install all Python packages from project requirements  
pip install -r requirements.txt
# Install current project in development mode  
python setup.py develop

# Install OpenCV extension library  
pip install --default-timeout=100 opencv-contrib-python==4.9.0.80
# Install numerical computing library numpy  
pip --default-timeout=100 install numpy==1.23.0
# Install JIT compiler numba  
pip --default-timeout=100 install numba==0.58.1

[//]: # (conda deactivate)
[//]: # (conda remove -n 2025_05_MDMT --all)
```

## System Configuration Instructions    
### 1. Add and Modify Configuration Files    
**Path**: ./Multi-Drone-Multi-Object-Detection-and-Tracking-main/configs/mot/bytetrack        
**Operations**:    
- Add new configuration files: `COCO_1.py` and  `COCO_2.py`    
- Modify pretrained weights path (Line 47):          
```python
# COCO_1.py
'./mmdetection-2.22.0/wordir/COCO_1/COCO_1.pth'     
# COCO_2.py  
'./mmdetection-2.22.0/wordir/COCO_2/COCO_2.pth'      
```
    
### 2. Fix Module Import Error     
**Path**: .//Multi-Drone-Multi-Object-Detection-and-Tracking-main/demo/utils/common.py    
**Operations**: 
- Comment out Line 10 import statement:    
```python
# Comment conflicting import
# from matching_pure import matching, calculate_cent_corner_pst  
```
    
### 3. Replace MMTrack Library Files        
Locate the target environment directory: **D:\anaconda3\envs\2025_05_MDMT\Lib\site-packages\mmtrack**         
- Replace the following modules with files from the project directory:    
```text
- apis/inference.py     
- models/mot/byte_track.py     
- models/trackers/byte_tracker.py   
```
  
## System Execution
### Inference & Evaluation
```bash
# Execute MIA network inference
python ./demo/supplement_MIA.py
# Format conversion (JSON -> TXT)
python ./demo/eval/json_2_txt.py
# Single-camera tracking evaluation
python ./demo/eval/txttxt_test.py
# Multi-camera matching evaluation
python ./demo/eval/mango_eval_MDA_perImage.py
```
 
## Partial Results Demonstration    
```bash
# Multi-camera matching evaluation
python ./demo/eval/mango_eval_MDA_perImage.py
``` 
```text
Results are consistent with TABLE VIII in the paper

26-1.json and 26-2.json
AAS:  0.31944363984225205
31-1.json and 31-2.json
AAS:  0.3032700669782865
……
71-1.json and 71-2.json
AAS:  0.22560263435576106
73-1.json and 73-2.json
AAS:  0.2914553132623469
Total AAS:  0.407408623458025
``` 

