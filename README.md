#### 这一版本是对第一个版本的补充（使用MMDetection进行目标检测的训练，COCO格式的数据集，环境配置等）
#### 使用时，请分别解压到2025_05_MMDet_MMTrack_MDMT文件夹下。

# 多无人机协同检测与跟踪 2.0     

本项目为 1.0 版本的升级迭代，主要改进如下：     
1. **新增目标检测训练代码**（参数设置最大程度参考 1.0 版本）     
2. **修复 1.0 版本已知问题**     
3. **提供完整环境配置**     
4. **优化系统集成流程**     

## 环境准备     
请按顺序完成以下步骤：     

### 项目克隆     
访问以下仓库并下载到本地文件夹 `2025_05_MMDet_MMTrack_MDMT` 中（可自行命名）：         

[//]: # (MMDetection下载网址)    
- MMDetection（v2.22.0）       
  [![MMDetection](https://img.shields.io/badge/MMDetection-v2.22.0-blue)](https://github.com/open-mmlab/mmdetection/tree/v2.22.0)     

[//]: # (MDMT下载网址)    
- MDMT       
  [![MDMT](https://img.shields.io/badge/MDMT-Main-green)](https://github.com/VisDrone/Multi-Drone-Multi-Object-Detection-and-Tracking)     

#### 目录结构     
```text
├── 2025_05_MMDet_MMTrack_MDMT  # 可自行命名  
│   ├── datasets  # 存放所有数据，需要提前准备    
│   │   └── MDMT_dataset      
│   │   │   ├── COCO_1  # 视角1的COCO格式数据集      
│   │   │   │   ├── test  # 测试集    
│   │   │   │   ├── train  # 训练集    
│   │   │   │   ├── trainval  # 训练验证集    
│   │   │   │   ├── val  # 验证集    
│   │   │   │   ├── test.json  # 测试集标注文件
│   │   │   │   ├── train.json  # 训练集标注文件
│   │   │   │   ├── trainval.json  # 训练验证集标注文件
│   │   │   │   └── val.json  # 验证集标注文件
│   │   │   └── COCO_2  # 视角2的COCO格式数据集，同视角1          
│   ├── mmdetection-2.22.0    
│   ├── Multi-Drone-Multi-Object-Detection-and-Tracking-main    
│   ├── mmdetection-2.22.0.zip    
│   └── Multi-Drone-Multi-Object-Detection-and-Tracking-main.zip    
```

## MMDetection 环境配置
### NVIDIA GeForce RTX 3090    
```bash
# 创建 Python 3.9 虚拟环境
conda create --name 2025_05_MMDet python=3.9 -y
conda activate 2025_05_MMDet

# 安装 PyTorch 套件（CUDA 11.3）
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# 安装指定版本的 setuptools 并设置超时时间为 50 秒（应对网络不稳定）
pip --default-timeout=50 install setuptools==50.3.2
# 安装与 PyTorch 1.11 + CUDA 11.3 兼容的 mmcv-full 1.5.0 版本
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html

# 编译 MMDetection
cd ./2025_05_MMDet_MMTrack_MDMT/mmdetection-2.22.0/
pip --default-timeout=50 install cython==3.0.12
pip --default-timeout=50 install -r requirements/build.txt
pip --default-timeout=50 install -v -e .

# 安装增强组件
pip --default-timeout=50 install instaboostfast
pip --default-timeout=200 install git+https://github.com/cocodataset/panopticapi.git
pip --default-timeout=300 install git+https://github.com/lvis-dataset/lvis-api.git
pip --default-timeout=200 install -r requirements/albu.txt

# 安装低版本的 numpy 库（2.0.2 → 1.26.4 版本）
pip --default-timeout=100 install numpy==1.26.4

[//]: # (conda deactivate)
[//]: # (conda remove -n 2025_05_MMDet --all)
```

## 关键配置修改
### 禁用配置校验
在 conda 环境的 mmcv/utils/config.py 中注释第502行（路径示例）：

```python
# text, _ = FormatCode(text, style_config=yapf_style, verify=True)
```

### 类别配置更新
修改以下文件中的类别定义：    

mmdet/datasets/coco.py (第25/40行)    

```python
CLASSES = ("person", "car", "bicycle")    
PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142)]    
```

mmdet/core/evaluation/class_names.py (第68-83行)    

```python
def coco_classes():
    """Class names of COCO."""
    return [
        "person", "car", "bicycle"
    ] 
```


## 模型训练配置
### 配置文件调整     
#### 1.目标检测配置文件修改    
**文件位置**：`2025_05_MMDet_MMTrack_MDMT/mmdetection-2.22.0/configs/`  
**操作文件**：`COCO_1.py` 和 `COCO_2.py`（分别对应视角1/2的目标检测训练配置）

**关键配置项说明**（以COCO_1.py为例）：
```python
# 数据路径配置（建议使用绝对路径）
data_root = '/root/2025_05_MMDet_MMTrack_MDMT/datasets/MDMT_dataset/COCO_1'  # 第118行：数据集根目录

# 训练集配置
ann_file=data_root + '/train.json',   # 第162行：标注文件路径
img_prefix=data_root + '/train',      # 第163行：图像存储路径

# 验证集配置
ann_file=data_root + '/val.json',     # 第180行
img_prefix=data_root + '/val',        # 第181行

# 测试集配置
ann_file=data_root + '/test.json',    # 第203行
img_prefix=data_root + '/test',       # 第204行

# 模型保存路径
work_dir = '/root/2025_05_MMDet_MMTrack_MDMT/mmdetection-2.22.0/wordir/COCO_1'         # 第244行
```

#### 2.训练脚本配置调整
**操作文件**：`train1.py` 和 `train2.py`

**关键参数说明**（以train1.py为例）：
```python
from argparse import ArgumentParser
parser = ArgumentParser()
# 训练配置文件路径
parser.add_argument('--config', default='/root/2025_05_MMDet_MMTrack_MDMT/mmdetection-2.22.0/configs/COCO_1.py')  # 第24行
# 模型输出目录
parser.add_argument('--work-dir', default='/root/2025_05_MMDet_MMTrack_MDMT/mmdetection-2.22.0/wordir/COCO_1')   # 第25行
# 训练轮次设置（示例仅训练1轮）
epochs = 1  # 第233行
```

#### 3.训练执行流程    
```bash
# 顺序执行视角1/2的训练（并行执行会导致GPU冲突）
# 训练视角1的目标检测模型  
python ./tools/train1.py     
# 训练视角2的目标检测模型  
python ./tools/train2.py    
```

### 模型训练结果
#### 视角1的检测性能（1 epoch）
```text
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.184
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.313
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.197
……
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.090
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.333
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.578  
```

#### 视角2的检测性能（1 epoch）
```text
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.118
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.219
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.120
……
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.085
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.206
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.661
```

### 模型文件处理
在 /root/2025_05_MMDet_MMTrack_MDMT/work_dirs/COCO_1 中找到视角1下的目标检测模型 epoch_1.pth，重命名为 COCO_1.pth    
在 /root/2025_05_MMDet_MMTrack_MDMT/work_dirs/COCO_2 中找到视角2下的目标检测模型 epoch_1.pth，重命名为 COCO_2.pth  



--------------------------------------------------------------------------------------------

## 配置 MDMT 环境
### NVIDIA GeForce RTX 1660Ti    
```bash
# 创建 Python 3.8 虚拟环境
conda create -n 2025_05_MDMT python=3.8 -y
conda activate 2025_05_MDMT

# 安装 PyTorch 套件（CUDA 10.2版本）
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch

# 进入项目目录
cd ./2025_05_MMDet_MMTrack_MDMT/Multi-Drone-Multi-Object-Detection-and-Tracking-main

# 安装MMCV完整版（必须与PyTorch和CUDA版本严格匹配，这里对应CUDA10.2+PyTorch1.10）
pip --default-timeout=100 install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html

# 安装MMDetection目标检测框架（版本需要与MMCV兼容）
pip --default-timeout=100 install mmdet==2.25.1
# 安装数据处理库pandas（指定1.3.5版本保持兼容性）
pip --default-timeout=100 install pandas==1.3.5
# 安装科学计算库scipy（较大依赖包，设置较长超时时间）
pip --default-timeout=200 install scipy==1.7.3
# 安装文档测试工具
pip --default-timeout=100 install xdoctest==0.10.0
# 安装MMTracking多目标跟踪框架（需确保与MMDet版本兼容）
pip --default-timeout=100 install mmtrack==0.14.0

# 安装项目依赖文件中的所有Python包
pip install -r requirements.txt
# 以开发模式安装当前项目
python setup.py develop

# 安装OpenCV扩展库
pip install --default-timeout=100 opencv-contrib-python==4.9.0.80
# 安装数值计算库numpy
pip --default-timeout=100 install numpy==1.23.0
# 安装即时编译器numba
pip --default-timeout=100 install numba==0.58.1

[//]: # (conda deactivate)
[//]: # (conda remove -n 2025_05_MDMT --all)
```

## 系统配置说明
### 1. 添加并修改配置文件
路径 ./Multi-Drone-Multi-Object-Detection-and-Tracking-main/configs/mot/bytetrack    
操作内容：
· 新增配置文件：COCO_1.py 和 COCO_2.py
· 修改预训练权重路径（第47行）：      
```python
# COCO_1.py
'./mmdetection-2.22.0/wordir/COCO_1/COCO_1.pth' 
# COCO_2.py 
'./mmdetection-2.22.0/wordir/COCO_2/COCO_2.pth'   
```
    
### 2. 修复模块导入错误    
路径 .//Multi-Drone-Multi-Object-Detection-and-Tracking-main/demo/utils/common.py    
操作内容：    
· 注释掉第10行的导入语句：    
```python
# 注释冲突的导入语句
# from matching_pure import matching, calculate_cent_corner_pst  
```
    
### 3. 替换 MMTrack 库文件         
定位目标环境目录：D:\anaconda3\envs\2025_05_MDMT\Lib\site-packages\mmtrack         
使用项目目录中的文件替换以下模块：    
```text
- apis/inference.py     
- models/mot/byte_track.py     
- models/trackers/byte_tracker.py   
```
  
## 系统执行流程
### 推理与评估
```bash
# 执行MIA网络推理
python ./demo/supplement_MIA.py
# 格式转换（JSON -> TXT）
python ./demo/eval/json_2_txt.py
# 单机跟踪性能评估
python ./demo/eval/txttxt_test.py
# 多机匹配性能评估
python ./demo/eval/mango_eval_MDA_perImage.py
```
 
## 部分结果展示    
```bash
# 多机匹配性能评估
python ./demo/eval/mango_eval_MDA_perImage.py
``` 
```text
结果和论文中 TABLE VIII 的结果相近

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

