# FD-SSD
[![Paper](https://img.shields.io/badge/Paper-Read-blue)](https://example.com/your-paper-link)
[![GitHub issues](https://img.shields.io/github/issues/tahirashehzadi/FD-SSD)](https://github.com/tahirashehzadi/FS-SSD/issues)
[![GitHub license](https://img.shields.io/github/license/tahirashehzadi/FD-SSD)](https://github.com/tahirashehzadi/FS-SSD/blob/main/LICENSE)
<div style="text-align: center;">
    <img src="resources/main_new.jpg" alt="warmup.png" width="900"/>
</div>

## Getting Started
### Table of Content
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Example](#example)
### Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/tahirashehzadi/FD-SSD.git
    cd FD-SSD
    ```

2. Create a virtual environment:
    ```sh
    conda create -n FD-SSD python=3.8
    conda activate FD-SSD

     ```

3. Install PyTorch:
    ```sh
    conda install pytorch==1.9.0 torchvision==0.10.0 torchtext==0.10.0 cudatoolkit=10.2 -c pytorch
    ```
    
4. Please install mmdet in editable mode first:
     ```sh
    cd thirdparty/mmdetection && python -m pip install -e .
     ```
        
5. Building on mmdetection, we have developed a detection transformer module (detr_od) and a semi-supervised module (detr_ssod) in a similar manner. 
  These modules need to be installed first. Ensure you update the module names (detr_od and detr_ssod) in the setup.py file if necessary.
     ```sh
    cd ../../ && python -m pip install -e  .
     ```
         
6. This process will install mmdet, detr_od, and detr_ssod in the environment. Additionally, you need to compile the CUDA operations required for deformable attention:
     
     ```sh
     cd detr_od/models/utils/ops
     python setup.py build install
    
     ```

### Data Preparation
Download the [FDTOOTH]( https://drive.google.com/uc?id=1Xm794_tzCh1TtIfJYJLFlmv013GTL_Uh) dataset.
  ```sh
/FDTOOTH/data/v1/
    ├── images_all/
    └── annotations/
         ├── 90_FD_train.json
         ├── 40_FD_test.json
         └── 20_FD_val.json
  ```


### Training

- To train the model in a fully supervised setting:
    ```sh
    sh tools/dist_train_detr_od.sh dino_detr ${GPUS}
    ```
- As an example, to train the model in a fully supervised setting with 2 GPUs, you would use the following command:
    ```sh
    sh tools/dist_train_detr_od.sh dino_detr 2
    ```
- To train the model with semi-supervised data:
    ```sh
    sh tools/dist_train_detr_ssod.sh dino_detr_ssod ${FOLD} ${PERCENT} ${GPUS}
    ```
- For instance, you can execute the following script to train our model using 10% labeled data with 2 GPUs on the first split:
    ```sh
    sh tools/dist_train_detr_ssod.sh dino_detr_ssod 1 10 2
    ```
### Evaluation
- To evaluate the model: 
    ```sh
    python tools/test.py <CONFIG_FILE_PATH> <CHECKPOINT_PATH> --eval bbox
    ```  

- For example, to evaluate the model in a semi-supervised setting: 
    ```sh
    python tools/test.py configs/detr_ssod/detr_ssod_dino_detr_r50_coco_120k.py \
    work_dirs_fdtooth/detr_ssod_dino_detr_r50_coco_120k/10/1/epoch_4000.pth --eval bbox
    ```
We provide detailed results and models trained by us bellow:

### Results
| Label (%) | mAP  | AP50 | AP75 | APFD | F1   | Pth files |
|-----------|------|------|------|------|------|-----------|
| 5%        | 40.5 | 44.8 | 43.2 | 42.5 | 0.53 |[CKPT](https://drive.google.com/file/d/1QX-ArM5jBZNHRuatoC3M5ub71uVoCQ3-/view?usp=drive_link) |
| 10%       | 49.9 | 55.6 | 52.2 | 45.7 | 0.64 |[CKPT](https://drive.google.com/file/d/1k9v2kVAAGK7br08-4JiBcHPhU__ONUKT/view?usp=drive_link) |
| 50%       | 55.0 | 59.7 | 56.5 | 50.3 | 0.69 |[CKPT](https://drive.google.com/file/d/1i_kh644t9m4zvHL79RUwsbIhP9RckUY3/view?usp=drive_link) |

### Example

<p align="center">
  <img src="demo/visual.png" alt="Image 1" width="45%" />
  <img src="demo/visual2.png" alt="Image 2" width="45%" />
</p>


### Acknowledgment


