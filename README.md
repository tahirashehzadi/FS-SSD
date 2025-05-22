# FD-SSD
[![Paper](https://img.shields.io/badge/Paper-Read-blue)](https://example.com/your-paper-link)
[![GitHub issues](https://img.shields.io/github/issues/tahirashehzadi/FS-SSD)](https://github.com/tahirashehzadi/FS-SSD/issues)
[![GitHub license](https://img.shields.io/github/license/tahirashehzadi/FS-SSD)](https://github.com/tahirashehzadi/FS-SSD/blob/main/LICENSE)
<div style="text-align: center;">
    <img src="main_new.jpg" alt="warmup.png" width="900"/>
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
    cd DocSemi
    ```

2. Create a virtual environment:
    ```sh
    conda create -n docsemi python=3.8
    conda activate docsemi

     ```

3. Install PyTorch:
    ```sh
    conda install pytorch==1.9.0 torchvision==0.10.0 torchtext==0.10.0 cudatoolkit=10.2 -c pytorch
    ```
    
4. Please install mmdet in editable mode first:
     ```sh
    cd thirdparty/mmdetection && python -m pip install -e .
     ```
        
4. Building on mmdetection, we have developed a detection transformer module (detr_od) and a semi-supervised module (detr_ssod) in a similar manner. 
  These modules need to be installed first. Ensure you update the module names (detr_od and detr_ssod) in the setup.py file if necessary.
     ```sh
    cd ../../ && python -m pip install -e  .
     ```
         
6.This process will install mmdet, detr_od, and detr_ssod in the environment. Additionally, you need to compile the CUDA operations required for deformable attention:
  ```sh
   cd detr_od/models/utils/ops
   python setup.py build install
  ```

### Download Data, Models, and Configs

5. Download the images:
    ```sh
    #download images
    gdown https://drive.google.com/uc?id=1Xm794_tzCh1TtIfJYJLFlmv013GTL_Uh
    unzip images_all.zip -d data/v1/
   ```
6. Download the model weights, and configs
   ```sh
    #download model weights
    gdown --folder https://drive.google.com/drive/folders/1zgNxQEXhGm3FTIQAKqkYd3YH0O5SHhm_
    ```


## Generating Predictions and Evaluation

To generate predictions using the trained model/weights, make sure to download the images, model weights, and configs:


1. Run inference:
    ```sh
    python inference.py
    ```

2. The model will infer into the evaluation folder. To evaluate COCO Metrics on the generated predictions:
    ```sh
    python evaluate.py
    ```

### Results and Models


We provide detailed results and model weights for reproducibility and further research.

| Methods                       | Multi-task | AP75FD    | APFD      | AP50FD    | AP75      | AP        | AP50      | Model Weights |
|-------------------------------|------------|-----------|-----------|-----------|-----------|-----------|-----------|--------------|
| **Traditional Detectors***    |            |           |           |           |           |           |           |              |
| Diffusion-DETR w/o pretraining | ✗          | 0.04      | 1.31      | 7.58      | 0.04      | 1.7       | 8.85      | [Download](https://drive.google.com/drive/folders/1l9EsF5x8QTV3x0QT6yKeBpAAkR8fIiqH?usp=drive_link) |
| Diffusion-DETR                | ✗          | 55.52     | 51.42     | 61.28     | 62.58     | 59.06     | 66.37     | [Download](https://drive.google.com/drive/folders/1wtbbvAHTwpmRHfyMjc2xuYbBAqvHrY7L?usp=drive_link) |
| DDETR                         | ✗          | 56.92     | 50.41     | 60.51     | 62.68     | 57.44     | 65.48     | [Download](https://drive.google.com/drive/folders/1lWdPoUGe5HQvq5eU4SPCPnXLCf47pqwD?usp=drive_link) |
| DINO                          | ✗          | 54.03     | 49.68     | 57.94     | 55.13     | 51.65     | 57.65     | [Download](https://drive.google.com/drive/folders/1yuxNT8OQefXn7fmcY6P7yvWgNY5pPRz3?usp=drive_link) |
| **Open-Set Detectors †**      |            |           |           |           |           |           |           |              |
| GLIP                          | ✗          | 40.57     | 32.0      | 46.34     | 51.3      | 40.47     | 55.85     | [Download](https://drive.google.com/drive/folders/1sqnFCCi9mWEBcGhUw1flZUwCz8Y70efO?usp=drive_link) |
| GDINO                         | ✗          | 58.32     | 56.59     | 61.07     | 63.69     | 62.59     | 65.89     | [Download](https://drive.google.com/drive/folders/1dnZ010Yo-Xix1Pd56beTPaIfopSpUfVb?usp=drive_link) |
| GLIP                          | ✔️         | 41.78     | 33.68     | 47.09     | 51.97     | 42.73     | 56.7      | [Download](https://drive.google.com/drive/folders/1cZWXUyxbvhJhiikW8srecyOyhMYAmEOA?usp=sharing) |
| GDINO (our baseline)          | ✔️         | 55.55     | 54.75     | 59.99     | 62.6      | 62.08     | 65.81     | [Download](https://drive.google.com/drive/folders/1wiwm1j90HTiriB5UX4gwFRN4_In679FL?usp=sharing) |
| **FD-SOS (ours)**             | ✔️         | **62.45** | **60.84** | **66.01** | **67.07** | **65.97** | **69.67** | [Download](https://drive.google.com/drive/folders/1tY1yDnCE3AA7crXGiHNBN5fGb-zi4XVN?usp=drive_link) |

#### *requires pre-training on public dental dataset after initialization from ImageNet pre-trained weights.
#### † refers to fine-tuning existing VLM pre-trained models.

Traditional object detectors fail without warmup on public dental datasets. We provide warmup models for traditional object detectors available [here]().

<div style="text-align: center;">
    <img src="graphs/warmup.png" alt="warmup.png" width="300"/>
</div>

## Training FD-SOS

To train FD-SOS , please follow the instructions to [get started](#getting-started
) and install dependencies.



All configs for all experiments are available in [train_FD.sh](train_FD.sh).

To run FD-SOS benchmark, make sure all images are available in [data/v1/images_all](data/v1/images_all) and run:
```
bash train.sh
```

### Acknowledgment
Code is built on [mmdet](https://mmdetection.readthedocs.io/en/latest/get_started.html).


