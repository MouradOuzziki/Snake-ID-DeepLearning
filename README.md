# Deep Learning-Based Snake Species Identification for Enhanced Snakebite Management
![Alt text](https://github.com/MouradOuzziki/Snake-ID-DeepLearning/blob/766d280510e95e1458ae545664bd3829174563d2/image%20of%20snake.png)
This repository implements a deep learning-based solution for classifying snake species using transfer learning and fine-tuning. The model is trained on a combination of global datasets SnakeCLEF 2021 and a localized dataset of Moroccan snake species. The goal is to build an accurate and efficient model for snake identification, which can be applied in various real-world applications, such as biodiversity research, conservation efforts, and educational tools.

## Link to Our Paper
For more details about the methodology and results, please refer to our published paper:  
[**Deep Learning-Based Snake Identification for Enhanced Snakebite Management**](https://www.mdpi.com/2673-2688/6/2/21)

# Citation
Consider giving this repository a star and cite our paper in your publications if it helps your research.
```
@article{iguernane2025deep,
  title={Deep Learning-Based Snake Species Identification for Enhanced Snakebite Management},
  author={Iguernane, Mohamed and Ouzziki, Mourad and Es-Saady, Youssef and El Hajji, Mohamed  and Lansari, Aziza and Bouazza, Abdellah},
  journal={AI},
  volume={6},
  number={2},
  pages={21},
  year={2025},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```


## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
  

## Overview

This project utilizes **transfer learning** and **fine-tuning** to classify snake species into 26 local species using pre-trained models. The following models are used in the project:

- EfficientNet B0
- VGG16
- VGG19

The model is trained in two stages:
1. **Global Training**: The model is initially trained on the **SnakeCLEF 2021** dataset, which contains 772 snake species, to learn generalized image features.
2. **Local Fine-Tuning**: The model's classification head is modified to output predictions for 26 Moroccan snake species, and the model is fine-tuned using a smaller learning rate on the local dataset.

## Datasets

- **SnakeCLEF 2021 Dataset**: A large-scale dataset containing images of 772 snake species from across the globe.
- **Local Moroccan Snake Dataset**: A smaller dataset containing images of 26 snake species specific to Morocco.

The global dataset is used for the initial training, while the local dataset is used for fine-tuning the model to adapt to the unique characteristics of Moroccan snake species.

## Model Architecture

The project uses pre-trained models from the **Timm library**, which includes architectures such as:

- **EfficientNet B0**
- **VGG16**
- **VGG19**

# Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

