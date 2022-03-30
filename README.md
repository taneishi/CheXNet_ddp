# Optimization of a classification model for thoracic diseases with PyTorch on Habana Gaudi and CUDA device.

## Introduction

Medical imaging is an indispensable technology for modern medicine, and the application of deep learning is also spreading to this field. 
A typical example is image reading using medical images such as X-rays, CT, MRI, etc.
By constructing a model that estimates the name of the disease and the location of the disease using convolutional networks (CNNs), etc., for medical images,
it is expected to reduce the burden on the image reading physician, equalize the diagnostic criteria,
and realize diagnosis that exceeds human capabilities, although diagnosis through reading is still the responsibility of the physician.

On the other hand, there are several challenges in deep learning for medical images. 
One is the collection and labeling of medical images, which requires collecting as many images as necessary for training, 
considering patient privacy, and attaching high-quality labels for training. 
In 2017, the National Institutes of Health (NIH) released a large dataset called ChestX-ray14, described below. 
Other medical institutions have also begun to release medical image datasets with case labels, and the environment for developing models for clinical use is now in place. 
*CheXNet*, which I used in this repository, is one of the models proposed in this study.

## Dataset

I used ChestX-ray14 as a dataset. This dataset is a chest X-ray image dataset provided by NIH. 
112,120 chest X-ray images of 30,805 patients are associated with multiple labels corresponding to each image from 14 different diseases. 
This data set is divided into training set (70%), validation set (10%), and test set (20%).

- [ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) 

The percentages of each disease are following.

```bash
python stats.py
                     train     val    test     bmt
Atelectasis         10.190   9.974  10.788   9.202
Cardiomegaly         2.485   2.139   2.594   3.921
Effusion            11.802  11.516  12.277   9.742
Infiltration        17.732  17.987  17.554  16.583
Mass                 5.082   5.571   5.051   3.161
Nodule               5.576   5.464   5.951   4.281
Pneumonia            1.246   1.185   1.079   1.260
Pneumothorax         4.722   4.492   4.854   3.981
Consolidation        4.158   3.984   4.266   4.101
Edema                2.154   1.783   1.841   1.800
Emphysema            2.293   1.854   2.269   2.501
Fibrosis             1.476   1.480   1.614   3.441
Pleural_Thickening   2.904   3.316   3.272   3.301
Hernia               0.184   0.365   0.187   0.540
```

## Model

As a model, I use an improved version of *CheXNet*, a model proposed by Rajpurkar et al. that takes chest X-ray images as input and performs multi-label classification for each chest disease. 
The structure of the neural network is based on DenseNet121, and the output layer for classification into 14 diseases is added to the trained model by ImageNet, and fine-tuning is performed by ChestX-ray14.

- [CheXNet](https://arxiv.org/abs/1711.05225)

## Usage

First, run the script to download the dataset.

```bash
bash batch_download.sh
```

The next step is to set up a Python environment to optimize and quantize the model. This procedure is described in `run.sh`.

```bash
bash run.sh
```

The following scripts are used to perform inference on the PyTorch, FP32 optimized and INT8 quantized models, respectively.

```bash
python main.py
```

## Results
