# Optimization of a classification model for thoracic diseases using Habana Gaudi

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

```
                     train     val    test
Atelectasis         10.190   9.974  10.788
Cardiomegaly         2.485   2.139   2.594
Effusion            11.802  11.516  12.277
Infiltration        17.732  17.987  17.554
Mass                 5.082   5.571   5.051
Nodule               5.576   5.464   5.951
Pneumonia            1.246   1.185   1.079
Pneumothorax         4.722   4.492   4.854
Consolidation        4.158   3.984   4.266
Edema                2.154   1.783   1.841
Emphysema            2.293   1.854   2.269
Fibrosis             1.476   1.480   1.614
Pleural_Thickening   2.904   3.316   3.272
Hernia               0.184   0.365   0.187
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

```
training 3750 batches 14999 images.
epoch   1/150 batch 3750/3750 train loss 1.5172 1921.828sec
The average AUC is 0.512 (0.558 0.493 0.559 0.532 0.505 0.536 0.488 0.541 0.532 0.467 0.479 0.511 0.518 0.453)
epoch   2/150 batch 3750/3750 train loss 1.4971 1875.190sec
The average AUC is 0.534 (0.644 0.490 0.671 0.608 0.476 0.543 0.431 0.593 0.588 0.283 0.507 0.592 0.554 0.504)
epoch   3/150 batch 3750/3750 train loss 1.4930 1872.186sec
The average AUC is 0.539 (0.665 0.449 0.680 0.622 0.467 0.563 0.443 0.602 0.643 0.253 0.496 0.594 0.550 0.521)
epoch   4/150 batch 3750/3750 train loss 1.4895 1871.525sec
The average AUC is 0.542 (0.679 0.448 0.678 0.615 0.460 0.571 0.418 0.614 0.682 0.232 0.515 0.609 0.560 0.510)
epoch   5/150 batch 3750/3750 train loss 1.4869 1871.564sec
The average AUC is 0.546 (0.674 0.427 0.690 0.615 0.484 0.572 0.443 0.622 0.696 0.230 0.509 0.605 0.569 0.508)
...
epoch  96/500 batch  3750/ 3750 train loss 1.2337 1973.414sec
The average AUC is 0.850 (0.988 0.981 0.986 0.457 0.993 0.989 0.503 0.996 0.989 0.569 0.950 0.940 0.991 0.564)
epoch  97/500 batch  3750/ 3750 train loss 1.2336 1970.914sec
The average AUC is 0.855 (0.986 0.983 0.987 0.468 0.993 0.989 0.525 0.995 0.989 0.586 0.951 0.942 0.991 0.580)
epoch  98/500 batch  3750/ 3750 train loss 1.2334 1967.211sec
The average AUC is 0.845 (0.983 0.981 0.985 0.472 0.994 0.990 0.482 0.997 0.989 0.558 0.951 0.942 0.992 0.514)
epoch  99/500 batch  3750/ 3750 train loss 1.2332 1968.969sec
The average AUC is 0.847 (0.983 0.982 0.984 0.466 0.994 0.989 0.486 0.998 0.988 0.557 0.950 0.940 0.992 0.550)
epoch 100/500 batch  3750/ 3750 train loss 1.2320 1966.244sec
The average AUC is 0.847 (0.986 0.984 0.987 0.466 0.993 0.990 0.517 0.997 0.989 0.583 0.952 0.940 0.992 0.486)
```
