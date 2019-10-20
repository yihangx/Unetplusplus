# UNet++: A Nested U-Net Architecture for Medical Image Segmentation
This is the implementation of Unet++ paper Using Keras. You can find the original paper at [here](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1).

# U-Net: Convolutional Networks for Biomedical Image Segmentation
To compare the performance and architecture, we also included U-Net code. You can find the original paper at [here](https://arxiv.org/pdf/1505.04597.pdf)

## Prepare data
We used "Find the nuclei in divergent images to advance medical discovery" data set from 2018 Data Science Bowl [URL](https://www.kaggle.com/c/data-science-bowl-2018). Download all data set and extract it. Again, extract both stage1_train and stage1_test.

## Clone
```
git clone git@github.com:yihangx/BME590_Unetplusplus.git
```

## Change the data path either python code or Jupyter notebook file.
```
TRAIN_PATH = './stage1_train/'
TEST_PATH = './stage1_test/'
```

### Execute the code with python
```
python3 unet.py
python3 unet++.py
```

### Execute the code with Jupyter notebook

# Version
tensorflow 2.0

# License MIT
