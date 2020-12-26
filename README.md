# CV Wheat Detection

Code for object detection adapting EfficientDet.

- Team: The Fastest Kokushimuso in Hsintsu
- Competetion: [Kaggle Global Wheat Detection ECCV 2020](https://www.kaggle.com/c/global-wheat-detection)
- Method: EfficientDet
 
 
## Environment
- Ubuntu 16.04 LTS

## Reference
- Most of the code based on [shonenkov's notebook](https://www.kaggle.com/shonenkov/training-efficientdet), we appreciate his awesome work.

## Outline
1. [Installation](#Installation)
2. [Dataset Preparation](#Dataset-Preparation)
3. [Training](#Training)


## Installation

### clone necessary repo

First, clone our [wheat-detection repo](https://github.com/osinoyan/wheat-detection)


```
$ https://github.com/osinoyan/wheat-detection
$ cd wheat-detection
```

### environment installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n wheat python=3.6
source activate wheat
pip install -r requirements.txt
```

## Dataset Preparation

### Global Wheat Detection Dataset
Download the dataset from [kaggle](https://www.kaggle.com/c/global-wheat-detection/data) and unzip *global-wheat-detection.zip* under the directory *data/*.
Make sure to place the data like below:
```
    wheat-detection/
        +- data/
        |   +- sample_submission.csv
        |   +- train.csv
        |   +- test/
        |   |   +- 2fd875eaa.jpg
        |   |   +- ...
        |   +- train/
        |   |   +- 0a3cb453f.jpg
        |   |   +- ...
        ... ...
```


## Training
We briefly provide the instructions to train the model

### Get EfficientDet model
Unfortunately, you have to download the weights manually.
Download *efficientdet_d5-ef44aea8.pth* from [This link](https://www.kaggle.com/mathurinache/efficientdet?select=efficientdet_d5-ef44aea8.pth) and make sure to place it under the directory *efficientdet_model/* like below:
```
    wheat-detection/
        +- efficientdet_model/
        |   +- efficientdet_d5-ef44aea8.pth
        +- data/
        +- timm-efficientdet-pytorch-revised/
        +- weighted-boxes-fusion/
        |
        ...
```

### Start training
Just run this:
```
python train.py
```
