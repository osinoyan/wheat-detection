# albumentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
# tools
import pandas as pd
import numpy as np
import cv2
import os
import re
import sys
from PIL import Image
# torch
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler
# matlab
from matplotlib import pyplot as plt
# more tools
from datetime import datetime
import time
import random
from sklearn.model_selection import StratifiedKFold
from glob import glob
# effdet
sys.path.insert(0, "timm-efficientdet-pytorch-revised")
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchEval
from effdet.efficientdet import HeadNet

# WBF
sys.path.insert(0, "weighted-boxes-fusion")
from ensemble_boxes import *

"""## Test
> ref: https://www.kaggle.com/shonenkov/inference-efficientdet?scriptVersionId=34956042

### Get Valid Transforms
"""

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

"""### Dataset Retriever"""

# TEST_ROOT_PATH = "../input/global-wheat-detection/test" # for Kaggle notebook
# TEST_ROOT_PATH = "test"  # for colab

root_dir = 'data'  # for colab
TEST_ROOT_PATH = os.path.join(root_dir, 'test')

class DatasetRetriever(Dataset):

    def __init__(self, image_ids, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = cv2.imread(f'{TEST_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

"""### TTA"""

class BaseWheatTTA:
    """ author: @shonenkov """
    image_size = 512

    def augment(self, image):
        raise NotImplementedError
    
    def batch_augment(self, images):
        raise NotImplementedError
    
    def deaugment_boxes(self, boxes):
        raise NotImplementedError

class TTAHorizontalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(1)
    
    def batch_augment(self, images):
        return images.flip(2)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]
        return boxes

class TTAVerticalFlip(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return image.flip(2)
    
    def batch_augment(self, images):
        return images.flip(3)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]
        return boxes
    
class TTARotate90(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = self.image_size - boxes[:, [1,3]]
        res_boxes[:, [1,3]] = boxes[:, [2,0]]
        return res_boxes
    
class TTARotate180(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return torch.rot90(image, 2, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 2, (2, 3))
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,1,2,3]] = self.image_size - boxes[:, [2,3,0,1]]
        return boxes
    
class TTARotate270(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return torch.rot90(image, 3, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 3, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = boxes[:, [1,3]]
        res_boxes[:, [1,3]] = self.image_size - boxes[:, [2,0]]
        return res_boxes
    
class TTACompose(BaseWheatTTA):
    """ author: @shonenkov """
    def __init__(self, transforms):
        self.transforms = transforms
        
    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image
    
    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images
    
    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)
        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)
        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)
        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)
        return result_boxes
    
    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)

"""### TTA Prediction Function"""

from itertools import product

tta_transforms = []
for tta_combination in product([TTAHorizontalFlip(), None], 
                               [TTAVerticalFlip(), None],
                               [TTARotate90(), TTARotate180(), TTARotate270(), None]):
    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))

def make_tta_predictions(net, images, score_threshold=0.3):
    images = torch.stack(images).cuda().float()
    predictions = []
    with torch.no_grad():
        for tta_transform in tta_transforms:
            result = []
            det = net(tta_transform.batch_augment(images.clone()), torch.tensor([1]*images.shape[0]).float().cuda())

            for i in range(images.shape[0]):
                boxes = det[i].detach().cpu().numpy()[:,:4]    
                scores = det[i].detach().cpu().numpy()[:,4]
                indexes = np.where(scores > score_threshold)[0]
                boxes = boxes[indexes]
                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                boxes = tta_transform.deaugment_boxes(boxes.copy())
                result.append({
                    'boxes': boxes,
                    'scores': scores[indexes],
                })
            predictions.append(result)
    return predictions

"""### Weighted Boxes Fusion Function"""

def run_wbf(predictions, image_index, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

"""### Run Testing Function"""

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)

def run_testing(ckpt, output):
    # load checkpoint
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)
    config.num_classes = 1
    config.image_size = 512
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
        norm_kwargs=dict(eps=.001, momentum=.01))
    
    checkpoint = torch.load(ckpt)['model_state_dict']
    net.load_state_dict(checkpoint)
    net = DetBenchEval(net, config)
    
    # retrieve dataset
    test_img_ids = np.array(
            [path.split('/')[-1][:-4] for path in glob(
                f'{TEST_ROOT_PATH}/*.jpg')])
    test_dataset = DatasetRetriever(
        image_ids=test_img_ids,
        transforms=get_valid_transforms()
        )
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    data_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        collate_fn=collate_fn
    )

    # # sample
    # print('-------------------------------------')
    # print('               SAMPLE                ')
    # print('-------------------------------------')

    # # net.load_state_dict(checkpoint['model_state_dict'])
    # # net = DetBenchEval(model, config)
    # net.to(device)
    
    # # DataLoadet does not support indexing
    # for j, (images, image_ids) in enumerate(data_loader):
    #     break
    # predictions = make_tta_predictions(net, images)
    # i = 3
    # sample = images[i].permute(1,2,0).cpu().numpy()

    # boxes, scores, labels = run_wbf(predictions, image_index=i)
    # boxes = boxes.round().astype(np.int32).clip(min=0, max=511)

    # fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # for j,box in enumerate(boxes):
    #     cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (1, 0, 0), 2)
    #     score=str(float("{:.3f}".format(scores[j])))

    # ax.set_axis_off()
    # ax.imshow(sample)
    
    print('-------------------------------------')
    print('             RUN TESTING             ')
    print('-------------------------------------')

    results = []
    net.eval()
    net.to(device)
    for images, image_ids in data_loader:
        predictions = make_tta_predictions(net, images)
        for i, image in enumerate(images):
            
            # ------------------- SAMPLE ---------------------
            # sample = image.permute(1,2,0).cpu().numpy()
            # boxes, scores, labels = run_wbf(predictions, image_index=i)
            # boxes = boxes.round().astype(np.int32).clip(min=0, max=511)
            # fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            # for j,box in enumerate(boxes):
            #     cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (1, 0, 0), 2)
            #     score=str(float("{:.3f}".format(scores[j])))

            # ax.set_axis_off()
            # ax.imshow(sample)
            # ------------------------------------------------

            boxes, scores, labels = run_wbf(predictions, image_index=i)
            boxes = (boxes*2).astype(np.int32).clip(min=0, max=1023)
            image_id = image_ids[i]

            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            for box in boxes:
                result = {
                    'image_id': image_id,
                    'width': 1024,
                    'height': 1024,
                    'bbox': "[{}, {}, {}, {}]".format(box[0], box[1], box[2], box[3]),
                    'source': 'pseudo'
                }
                results.append(result)
    
    train_csv_path = 'data/train.csv'
    df_mark = pd.read_csv(train_csv_path)
    test_df = pd.DataFrame(results, columns=['image_id', 'width', 'height', 'bbox', 'source'])
    test_df = test_df.append(df_mark, ignore_index=True) 
    test_df.to_csv(output, index=False)
    test_df.head()

"""### Check GPU"""

if torch.cuda.is_available():
    device = torch.device('cuda')
    device_count = torch.cuda.device_count()
    print(f'device_count: {device_count}')
else:
    print('Cuda is not available. That is too bad, bro.')
    torch.device('cpu')

"""### Start Testing ⭐"""

run_testing(
    ckpt='results/best-checkpoint-076epoch_1226.bin',
    output='train_psu.csv'
    )