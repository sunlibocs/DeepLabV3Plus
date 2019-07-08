### Pre-trained model

| Backbone  | train/eval os  |mIoU in val |Pretrained Model|
| :-------- | :------------: |:---------: |:--------------:|
| ResNet    | 16/16          | 78.43%     | [google drive](https://drive.google.com/open?id=1NwcwlWqA-0HqAPk3dSNNPipGMF0iS0Zu) |
| MobileNet | 16/16          | 70.81%     | [google drive](https://drive.google.com/open?id=1G9mWafUAj09P4KvGSRVzIsV_U5OqFLdt) |
| DRN       | 16/16          | 78.87%     | [google drive](https://drive.google.com/open?id=131gZN_dKEXO79NknIQazPJ-4UmRrZAfI) |



### In Progress

I am training the VOC model. And try to change the backbone to CSPN's backbone.

### Training

Configure your dataset path in [mypath.py]
    

### Test Environment

Pytroch 1.1.0(python 3.7). 

### Acknowledgement
The code comes from https://github.com/jfzhang95/pytorch-deeplab-xception 


