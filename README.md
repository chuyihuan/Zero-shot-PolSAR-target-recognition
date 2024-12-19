
# Zero-shot-PolSAR-target-recognition
This is the official implementation of ***POlSAR-ZSL***, a PolSAR recognition method. For more details, Comming Soon!

**Zero-shot-PolSAR-target-recognition [[Paper]]()**  <br />
Xiaojing Yang<br />

![intro](Network.jpg)
## Installation

Please refer to [install.md](docs/install.md) for installation.


## Getting Started
## Preparation
Clone the code
```
git clone https://github.com/XinZhangNLPR/JSTARs_MTLDet.git
```


Download the model weight used in the paper:

#### HRSID dataset
|                                             |Backbone|   AP    |   AP@50   |   AP@75   |   AP_S    |   AP_M    |    AP_L   | download | 
|---------------------------------------------|:-------:|:-------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| MTL-Det |[ResNeXt-101-64×4](work_dirs/HTL_1x_renext/HTL_cascade_rcnn_x101_64x4d_fpn_1x_hrsid.py)| 68.0 | 89.5 |  77.7 | 68.7 | 69.6 |25.8 |[Google](https://drive.google.com/file/d/1I1OZ4Aqu7XF_6olL9E0MCEkAMgrLJaGl/view?usp=sharing)

Put the model to ***work_dirs/HTL_1x_renext/***
#### LSSDD-v1.0 dataset
|                                             |Backbone|Off-shore|In-shore |  ALL  | download | 
|---------------------------------------------|:-------:|:-------:|:---------:|:---------:|:---------:|
| MTL-Det |[ResNet-50](work_dirs/HTL_1x_faster/HTL_ins_faster_rcnn_r50_fpn_1x_hrsid.py)| 88.7 | 38.7 |  71.7 |[Google](https://drive.google.com/file/d/1kzTY-dijPJQM2GWmw0erzrCDdsOSxr28/view?usp=sharing)

Put the model to ***work_dirs/HTL_1x_faster/***


## Evaluate
1.GOTCHA on Ours(CE-GZSL) Test
```shell
./tools/dist_test.sh work_dirs/HTL_1x_faster/HTL_ins_faster_rcnn_r50_fpn_1x_hrsid.py work_dirs/HTL_1x_faster/epoch_11.pth 8 --eval mAP
```

