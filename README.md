
Our code is based on mmsegmentation with reference to the code of "DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting".
If you want to use this project, first configure the environment according to the conditions in DenseCLIP.

Requirements

- torch>=1.8.0
- torchvision
- timm
- mmcv-full==1.3.17
- mmseg==0.19.0
- mmdet==2.17.0
- regex
- ftfy
- fvcore

To use our code, please first install the `mmcv-full` and `mmseg`/`mmdet` following the official guidelines
([`mmseg`](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md), [`mmdet`](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)) and prepare the datasets accordingly.

Pre-trained CLIP Models

Download the pre-trained CLIP models (`RN50.pt`, `RN101.pt`, `VIT-B-16.pt`) and save them to the `pretrained` folder.
