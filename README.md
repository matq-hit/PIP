# PIP: Delving Deeper Into Pixel Prior for Box-Supervised Semantic Segmentation

The implementation for of ["Delving Deeper Into Pixel Prior for Box-Supervised Semantic Segmentation"](https://ieeexplore.ieee.org/document/9684236), IEEE TIP.

## Dependency
- python 3.7 / pytorch 1.2.0
- pydensecrf
- opencv

## Datasets
- [Pascal VOC 2012 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
    - extract 'VOCtrainval_11-May-2012.tar' to 'VOCdevkit/'
- [WSSL pseudo labels](http://liangchiehchen.com/projects/Datasets.html)
    - extract 'SegmentationClassBboxSeg_Visualization.zip' to 'VOCdevkit/SegmentationClassBboxCRF'
- [SDI pseudo labels](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/weakly-supervised-learning/simple-does-it-weakly-supervised-instance-and-semantic-segmentation)
    - extract 'VOC12_M&G+.zip' to 'VOCdevkit/VOC12_M&G+'


Finally, it should like this
```
VOCdevkit
├── Annotations
├── JPEGImages
├── SegmentationClassBboxCRF
└── VOC12_M&G+
```

## Performance

<table>
    <tr>
        <th>Train set</th>
        <th>Eval set</th>
        <th>Supervision</th>
        <th>Method</th>
        <th>Mean IoU</th>
    </tr>
    <tr>
        <td rowspan="3">
            <i>train_aug</i><br>
        </td>
        <td rowspan="3"><i>val</i></td>
        <td rowspan="2">Box</td>
        <td>WSSL_CRF + PIP</td>
        <td>63.6</td>
    </tr>
    <tr>
        <td>SDI + PIP</td>
        <td>67.9</td>
    </tr>
    <tr>
        <td>Full</td>
        <td>DeepLab-LargeFOV</td>
        <td>69.6</td>
    </tr>
</table>


## Usage
### Train
```
python main.py --type=train
```
### Test
without CRF
```
python main.py --type=test
```

with CRF
```
python main.py --type=test --use_crf
```

### Evaluate
```
python evalate.py
```
## Others
1. [trained model]().
2. For DeepLabv2-ResNet-101 backbone, we refer to [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch). We also provide the [ImageNet pretrained model]().
## Reference
1. [DeepLab-V1-PyTorch](https://github.com/wangleihitcs/DeepLab-V1-PyTorch)
