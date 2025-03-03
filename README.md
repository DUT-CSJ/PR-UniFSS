# PR-UniFSS
Code of Beyond Mask: Rethinking Guidance Types in Few-shot Segmentation

This repository is still being gradually updated. Please stay tuned.

### Datasets

Please follow [HSNet](https://github.com/juhongm999/hsnet?tab=readme-ov-file#preparing-few-shot-segmentation-datasets) to prepare few-shot segmentation datasets.

### Training and Testing

Training

```
python train.py --benchmark pascal/coco/fss1000/isaid/pspds --logpath ./your_path --fold 0/1/2/3 --img_size 400 --backbonoe resnet101 --bsz 12
```

Testing

```
python test.py --benchmark pascal/coco/fss1000/isaid/pspds --load ./your_path/model.pt --fold 0/1/2/3 --nshot 1/5 --img_size 400 --backbonoe resnet101 --bsz 12
```
