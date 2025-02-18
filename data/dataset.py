r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.fss import DatasetFSS
from data.isaid import DatasetISAID
from data.pspds import DatasetPSPDS
import albumentations as A
import albumentations.pytorch
import numpy as np


class Compose(A.Compose):
    def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1):
        super().__init__(transforms, bbox_params=bbox_params, keypoint_params=keypoint_params,
                         additional_targets=additional_targets, p=p)

    def __call__(self, image, mask):
        augmented = super().__call__(image=np.array(image), mask=np.array(mask))
        return augmented['image'], augmented['mask']

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
            'isaid': DatasetISAID,
            'pspds': DatasetPSPDS
        }
        cats_augmentation = [
            A.ToGray(p=0.2),
            A.Posterize(p=0.2),
            A.Equalize(p=0.2),
            A.Sharpen(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Solarize(p=0.2),
            A.ColorJitter(p=0.2),
            # A.HorizontalFlip(p=0.5)
        ]

        # cls.img_mean = [0.485, 0.456, 0.406]
        # cls.img_std = [0.229, 0.224, 0.225]
        cls.img_mean = [0.48145466, 0.4578275, 0.40821073]
        cls.img_std = [0.26862954, 0.26130258, 0.27577711]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.trn_transform = Compose([
            *cats_augmentation,
            # A.Rotate(p=0.5), # add for isaid
            A.Resize(img_size, img_size),
            A.Normalize(cls.img_mean, cls.img_std),
            A.pytorch.transforms.ToTensorV2(),
        ])

        cls.transform = Compose([
            A.Resize(img_size, img_size),
            A.Normalize(cls.img_mean, cls.img_std),
            A.pytorch.transforms.ToTensorV2(),
        ])

        # cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
        #                                     transforms.ToTensor(),
        #                                     transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0
        transform = cls.trn_transform if split == 'trn' else cls.transform

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader, dataset
