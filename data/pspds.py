r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import open_clip


class DatasetPSPDS(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = split
        self.benchmark = 'fss'
        self.shot = shot
        self.fold = fold

        # self.base_path = os.path.join(datapath, 'FSS-1000')
        self.base_path = './defect_datasets/FS_dataset'

        # Given predefined test split, load randomly generated training/val splits:
        # (reference regarding trn/val/test splits: https://github.com/HKUSTCV/FSS-1000/issues/7))
        self.class_names = ['poke', 'mark', 'liquid', 'oil', 'water', 'spot', 'scratch']
        with open('./data/splits/pspds/fold'+str(self.fold)+'.txt', 'r') as f:
            self.categories = f.read().split('\n')[:-1]
            if self.split != 'trn':
                result = [item for item in self.class_names if item not in self.categories]
                self.categories = result
        # self.categories = sorted(self.categories)

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata() * 50

        self.transform = transform
        
        self.tokenizer = open_clip.get_tokenizer('RN101')

    def __len__(self):
        return 1000#len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img, query_mask = self.transform(query_img, query_mask)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_transformed = [self.transform(support_img, support_cmask) for support_img, support_cmask in zip(support_imgs, support_masks)]
        support_masks = [x[1] for x in support_transformed]
        support_imgs = torch.stack([x[0] for x in support_transformed])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,

                 'class_id': torch.tensor(class_sample),
                 'class_name': self.tokenizer(self.class_names[torch.tensor(class_sample)])}

        return batch

    def load_frame(self, query_name, support_names):
        # print('image name: ', query_name, support_names)
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_id = query_name.split('/')[-1].split('_')[-1]
        query_name = os.path.join(os.path.dirname(query_name), 'label_' + query_id)
        support_ids = [name.split('/')[-1].split('_')[-1] for name in support_names]
        support_names = [os.path.join(os.path.dirname(name), 'label_' + sid) for name, sid in zip(support_names, support_ids)]
        # print('mask name: ', query_name, support_names)

        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        # mask[mask!=0] = 1
        # mask[mask!=1] = 0
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        class_name = query_name.split('/')[-2]
        img_num = len(glob.glob(os.path.join(self.base_path, class_name, "label*")))
        class_sample = self.class_names.index(class_name)

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(range(1, img_num + 1), 1, replace=False)[0]
            support_name = os.path.join(os.path.dirname(query_name), 'img_' + str(support_name)) + '.png'
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break
        return query_name, support_names, class_sample

    def build_class_ids(self):
        if self.split == 'trn':
            if self.fold == 0:
                class_ids = [3, 4, 5, 6]
            elif self.fold == 1:
                class_ids = [0, 1, 2, 5, 6]
            else:
                class_ids = [0, 1, 2, 3, 4]
        elif self.split == 'val' or self.split == 'test':
            if self.fold == 0:
                class_ids = [0, 1, 2]
            elif self.fold == 1:
                class_ids = [3, 4]
            else:
                class_ids = [5, 6]
        return class_ids

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, cat))])
            for img_path in img_paths:
                # if os.path.basename(img_path).split('.')[1] == 'jpg':
                if 'img' in img_path:
                    img_metadata.append(img_path)
        return img_metadata
