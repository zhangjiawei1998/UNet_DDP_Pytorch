import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, img_height: int, img_width: int, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, newH, newW, is_mask):
        
        assert newW > 0 and newH > 0, 'img_height or img_width must > 0'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        #img预处理
        if img_ndarray.ndim == 2 and not is_mask: #升维，变成rgb
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:#转置,Image读取图片转化为array后为()
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        #mask预处理
        if img_ndarray.ndim == 3 and is_mask:     #降维，变成bin
            img_ndarray = np.max(img_ndarray, axis=2)
            # print(img_ndarray)
        # if np.max(img_ndarray) > 1 and is_mask:   #归一化
        #     img_ndarray = img_ndarray / 255
        #     print(img_ndarray)
        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.img_height, self.img_width, is_mask=False)
        mask = self.preprocess(mask, self.img_height, self.img_width, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
