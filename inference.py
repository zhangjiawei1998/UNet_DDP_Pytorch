import argparse
import logging
import os
from pathlib import Path
import cv2
from matplotlib.pyplot import get

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from utils.utils import draw_mask
from unet import UNet

def predict_img(net,
                full_img,
                device,
                img_height : int,
                img_width : int,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, img_height, img_width, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return full_mask.argmax(dim=0).cpu().numpy()  # return (H, W)

def get_filenames(dir_input:Path, dir_output:Path):
    dir_output.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    in_filenames  = [str(dir_input) + '/'+ file.name for file in dir_input.iterdir()] 
    out_filenames = [str(dir_output) + '/mask_' +file.name for file in dir_input.iterdir()]
    return in_filenames, out_filenames

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3: 
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--img_dir',   '-di', type=Path, default = Path('../dataset/shein'),      help='the direction of input img')
    parser.add_argument('--mask_dir',  '-dm', type=Path, default = Path('../dataset/out_mask_shein_class1_epoch47'), help='the direction of out mask')
    parser.add_argument('--pth',       '-p',  type=Path, default = Path('./pth/class1_epoch47.pth'),  help='the direction of checkpoint')
    parser.add_argument('--img_height','-ih', type=int,  default=300, help='height after scale on origin img')
    parser.add_argument('--img_width', '-iw', type=int,  default=225, help='width after scale on origin img')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    in_files, out_files = get_filenames(args.img_dir, args.mask_dir)
    
    net = UNet(n_channels=3, n_classes=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.pth}')
    logging.info(f'Using device {device}')

    model_dicet = torch.load(str(args.pth), map_location=device)
    net.load_state_dict(model_dicet, strict=False)
    net.to(device=device)
    #net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(path_model, map_location=device).items()})

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)
        print(img.size)
        mask = predict_img(net=net,
                           full_img=img,
                           img_height=args.img_height,
                           img_width=args.img_width,
                           out_threshold=0.5,
                           device=device)
        
        out_filename = out_files[i]
        out_img = draw_mask(img, mask)
        cv2.imwrite(out_filename, out_img)
        print(out_filename)
        logging.info(f'Mask saved to {out_filename}')
        # logging.info(f'Visualizing results for image {filename}, close to continue...')
        #plot_img_and_mask(img, mask)
