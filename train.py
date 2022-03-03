import argparse
from ctypes import Union
import logging
import sys
import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
#import wandb
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
import wandb
import torch.distributed
import torch.cuda
import torch.distributed.launch

def train_net(net,
              device,
              dir_img: Path,
              dir_mask: Path,
              dir_checkpoint: Path,
              epochs: int ,
              batch_size: int ,
              learning_rate: float,
              val_percent: float,
              save_checkpoint: bool,
              img_height : int,
              img_width : int,
              local_rank
              ):
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_height, img_width)
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # 3. Create data loaders, use DistributedSampler
    train_sampler = DistributedSampler(train_set, shuffle=True) 
    loader_args = dict(batch_size=batch_size, num_workers=1, drop_last=True, pin_memory=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_set, **loader_args)

    # (Initialize wandb)
    wandb.init(project='UNet_deepfashion', entity='zhangjiawei1998')
    wandb.config = dict(epochs=epochs, 
                        batch_size=batch_size, 
                        learning_rate=learning_rate,
                        val_percent=val_percent, 
                        save_checkpoint=save_checkpoint, 
                        img_height= img_height,
                        img_width = img_width
                        )
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images height:   {img_height}
        Images width:    {img_width}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler
    optimizer = optim.RMSprop(net.module.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, cooldown=20) 在分布式训练时会出现问题，具体见：https://wandb.ai/zhangjiawei1998/UNet_deepfashion%20loss=ce+dice?workspace=user-zhangjiawei1998
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= epochs // 6, gamma=0.5)
    weights = [1.0, 10.0] # 各类别损失函数的权重 [背景, 类别1，类别2...]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    global_step = 0
    
    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.module.n_channels, \
                    f'Network has been defined with {net.module.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                
                true_masks = true_masks.to(device=device, dtype=torch.long)
                masks_pred = net(images)

                CE_loss = criterion(masks_pred, true_masks)
                Dice_loss = dice_loss(F.softmax(masks_pred, dim=1).float(),  # (B,C,H,W)
                                      F.one_hot(true_masks, net.module.n_classes).permute(0, 3, 1, 2).float(), #（B,C,H,W）
                                      classes_weight=weights) # weight of classes
                
                loss = CE_loss + Dice_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.update(images.shape[0])
                global_step += 1
                
                wandb.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0: # global_step = n_train // batch_size，1个epoch里，进行10次evaluation round
                        histograms = {}
                        for tag, value in net.module.named_parameters(): 
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net.module, val_loader, device)
                    
                        wandb.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })
        
        # 学习率衰减
        scheduler.step()
        
        # 分布式训练时，只保存第一块GPU上的网络参数，防止重复保存
        if save_checkpoint and local_rank == 0: 
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.module.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'local_rank = {local_rank}, Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--local_rank',           type=int,   default=0)
    parser.add_argument('--dir_img',       '-di', type=Path,  default = Path('../dataset/img'), help='the direction of in img')
    parser.add_argument('--dir_mask',      '-dm', type=Path,  default = Path('../dataset/mask'), help='the direction of true mask')
    parser.add_argument('--dir_pth',       '-dp', type=Path,  default = Path('./pth'), help='the direction of checkpoint')
    parser.add_argument('--img_height',    '-ih',  type=int,  default=300, help='height after scale on origin img')
    parser.add_argument('--img_width',     '-iw',  type=int,  default=225, help='width after scale on origin img')
    parser.add_argument('--epochs',        '-e',  type=int,   default=100, help='Number of epochs')
    parser.add_argument('--batch_size',    '-b',  type=int,   default=1,  help='Batch size')
    parser.add_argument('--lr',            '-l',  type=float, default=0.0005, help='Learning rate', dest='lr')
    parser.add_argument('--load',          '-f',  type=str,   default=False, help='Load model from a .pth file')
    parser.add_argument('--validation',    '-v',  dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    # 指定该进程的GPU
    device = torch.device(f'cuda:{args.local_rank}') 
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels = 3 for RGB images
    # n_classes = the number of classes + background
    net = UNet(n_channels=3, n_classes=2, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        pre_dict = torch.load(args.load, map_location=device)
        pre_dict['outc.conv.weight_hhh'] = pre_dict.pop('outc.conv.weight')
        pre_dict['outc.conv.bias_hhh']   = pre_dict.pop('outc.conv.bias')
        net.load_state_dict(pre_dict, strict=False)
        logging.info(f'Model loaded from {args.load}, GPU={args.local_rank}')

    # DistributedDataParallel
    if torch.cuda.device_count() > 1 and torch.distributed.is_available():
        print("DDP is available, Let's use", torch.cuda.device_count(), "GPUs!")
        print(f"生成{os.environ['WORLD_SIZE']}个进程")
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        net.to(device=device)
        net = DDP(module=net, device_ids=[args.local_rank], find_unused_parameters=True)
        
    try:
        train_net(net=net,
                  dir_img=args.dir_img,
                  dir_mask=args.dir_mask,
                  dir_checkpoint=args.dir_pth,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_height=args.img_height,
                  img_width=args.img_width,
                  val_percent=args.val / 100,
                  save_checkpoint=True,
                  local_rank = args.local_rank
                  )
    except KeyboardInterrupt:
        torch.save(net.module.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
