import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    '''
        Average of Dice coefficient for all batches, or for a single mask
        只计算(H,W)或者(B,H,W)维度的dice分数
        @param1: input.shape  = (H,W) or (B,H,W)
        @param2: target.shape = (H,W) or (B,H,W)
        @param3: reduce_batch_first. 
        @param4: epsilon. 
    '''
    assert input.shape == target.shape, f'input.shape != target.shape, {input.shape} != {target.shape}'
    
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, classes_weight: list=None, reduce_batch_first: bool = False, epsilon=1e-6):
    '''
        Average of Dice coefficient for all classes
        计算(B,C,H,W)或者(C,H,W)维度的dice分数
        @param1: input.shape  = (B,C,H,W) or (C,H,W)
        @param2: target.shape = (B,C,H,W) or (C,H,W)
        @param3: classes_weight. the weight of classes in compute the loss, 
                                  example: classes_weight = [2, 4, 5] for three classes(include background)
                                           loss = (2 * loss_class1 + 5 * loss_class2 + 4 * loss_class3) / 11
        @param4: reduce_batch_first. 
        @param5: epsilon. 
    '''
    assert input.shape == target.shape
   
    dice = 0

    if classes_weight is None:
        for channel in range(input.shape[1]):
            dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
        return dice / input.shape[1] # 平均
    
    else:
        for channel in range(input.shape[1]):
            dice += classes_weight[channel] * dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
        return dice / sum(classes_weight) # 加权平均

 
def dice_loss(input: Tensor, target: Tensor, classes_weight: list=None):
    '''
        Dice loss (objective to minimize) between 0 and 1
        @param1: input.shape  = (B,C,H,W)
        @param2: target.shape = (B,C,H,W)
        @param3: classes_weight. the weight of classes in compute the loss, 
                                  example: classes_weight = [0.1, 0.5, 0.4] for three classes(include background)
                                  loss = 0.1 * loss_class1 + 0.5 * loss_class2 + 0.4 * loss_class3
    '''
    assert input.shape == target.shape, f'input.shape != target.shape, {input.shape} != {target.shape}'
    
    return 1 - multiclass_dice_coeff(input, target, classes_weight, reduce_batch_first=True)
