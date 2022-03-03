import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()
    

def draw_mask(img, mask):

    """

    Args:
        img: origin image, (C,H,W)
        mask: (H,W),有几个类别, max(mask)就等于几,0值默认为背景,因此画黑色,其他根据COLOR来画
        
    Returns:
        a image with different color mask

    """
    COLORS = [(0,0,0),(0,0,255),(0,255,0),(0,0,255)]
    img = np.array(img)
    mask= np.array(mask)
    for i in range(1,np.max(mask)+1):
        mask_area = np.array(mask == i)
        img[mask_area] = 0.5 * img[mask_area] + 0.5 * np.array(COLORS[i])
    
    img = img[...,::-1] # 在最后一维上，倒序索引，实现RGB->BGR
    return img
    