# UNet_Pytorch

用deepfasion2数据集训练UNet语义分割网络
使用多GPU分布式训练, 分布式训练教程见 https://zhuanlan.zhihu.com/p/86441879

# Quickly Start

## Train
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 
    train.py 
    -di ../dataset/img  
    -dm ../dataset/mask 
    -dp ./pth 
    -ih 300 
    -iw 225 
    -e 100 
    -b 4 
    -l 0.005
        
    以上是我执行的命令行，公司分配给我1台有8个GPU的服务器，我能用后4个, 因此指定GPU 4, 5, 6, 7可见.
    python -m torch.distributed.launch 表示调用torch.distributed.launch 这个.py文件进行分布式训练;
     --nproc_per_node=4 这是launch.py的参数之一，说明创建进程数为4，这个值通常与训练使用的GPU数量一致.（一个GPU一个进程）
     train.py 为训练脚本
    -di 输入图片的目录
    -dm mask图片的目录, 我倾向于 总的类别数 = 背景 + 类别数, 因此我的mask图片通常0代表背景, 1.2.3...代表类别
    -dp 保存训练脚本的目录
    -ih 图像压缩后的高
    -iw 图像压缩后的宽
    -e  epochs
    -b  batch_size, 该参数为4表示一个GPU上batch_size=4, 总的batch_size=4×4=16
    -l  学习率
    
    为了代码的简洁, 其他很多参数没有放在命令行, 通常都是直接改的qwq, 例如损失函数权重, 每多少轮保存一次.pth, 每多少轮降低学习率
 
## Inference
    
    python inference.py -di ../dataset/img -dm ../dataset/outmask -p ./pth/checkpoint_epoch52.pth -ih 300 -iw 225
  
