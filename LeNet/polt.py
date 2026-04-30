# 从logging模块导入root（注意：此导入在当前代码中未使用）
from logging import root

# 导入FashionMNIST数据集，这是一个常用的服装图像分类数据集
from torchvision.datasets import FashionMNIST
# 导入transforms模块，用于图像预处理和数据增强
from torchvision import transforms
# 导入numpy库，用于数值计算和数组操作
import numpy as np
# 导入PyTorch的数据加载工具，用于批量处理数据
import torch.utils.data as Data
# 导入matplotlib绘图库，用于可视化图像
import matplotlib.pyplot as plt


# 下载并加载FashionMNIST训练数据集
train_data = FashionMNIST(
    root = './data',  # 数据存储的根目录
    train = True,  # 设置为True表示加载训练集，False为测试集
    transform = transforms.Compose([transforms.Resize(size = 224),transforms.ToTensor()]),  # 定义数据预处理步骤：先调整图像大小为224x224，再转换为tensor格式
    download = True  # 如果本地没有数据，则自动下载
)

# 创建数据加载器，用于批量加载数据
train_loader = Data.DataLoader(
    dataset = train_data,  # 指定要加载的数据集
    batch_size = 64,  # 每个批次包含64张图像
    shuffle = True,  # 每个epoch开始时打乱数据顺序，提高模型泛化能力
    num_workers = 0  # 使用0个工作进程，表示在主进程中加载数据（Windows环境下常用）
)

# 遍历数据加载器，获取一个批次的数据进行可视化
for step, (b_x,b_y) in enumerate(train_loader):
    if step > 0 :  # 只取第一个批次的数据后就退出循环
        break
# 将图像数据从tensor转换为numpy数组，并去除多余的维度（batch_size, 1, 224, 224）-> (batch_size, 224, 224)
batch_x = b_x.squeeze().numpy()
# 将标签数据从tensor转换为numpy数组
batch_y = b_y.numpy()
# 获取FashionMNIST数据集的类别名称列表
class_labels = train_data.classes
# 打印所有类别名称
print(class_labels)

# 创建一个画布，设置图像大小为12x5英寸
plt.figure(figsize=(12,5))
# 遍历当前批次中的所有图像并进行可视化
for ii in np.arange(len(batch_y)):
    # 创建子图，布局为4行16列，当前是第ii+1个子图（索引从1开始）
    plt.subplot(4,16,ii+1)
    # 显示灰度图像，batch_x[ii,:,:]表示第ii张图像的所有像素数据
    plt.imshow(batch_x[ii,:,:],cmap = plt.cm.gray)
    # 设置子图标题为对应的类别名称，字体大小为10
    plt.title(class_labels[batch_y[ii]],size = 10)
    # 关闭坐标轴显示，使图像更清晰
    plt.axis("off")
    # 调整子图之间的间距，wspace=0.05表示水平间距很小
    plt.subplots_adjust(wspace = 0.05)
# 显示最终的图像网格
plt.show()