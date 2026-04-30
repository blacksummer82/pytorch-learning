# 导入PyTorch核心库，提供张量操作和自动微分功能
import torch
# 导入神经网络模块，包含各种层和激活函数
from torch import nn
# 导入模型摘要工具，用于显示网络结构和参数统计（需要安装torchsummary）
from torchsummary import summary

# 定义LeNet-5卷积神经网络类，继承自nn.Module基类
class LeNet(nn.Module):
    """
    LeNet-5网络架构实现，由Yann LeCun于1998年提出。
    主要用于手写数字识别任务（如MNIST数据集）。
    网络结构：输入->C1->S2->C3->S4->F5->F6->F7->输出
    """
    def __init__(self):
        # 调用父类构造函数，初始化nn.Module的基础功能（如参数管理、梯度追踪等）
        super().__init__()
        
        # C1: 第一层卷积层 - 特征提取阶段开始
        # 参数说明：
        # - in_channels=1: 输入通道数（灰度图像为1通道）
        # - out_channels=6: 输出通道数（生成6个特征图）
        # - kernel_size=5: 卷积核大小为5x5像素窗口
        # - padding=2: 零填充2像素，保持空间维度从28x28到28x28
        # 计算：输出尺寸 = (28 + 2*2 - 5)/1 + 1 = 28，所以输出形状为[batch, 6, 28, 28]
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)
        
        # Sigmoid激活函数：将线性变换后的值映射到(0,1)区间，引入非线性特性
        # 公式：σ(x) = 1 / (1 + e^(-x))
        self.sigmoid = nn.Sigmoid()
        
        # S2: 第二层池化层（平均池化）- 降维和特征压缩
        # 参数说明：
        # - kernel_size=2: 池化窗口大小为2x2像素区域
        # - stride=2: 步长为2，不重叠地滑动窗口，使尺寸减半（28x28 -> 14x14）
        # 作用：减少计算量、增强平移不变性、防止过拟合
        self.s2 = nn.AvgPool2d(2,stride=2)
        
        # C3: 第三层卷积层 - 进一步提取高级特征
        # 参数说明：
        # - in_channels=6: 输入通道数（来自S2的6个特征图）
        # - out_channels=16: 输出通道数（生成16个更抽象的特征图）
        # - kernel_size=5: 卷积核大小仍为5x5像素窗口
        # - padding默认为0，无额外填充（14+0-5)/1+1 = 10，输出形状为[batch, 16, 10, 10]
        self.c3 = nn.Conv2d(6, 16, 5)
        
        # S4: 第四层池化层（平均池化）- 再次降维和特征压缩
        # 参数说明：
        # - kernel_size=2: 池化窗口大小为2x2像素区域  
        # - stride=2: 步长为2，使尺寸再次减半（10x10 -> 5x5）
        # 输出形状变为[batch, 16, 5, 5]
        self.s4 = nn.AvgPool2d(2, 2)

        # Flatten层：将多维特征图展平为一维向量，准备进入全连接层分类器阶段
        # 将[batch, 16, 5, 5]展平为[batch, 16*5*5] = [batch, 400]
        self.flatten = nn.Flatten()
        
        # F5: 第五层全连接层（隐藏层1）- 特征组合与转换阶段开始  
        # 参数说明：
        # - in_features=400: 输入特征数（来自flatten层的400维向量）
        # - out_features=120: 输出神经元数量（压缩表示学习）
        # 注意：这里省略了激活函数，但原始LeNet使用sigmoid激活，建议后续补充激活函数  
        self.f5 = nn.Linear(400, 120)
        
        # F6: 第六层全连接层（隐藏层2）- 继续特征抽象和模式识别  
        # 参数说明：
        # - in_features=120: 输入特征数（来自F5的120维向量）
        # - out_features=84: 输出神经元数量（进一步压缩表示）
        # 同样缺少显式激活函数，实际应用中通常需要添加激活函数  
        self.f6 = nn.Linear(120, 84)
        
        # F7: 第七层全连接层（输出层）- 最终分类决策层  
        # 参数说明：
        # - in_features=84: 输入特征数（来自F6的84维向量）
        # - out_features=10: 输出神经元数量（对应10个类别，如MNIST中的0-9数字）
        # 输出的是未经softmax归一化的logits，训练时配合CrossEntropyLoss使用（内部含softmax）
        self.f7 = nn.Linear(84, 10)

    def forward(self, x):
        """
        前向传播函数：定义数据在网络中的流动路径。
        PyTorch会自动根据此方法构建计算图并支持反向传播。
        
        Args:
            x: 输入张量，形状为[batch_size, channels, height, width]
               对于MNIST数据，典型形状为[batch_size, 1, 28, 28]
               
        Returns:
            logits: 未归一化的预测分数，形状为[batch_size, num_classes]
                    对于MNIST，形状为[batch_size, 10]
        """
        # 第一阶段：特征提取（卷积层+激活函数+池化层交替）
        
        # C1层：卷积操作提取局部特征（边缘、角点等低级特征）
        # 输入:[batch, 1, 28, 28] -> 输出:[batch, 6, 28, 28]
        x = self.c1(x)
        
        # 应用Sigmoid激活函数，引入非线性能力以学习复杂模式  
        # 输入:[batch, 6, 28, 28] -> 输出:[batch, 6, 28, 28]
        x = self.sigmoid(x)
        
        # S2层：平均池化降低空间维度，保留重要信息同时减少计算负担  
        # 输入:[batch, 6, 28, 28] -> 输出:[batch, 6, 14, 14]
        x = self.s2(x)
        
        # C3层：第二次卷积提取更高层次的抽象特征（纹理、部件等中级特征）
        # 输入:[batch, 6, 14, 14] -> 输出:[batch, 16, 10, 10]
        x = self.c3(x)
        
        # 再次应用Sigmoid激活函数，增强模型表达能力  
        # 输入:[batch, 16, 10, 10] -> 输出:[batch, 16, 10, 10]
        x = self.sigmoid(x)
        
        # S4层：第二次池化进一步压缩空间维度  
        # 输入:[batch, 16, 10, 10] -> 输出:[batch, 16, 5, 5]
        x = self.s4(x)
        
        # 第二阶段：分类决策（全连接层进行全局推理）
        
        # Flatten层：将二维特征图展平为一维向量，适配全连接层输入要求  
        # 输入:[batch, 16, 5, 5] -> 输出:[batch, 400] （16×5×5=400）
        x = self.flatten(x)
        
        # F5层：第一个全连接层，将局部特征整合成全局表示  
        # 输入:[batch, 400] -> 输出:[batch, 120]
        # 注意：此处缺少激活函数，建议添加ReLU或Sigmoid以获得更好效果  
        x = self.f5(x)
        
        # F6层：第二个全连接层，进一步提炼特征表示  
        # 输入:[batch, 120] -> 输出:[batch, 84]
        # 同样缺少激活函数，可能需要补充  
        x = self.f6(x)
        
        # F7层：输出层，产生每个类别的得分（logits）  
        # 输入:[batch, 84] -> 输出:[batch, 10]
        # 这些值将在损失函数中通过softmax转换为概率分布  
        x = self.f7(x)
        
        # 返回最终的分类预测结果（未经softmax的原始分数）
        return x

# 主程序入口：当直接运行此脚本时执行测试代码（被导入时不执行）
if __name__ == "__main__":
    # 检测设备可用性：优先使用GPU（CUDA），若不可用则回退到CPU
    # torch.cuda.is_available()检查是否有可用的NVIDIA GPU及正确的驱动环境  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 实例化LeNet模型对象，并将所有参数移动到指定设备（GPU或CPU）
    # .to(device)确保模型权重和缓冲区位于正确设备上以便高效计算  
    model = LeNet().to(device)
    
    # 打印模型详细信息：包括每层的输出形状、参数量、总参数量等统计信息  
    # summary(model, input_size)需要指定输入数据的形状（不含batch维度）
    # 对于MNIST数据集，输入为单通道28x28像素的灰度图像，故输入形状为(1, 28, 28)
    # 输出内容示例：
    # ==================================================================
    # Layer (type)                 Output Shape              Param #
    # ==================================================================
    # Conv2d-1                   [-1, 6, 28, 28]                  156
    # Sigmoid-2                  [-1, 6, 28, 28]                    0
    # AvgPool2d-3                [-1, 6, 14, 14]                    0
    # ...
    # ==================================================================
    # Total params: 61,706
    # Trainable params: 61,706
    # Non-trainable params: 0
    # ------------------------------------------------------------------
    print(summary(model, (1, 28, 28)))
