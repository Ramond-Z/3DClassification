# Homework3 3D Classification

本次作业中，我选择复现了PointNet与PointCNN两篇论文。

## 算法概述
- PointNet：Pointnet使用共享权重的MLP为点云中的每个点分别提取特征，并通过全局的maxpooling得到点云整体的特征。除此之外，PointNet还设计了两组特征变换网络（T-Net），对点云的位置以及特征进行变换以提升效果。在实现中选用了步长为一，卷积核大小为1的`nn.Conv1d`实现共享权重的MLP。

- PointCNN：PointCNN提出了`X-conv`这一卷积算子以实现点云上的卷积神经网络，具体而言，`X-conv`算子以代表点及其k近邻上点云的位置与特征作为输入，输出代表点经卷积后聚集的特征。其首先将邻域里点云的位置坐标过平移变换移动到代表点的局部坐标系下，利用MLP计算一个k*k的X-变换矩阵，并利用类似PointNet中的共享权重MLP计算将点云位置升维后的特征。将输入特征与升维得到的特征拼在一起后经过X-变换矩阵得到最终的特征，再利用一维卷积计算出代表点的特征。算法伪代码如下
    ```
    X_conv(representitive_position, neighbour_positions, neighbour_features, k):
        neighbour_positions -= representitive_position
        transformation = MLP(neighbour_positions)
        lifted_features = SharedMLP(neighbour_positions)
        features = [neighbour_features, lifted_features]
        features = transformation @ features
        representitive_feature = Conv(features)
        return representitive_feature
    ```
    利用如上定义的卷积算子，我们可以设计点云上的卷积层，其输入为点云的位置与特征，随机选取若干代表点后，将代表点的坐标及其邻域信息输入`X-conv`算子，得出若干代表点经卷积后得到的特征。将若干卷积层堆叠起来，逐渐减少带标点的个数，增加特征维度，即可获得整个点云的特征。在实现中，最终不一定需要保证仅保留一个代表点，而是可以将多个代表点的特征通过一个共同的分类层后再取平均值，作为最终的分类结果。

## 实现细节

实验中选取PointNet代码仓库提供的ModelNet40数据集，对于PointNet，其输入点的个数为2048个，对PointCNN而言，遵从论文中设定，选取1024个点作为输入，并在训练时选取$n\thicksim N(1024, 128^2)$个点作为输入。

本次作业选取PyTorch框架进行实现，并利用Huggingface Accelerator以简化训练代码。不同于以上两篇论文中的操作，在训练时选取了更优的`AdamW`优化器与`CosineAnnealingWarmRestarts`scheduler进行训练。由于计算资源有限，仅在256个epoch后便停止训练。这可能带来了较大的性能损失。

## 实现结果

训练过程中的loss与训练集上的准确度如下图所示
![[loss.jpg]]
![[acc.jpg]]
在训练256个epoch后，PointNet在测试集上的准确度达到了70.29%，PointCNN在测试集上的准确度达到了80.3%