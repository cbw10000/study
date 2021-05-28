import os
import random
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
import numpy as np
from PIL import Image

train_data = paddle.dataset.mnist.train()
train_data = paddle.reader.shuffle(train_data, 100)
train_data = paddle.batch(train_data, 100)


# 定义模型结构
class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        # 定义一个卷积层，使用relu激活函数
        self.conv1 = Conv2D(num_channels=1, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
        # 定义一个池化层，池化核为2，步长为2，使用最大池化方式
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # 定义一个卷积层，使用relu激活函数
        self.conv2 = Conv2D(num_channels=20, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
        # 定义一个池化层，池化核为2，步长为2，使用最大池化方式
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # 定义一个全连接层，输出节点数为10
        self.fc = Linear(input_dim=980, output_dim=10, act='softmax')

    # 定义网络的前向计算过程
    def forward(self, inputs, label):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = fluid.layers.reshape(x, [x.shape[0], 980])
        x = self.fc(x)
        if label is not None:
            acc = fluid.layers.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
from visualdl import LogWriter

with fluid.dygraph.guard(place):
    model = MNIST()
    model.train()
    EPOCH_NUM = 5
    BATCH_SIZE = 100
    # 定义学习率，并加载优化器参数到模型中
    total_steps = (int(60000 // BATCH_SIZE) + 1) * EPOCH_NUM
    lr = fluid.dygraph.PolynomialDecay(0.01, total_steps, 0.001)

    # 使用Adam优化器
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, parameter_list=model.parameters())

    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_data()):
            # 准备数据，变得更加简洁
            img_data = np.array([x[0] for x in data]).astype('float32').reshape(-1, 1, 28, 28)
            # 获得图像标签数据，并转为float32类型的数组
            label_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)
            image = fluid.dygraph.to_variable(img_data)
            label = fluid.dygraph.to_variable(label_data)

            # 前向计算的过程，同时拿到模型输出值和分类准确率
            predict, acc = model(image, label)
            avg_acc = fluid.layers.mean(acc)

            # 计算损失，取一个批次样本损失的平均值
            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)

            # 每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(),
                                                                            avg_acc.numpy()))
            # 后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

        # 保存模型参数和优化器的参数
        fluid.save_dygraph(model.state_dict(), './checkpoint/mnist_epoch{}'.format(epoch_id))
        fluid.save_dygraph(optimizer.state_dict(), './checkpoint/mnist_epoch{}'.format(epoch_id))

test_data = paddle.dataset.mnist.test()
# 乱序、缓冲区
test_data = paddle.reader.shuffle(test_data, 100)
# 抽取100张
test_data = paddle.fluid.io.firstn(test_data, 100)
test_data = paddle.batch(test_data, 100)

with fluid.dygraph.guard():
    print('********************随机抽取原始mnist测试集100张图片进行测试 ********************')
    # 加载模型参数
    model = MNIST()
    model_state_dict, _ = fluid.load_dygraph('checkpoint/mnist_epoch4.pdopt')
    model.load_dict(model_state_dict)

    model.eval()

    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(test_data()):
        x_data = np.array([x[0] for x in data]).astype('float32').reshape(-1, 1, 28, 28)
        # 获得图像标签数据，并转为float32类型的数组
        y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)
        img = fluid.dygraph.to_variable(x_data)
        label = fluid.dygraph.to_variable(y_data)
        prediction, acc = model(img, label)
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = fluid.layers.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

    # 计算多个batch的平均损失和准确率
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print('测试结果：loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))