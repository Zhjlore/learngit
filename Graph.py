# 动态图：适合研究和开发阶段，提供灵活性和易用性，便于快速迭代和调试。
# 静态图：适合部署阶段，提供更高的性能和资源效率，适合固定结构的模型和生产环境。
# PyTorch 默认使用动态图机制，而像 TensorFlow（早期版本）这样的框架默认使用静态图机制。

# 动态图机制
# 1.逐步执行
import torch
x = torch.tensor([1.0, 2.0, 3.0],requires_grad=True)
y = x*2
z = y.sum()
# 如果需要多次计算梯度，需要在每次调用 backward() 之前清空梯度，否则梯度会累加
# 如果你需要多次调用 backward()，可以在第一次调用时设置 retain_graph=True，以保留计算图
# 在反向传播过程中，PyTorch 会自动计算每个张量的梯度，并将结果存储在 .grad 属性中
# 在一次前向传播中计算多个损失，并将这些损失组合起来进行一次反向传播，从而避免多次调用 backward()
# 第一次反向传播
z.backward(retain_graph=True)
print("x.grad:", x.grad)  # tensor([2., 2., 2.])
# 清空梯度
x.grad.zero_()
# 第二次反向传播
z.backward()
# 打印x的梯度，就是y对x的导数
print("x.grad:", x.grad)  # tensor([2., 2., 2.])

# 2.动态调整
import torch.nn as nn


class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10,1)

    def forward(self, x):
        if x.mean() > 0:  # 动态判断
            return self.fc(x)
        else:
            return self.fc(x)*2


model = DynamicModel()
x = torch.randn((1, 10))
output = model(x)
print("\n动态调整 output:", output)

# 避免多次反向传播
import torch.optim as optim
model1 = nn.Linear(10,1)
optimizer = optim.SGD(model1.parameters(), lr=0.01)

# 创建输入和目标
x = torch.randn(1,10)
y = torch.randn(1,1)

# 前向传播
output1 = model1(x)
# 计算多个损失
# nn.MSELoss() 是一个类 它需要先被实例化，然后再调用这个实例来计算损失
# 在实例化时传递参数，这样可以简化代码
# 下面的写法实际上是在 实例化一个类并立即调用它的 __call__ 方法
# 在 Python 中，当你在对象后面加上 ()，实际上是在调用该对象的 __call__ 方法
loss1 = nn.MSELoss()(output, y)
loss2 = torch.abs(output-y).mean()  # 自定义误差（绝对误差）

# 组合损失
total_loss = loss1+loss2  # 也可加权求和

# 查看模型的输出和损失
print("\n避免多次反向传播例子结果：")
print("模型输出:", output1)
print("均方误差损失:", loss1)
print("绝对误差均值损失:", loss2)
print("总损失:", total_loss)

# 反向传播
optimizer.zero_grad()  # 清空梯度
total_loss.backward()   # 一次反向传播
optimizer.step()        # 更新参数

# 查看更新后的模型参数
print("\n更新后的模型权重:", model1.weight)
print("更新后的模型偏置:", model1.bias)




