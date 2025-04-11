# torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None)

# 1.retain_grad参数 保留计算图
import torch
# w1 = torch.tensor([1.], requires_grad=True)
# x = torch.tensor([2.],requires_grad=True)
# # y=(x+w)*(w+1)
# a = torch.add(w1, x)
# b = torch.add(w1, 1)  # 1是标量，表示要加到每个元素上的值
# y = torch.mul(a, b)
#
# # 第一次求导，设置 retain_graph=True，保留计算图
# y.backward(retain_graph=True)
# print(w1.grad)
# # 第二次求导成功
# y.backward()

# 2.grad_tensors参数 传入backward函数的梯度权重
w = torch.tensor([1.], requires_grad=True)  # 小数点是为了明确指定张量的数据类型为浮点数
x = torch.tensor([2.],requires_grad=True)
# y=(x+w)*(w+1)
a = torch.add(w, x)
b = torch.add(w, 1)  # 1是标量，表示要加到每个元素上的值

y0 = torch.mul(a, b)  # y0 = (x+w) * (w+1)
y1 = torch.add(a, b)  # y1 = (x+w) + (w+1)    dy1/dw = 2

# 把两个loss拼接到一起
loss = torch.cat([y0, y1], dim=0)
# 设置两个loss的权重  y0 的权重是 1，y1 的权重是 2
grad_tensors = torch.tensor([1.,2.])  # 为什么要加符号.
#  gradient 传入 torch.autograd.backward()中的grad_tensors
loss.backward(gradient=grad_tensors)
# 最终的 w 的导数由两部分组成。∂y0/∂w * 1 + ∂y1/∂w * 2 = x+2w+1 +2*2 = 2+2+1+4 = 9
print("w.grad", w.grad)

# torch.autograd.grad() 求取梯度
# torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None,
# create_graph=False, only_inputs=True, allow_unused=False)
# outputs: 用于求导的张量，【如 loss】
# inputs: 需要梯度的张量
# create_graph: 创建导数计算图，用于高阶求导
# retain_graph:保存计算图
# grad_outputs: 多梯度权重计算
# 【注意！！！！！】
# torch.autograd.grad()的返回结果是一个 tunple，需要取出【第0个元素】才是真正的梯度。
x1 = torch.tensor([3.], requires_grad=True)
y1 = torch.pow(x1, 2)  # y = x**2
# 若求二阶导，需设置create_graph=True 让一阶导函数grad_1也拥有计算图
grad_1 = torch.autograd.grad(y1, x1, create_graph=True)  # grad_1 = dy/dx = 2x = 2 * 3 = 6
print("\ngrad_1", grad_1)  # (tensor([6.], grad_fn=<MulBackward0>),)
# 求二阶导
grad_2 = torch.autograd.grad(grad_1[0], x1)  # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2
print("\ngrad_2", grad_2)  # (tensor([2.]),)

w2 = torch.tensor([1.], requires_grad=True)
x2 = torch.tensor([2.], requires_grad=True)
# 进行 4 次反向传播求导，每次最后都没有清零
for i in range(4):
    a2 = torch.add(w2, x)
    b2 = torch.add(w2, 1)
    y = torch.mul(a2, b2)
    y.backward()
    print("\nw2.grad", w2.grad)

# inplace 和 非inplace操作
# inplace 操作有a += x，a.add_(x) 改变后的值和原来的值内存地址是同一个
# 非 inplace 操作有a = a + x，a.add(x) 改变后的值和原来的值内存地址不是同一个
print("\n非inplace 操作")
a3 = torch.ones((1,))
print("\nid(a3):{} a3:{}".format(id(a3), a3))
# 非inplace 操作 内存地址不一样
a3 = a3+torch.ones((1,))
print("\nid(a3):{} a3:{}".format(id(a3), a3))

print("\ninplace 操作")
a4 = torch.ones((1,))
print("\nid(a4):{} a4:{}".format(id(a4), a4))
# inplace 操作，内存地址一样
a4 += torch.ones((1,))
print("\nid(a4):{} a4:{}".format(id(a4), a4))






