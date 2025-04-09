
# Tensor 与 Variable

# tensor 创建方法
# torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)
import numpy as np
import torch

# 外层括号是函数调用的括号 内层括号表示(3,3)是一个元组
arr1 = np.ones((3, 3))
print("arr的数据类型：", arr1.dtype)
# tensor是默认创建在cpu上的

# 1.直接创建tensro
t1 = torch.tensor(arr1)
# 创建存放在 GPU 的数据
# t1 = torch.tensor(arr, device='cuda')
print(t1.device)
print(t1)

# 2.从 numpy 创建 tensor
# torch.from_numpy(ndarray)
# 利用这个方法创建的 tensor 和原来的 ndarray 共享内存，当修改其中一个数据，另外一个也会被改动。
arr2 = np.array([[1,2,3], [4,5,6]])
t2 = torch.from_numpy(arr2) # torch.from_numpy() 创建的张量只是对 NumPy 数组的内存视图，而不是创建一个独立的副本

# # 修改arr tensor也会被改变
# print("\n修改arr---------------")
# arr2[0][0] = 0
# print("numpy arr:", arr2)
# print("tensor:", t2)

# 修改tensor arr也会被改变
print("\n-----------------修改tensor---------------")
t2[0][0] = -1
print("numpy arr:", arr2)
print("tensor:", t2)

# 3.根据数值创建tensor
# torch.zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# 3.1 torch.zeros()
out_t = torch.tensor([1]) # 指定out形状
t3 = torch.zeros((3, 3), out=out_t)
print(t3, '\n', out_t)
# id是取内存地址 最终t3和out_t是一个内存地址
print(id(t3), id(out_t), id(t3) == id(out_t))

# 3.2 torch.zeros_like
#torch.zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
# 创建一个二维张量
x = torch.tensor([[1, 2], [3, 4]])
# 创建一个与x形状相同的零张量
zero_like_x = torch.zeros_like(x)
print("\n与 x 形状相同且数据类型为 int32 的零张量zero_like_x：", zero_like_x)

# 指定数据类型
y = torch.tensor([[1.0,2.0],[3.0,4.0]],dtype = torch.float32)
print("\n原始张量 y：", y)
zero_like_y = torch.zeros_like(y,dtype = torch.int32)
print("\n与 y 形状相同且数据类型为 int32 的零张量zero_like_y：", zero_like_y)

# 3.3 torch.arrange()
# torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# start: 数列起始值
# end: 数列结束值，开区间，取不到结束值
# step: 数列公差，默认为 1
t4 = torch.arange(5,10,2, dtype=torch.float32)
print("\nt4:",t4)

# 3.4 创建均分的 1 维张量
t_linspace = torch.linspace(2,10,6)
print("\nt_linespace",t_linspace)

# 3.5 创建自定义数值的张量 torch.full() torch.full_like()
# torch.full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
t5 = torch.full((3,3),2)
print("\nt5:" ,t5)

# 3.6 创建对数均分的1维张量 数值区间为[start,end] 底为base
# torch.logspace(start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# start: 数列起始值
# end: 数列结束值
# steps: 数列长度 (元素个数)
# base: 对数函数的底，默认为 10
t6 = torch.logspace(2,10,5)
print("\n t6", t6)

# 3.7 创建单位对角矩阵(2 维张量)，默认为方阵 torch.eye()
# n: 矩阵行数。通常只设置 n，为方阵。
# m: 矩阵列数
t7 = torch.eye(3)
print("\n t7",t7)

# 4.根据概率创建 Tensor torch.normal()
# 4.1 torch.normal(mean, std, *, generator=None, out=None)
# 生成正态分布 (高斯分布)

# mean 为标量，std 为标量。这时需要【设置 size】
t_normal1 = torch.normal(0, 1, size=(4,))
print("\nt_normal1", t_normal1)

# mean为张量 std为标量
mean2 = torch.arange(1, 5, dtype=torch.float)
std2 = 1
t_normal2 = torch.normal(mean2, std2)
print("\nmean2:{}\nstd2:{}".format(mean2, std2))
print("\n t_normal2", t_normal2)

# std为张量 mean为标量
mean3 = 1
std3 = torch.arange(0, 4,dtype=torch.float)
t_normal3 = torch.normal(mean3, std3)
print("\nmean3:{}\nstd3:{}".format(mean3, std3))
print("\n t_normal3", t_normal3)

# mean 为张量，std 为张量
mean4 = torch.arange(1, 5, dtype=torch.float)
std4 = torch.arange(1, 5, dtype=torch.float)
t_normal4 = torch.normal(mean4, std4)
print("\nmean4:{}\nstd4:{}".format(mean4, std4))
print("\n t_normal4", t_normal4)

# 4.2 torch.randn torch.randn_like()
# 功能：生成标准正态分布。
t_randn = torch.randn(2, 3)
print("\ntorch.randn 输出：", t_randn)

input_tensor1 = torch.tensor([[1., 2., 3.], [4., 5., 6.]])  # 不支持整数类型
print("\n输入张量：", input_tensor1)
t_randn_like = torch.randn_like(input_tensor1)
print("\nrandn.randn_like 输出", t_randn_like)

# 4.3 torch.rand() 和 torch.rand_like()
# 功能：在区间 [0, 1) 上生成均匀分布。
t_rand = torch.rand(2,3)
print("\ntorch.rand 输出：", t_rand)

input_tensor2 = torch.tensor([[2.,3.,4.,], [5., 6., 7.]])
t_rand_like = torch.rand_like(input_tensor2)

# 4.4 torch.randint() 和 torch.randint_like()
# 功能：在区间 [low, high) 上生成整数均匀分布
t_randint = torch.randint(0, 5,(3,)) # low,high,size
print("\ntorch.randint 输出：", t_randint)
inout_tensor3 = torch.tensor([[2.,3.,4.,], [5., 6., 7.]])
t_randint_like = torch.randint_like(inout_tensor3,0,5) # tensor,low,high
print("\nt_randint_like", t_randint_like)

# 4.5 torch.randperm()
# 生成从 0 到 n-1 的随机排列。常用于【生成索引】
t_randperm = torch.randperm(10) # n
print("\nt_randperm ", t_randperm)

# 4.6 torch.bernoulli() # 输入张量的值在 [0, 1] 范围内
# 以 input 为概率，生成伯努利分布 (0-1 分布，两点分布)
input_tensor3 = torch.tensor([[0.1, 0.2],[0.3, 0.4]])
t_bernoulli = torch.bernoulli(input_tensor3)
print("\nt_bernoulli",t_bernoulli)


