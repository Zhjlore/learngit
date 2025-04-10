
# 张量操作与线性回归

import torch

# torch.cat() 除了拼接的维度外，其他维度的大小必须一致
# 在原来的维度上进行拼接
t1 = torch.tensor([[1,2],[3,4]])
t2 = torch.tensor([[5,6],[7,8]])
t_1 = torch.cat([t1, t2], dim=0)
t_2 = torch.cat([t1, t2], dim=1)
print("\nt_1:{} shape:{} \nt_2:{} shape:{}".format(t_1, t_1.shape, t_2, t_2.shape))

# torch.stack()
# 在新创建的维度上进行拼接 在拼接的维度上新增N的维度 N为待拼接的张量的数量
# 1.两个2维tensor张量进行stack
t3 = torch.tensor([[1,2,3], [4,5,6]])
t4 = torch.tensor([[7,8,9], [10,11,12]])
t_stack0 = torch.stack([t3,t4],dim=0)
print("\nt3和t4在维度 0 上堆叠：")  # N为待拼接的张量的数量
print("\nt3和t4 t_stack0.shape:{}".format(t_stack0.shape)) # [2, 2, 3] 在0维度上新增一个N=2的维度
print("\nt3和t4 t_stack0", t_stack0)

t_stack1 = torch.stack([t3,t4],dim=1)
print("\nt3和t4 在维度 1 上堆叠：")
print("\nt3和t4 t_stack1.shape:{}".format(t_stack1.shape)) # [2, 2, 3] 在1维度上新增一个N=2的维度
print("\nt3和t4 t_stack1", t_stack1)

t_stack2 = torch.stack([t3,t4],dim=2)
print("\nt3和t4 在维度 2 上堆叠：")
print("\nt3和t4 t_stack2.shape:{}".format(t_stack2.shape)) # [2, 3, 2] 在2维度上新增一个N=2的维度
print("\nt3和t4 t_stack2", t_stack2)

# 2.三个2维tensor张量进行stack
t5 = torch.tensor([[13,14,15], [16,17,18]])
r1 = torch.stack([t3,t4,t5],dim=0)
print("\nt3、t4、t5 在维度 0 上堆叠：")
print("\nt3、t4、t5 r1.shape:{}".format(r1.shape)) # [3, 2, 3] 在0维度上新增一个N=3的维度
print("\nt3、t4、t5 r1", r1)

r2 = torch.stack([t3,t4,t5],dim=1)
print("\nt3、t4、t5 在维度 1 上堆叠：")
print("\nt3、t4、t5 r2.shape:{}".format(r2.shape)) # [2, 3, 3] 在1维度上新增一个N=3的维度
print("\nt3、t4、t5 t_stack1", r2)

r3 = torch.stack([t3,t4,t5],dim=2)
print("\nt3、t4、t5 在维度 2 上堆叠：")
print("\nt3、t4、t5 r3.shape:{}".format(r3.shape)) # [2, 3, 3] 在2维度上新增一个N=3的维度
print("\nt3、t4、t5 r3", r3)

# torch.chunk()
# torch.chunk(input, chunks, dim=0)
# 将张量按照维度 dim 进行平均切分。若不能整除，则最后一份张量小于其他张量
tensor1 = torch.ones((3, 8))
list_of_tensors1 = torch.chunk(tensor1, dim=0, chunks=2) # chunks: 要切分的份数
for i, t in enumerate(list_of_tensors1):
    print("\n第{}个张量：{},shape is {}".format(i+1,t,t.shape)) # [2,3] [2,3] [2,2]

# torch.split()
# torch.split(tensor, split_size_or_sections, dim=0)
# 将张量按照维度 dim 进行平均切分。可以指定每一个分量的切分长度。
tensor2 = torch.ones((2, 5))
list_of_tensors2 = torch.chunk(tensor2, dim=1, chunks=2)
for i, t in enumerate(list_of_tensors2):
    print("\n第{}个张量：{}, shape is {}".format(i+1, t,t.shape))

# torch.index_select()
# torch.index_select(input, dim, index, out=None)
# 在维度 dim 上，按照 index 索引取出数据拼接为张量返回
# torch.randint(low,high,size)
tensor3 = torch.randint(0, 9, size=(3, 3))
# 注意 idx 的 dtype 不能指定为 torch.float
idx = torch.tensor([0,2], dtype=torch.long)
# 取出第 0 行和第 2 行
t_select = torch.index_select(tensor3, dim=0, index=idx)
print("\ntensor3:{}\nt_select:{}".format(tensor3, t_select))

# torch.masked_select()
# torch.masked_select(input, mask, out=None)
# 按照 mask 中的 True 进行索引拼接得到一维张量返回
tensor4 = torch.randint(0, 9, size=(3, 3))
# mask = tensor4.le(5) # 生成了一个布尔掩码，表示张量 tensor 中哪些元素小于等于 5
mask = tensor4.gt(5)
# 取出大于 5 的数
t_select_mask = torch.masked_select(tensor4,mask)
print("\ntensor4:{}\nmask:{}\n t_select_mask:{}".format(tensor4,mask,t_select_mask))

# torch.reshape()
# 变换张量的形状。当张量在内存中是连续时，返回的张量和原来的张量共享数据内存，改变一个变量时，另一个变量也会被改变
# 生成 0 到 8 的随机排列
tensor5 = torch.randperm(8)
# -1 表示这个维度是根据其他维度计算得出的
t_reshape = torch.reshape(tensor5, shape=(-1,2,2))
print("\ntensor5:{} \nt_reshape:{} ".format(tensor5, t_reshape))
# 修改张量 tensor5 的第 0 个元素，张量 t_reshape 也会被改变
tensor5[0] = 1024
print("\ntensor5:{} \nt_reshape:{} ".format(tensor5, t_reshape))
print("\ntensor5内存地址：", id(tensor5))
print("\nt_reshape内存地址：", id(t_reshape))

# torch.transpose()
# torch.transpose(input, dim0, dim1)
# 交换张量的两个维度。常用于图像的变换，比如把c*h*w变换为h*w*c
# rand在区间 [0, 1) 上生成均匀分布
tensor6 = torch.rand(size=(2, 3, 4))
t_transpose = torch.transpose(tensor6,dim0=1,dim1=2)  # 交换维度1和维度2
print("\ntensor6.shape:{} t_transpose.shape:{}".format(tensor6.shape,t_transpose.shape))

# torch.t()
# 2 维张量转置，对于 2 维矩阵而言，等价于torch.transpose(input, 0, 1)

# torch.squeeze()
# torch.squeeze(input, dim=None, out=None)
# 【压缩】长度为 1 的维度
#  dim: 若为 None，则移除所有长度为 1 的维度；若指定维度，则当且仅当该维度长度为 1 时可以移除
# 维度 0 和 3 的长度是 1
t = torch.rand((1, 2, 3, 1))
# 可以移除维度 0 和 3
t_sq = torch.squeeze(t)
# 可以移除维度 0
t_0 = torch.squeeze(t, dim=0)
# 不能移除 1
t_1 = torch.squeeze(t, dim=1)
print("t.shape: {}".format(t.shape)) # [1, 2, 3, 1]
print("t_sq.shape: {}".format(t_sq.shape)) # [2, 3]
print("t_0.shape: {}".format(t_0.shape)) # [2, 3, 1]
print("t_1.shape: {}".format(t_1.shape)) # [1, 2, 3, 1]

# torch.unsqueeze()
# torch.unsqueeze(input, dim)
# 根据 dim 【扩展】维度，长度为 1
x = torch.tensor([1,2,3,4])
print(x.shape)
x_unsqueezed = torch.unsqueeze(x, dim=0)
print(x_unsqueezed.shape)
print(x_unsqueezed)  # tensor([[1, 2, 3, 4]])

# torch.add()
# torch.add(input, other, out=None)
# torch.add(input, other, *, alpha=1, out=None)
# 逐元素计算 input + alpha * other。因为在深度学习中经常用到先乘后加的操作
# input: 第一个输入张量。
# other: 第二个输入张量或标量

# 1.标量加法
x11 = torch.tensor(2.0)
x22 = torch.tensor(3.0)
result1 = torch.add(x11,x22)
print("\nresult1", result1)

# 2.张量加法
x1 = torch.tensor([1.0, 2.0, 3.0])
y1 = torch.tensor([4.0, 5.0, 6.0])
result2 = torch.add(x1, y1)
print("\nresult2", result2)  # tensor([5., 7., 9.])

# 3.广播机制
# 如果两个张量的形状不一致，但符合广播规则，torch.add() 会自动广播较小的张量以匹配较大的张量
x2 = torch.tensor([[1.0,2.0,3.0]])
y2 = torch.tensor([4.0, 5.0, 6.0])
result3 = torch.add(x2, y2)
print("\nresult3", result3)

# 4.使用alpha函数
x4 = torch.tensor([1.0,2.0,3.0])
y4 = torch.tensor([4.0, 5.0, 6.0])
result4 = torch.add(x4, y4, alpha=2)
print("\nresult4", result4)

# torch.addcmul()
# torch.addcmul(input, tensor1, tensor2, *, value=1, out=None)
input_tensor = torch.tensor([1.0, 2.0, 3.0])
tensor111 = torch.tensor([2.0, 3.0, 4.0])
tensor222 = torch.tensor([3.0, 2.0, 1.0])
result_addmul = torch.addcmul(input_tensor, tensor111, tensor222)
print("\n基本用法result", result_addmul)  # 输出: tensor([7., 8., 7.])

# 使用value参数
result_value = torch.addcmul(input_tensor, tensor111, tensor222, value=0.5)
print("\nresult_value", result_value)  # 输出: tensor([4., 5., 5.])





