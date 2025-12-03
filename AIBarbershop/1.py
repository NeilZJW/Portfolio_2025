# -*- coding: utf-8 -*-
# @Author  : Neil
# @Time    : 2024/5/3 14:51

# import torch
#
#
# # print(torch.zeros(1, 3, 1, 1))
# def make_kernel(k):
#     """
#     create convlutional kernel
#     :param k: 数组k
#     :return: 二维卷积核
#     """
#     k = torch.tensor(k, dtype=torch.float32)
#     if k.ndim == 1:
#         k = k[None, :] * k[:, None]
#         print(k)
#     k /= k.sum()
#     return k
#
# print(make_kernel([1, 2 ,3]))



# Name = "Neil"
# step = 1
#
# print("{}_{:04}".format(Name, step))
#
#

# a = ['E:/BBShop/Barbershop/input/face/53.png']
# output_name = a.split("/")[-1].split(".")[0]
# print(a)
a = "ZRX.png"
print(a.split(".")[0])

