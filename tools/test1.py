# -*- coding: UTF-8 -*-
import numpy as np

x = np.arange(6, 0, -1)
arr1 = x[:-2]
arr2 = x[1:-1]
arr3 = x[2:]
# 使用乘法操作计算每个数组的元素积
result = arr1 * arr2 * arr3
print(x)
print(x[-4:])
print(result)
