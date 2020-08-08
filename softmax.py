# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 21:19:00 2020

@author: 86182
"""


import numpy as np

m = np.random.randn(10,10)*10 +1000
print(m)

m_row_max = m.max(axis = 1)
print(m_row_max,m_row_max.shape)

m = m - m_row_max.reshape(10,1)
print(m)

m_exp = np.exp(m)

print(m_exp,m_exp.shape)

m_exp_row_sum = m_exp.sum(axis = 1).reshape(10,1)
print(m_exp_row_sum)

softmax = m_exp / m_exp_row_sum
print(softmax)

print(softmax.sum(axis =1).reshape(10,1))