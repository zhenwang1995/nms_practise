# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 23:09:28 2020

@author: 86182
"""

import numpy as np
import matplotlib.pyplot as plt
'''
def trans(anchor):     #将原始anchor转换成 w,h,中心x，中心y
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor [1] +1
    x_c = anchor[0] + 0.5 * (w - 1)
    y_c = anchor[1] + 0.5 * (h - 1)
    return w,h,x_c,y_c

def v_trans(w, h, x_c, y_c):  #将计算好的w,h,中心x，中心y转化回坐标系表示，用来画图
    anchor[0] = x_c - 0.5 * (w - 1)
    anchor[1] = y_c - 0.5 * (h - 1)
    anchor[2] = int(w - 1 + anchor[0])
    anchor[3] = int(h - 1 + anchor [1])
    return anchor


def makeanchors(tr_anchor, ratio, ax): #在不同比值下画框
    size = tr_anchor[0] * tr_anchor[1]
    for i, rat in enumerate(ratio):
        size_ratios = size/rat
        ws = np.sqrt(size_ratios)
        hs = ws * rat
        anchor_ = v_trans(ws, hs, tr_anchor[2], tr_anchor[3])
        rect = plt.Rectangle((anchor_[0], anchor_[1])
                             , anchor_[2]-anchor_[0], anchor_[3]-anchor_[1], fill=False)
        ax.add_patch(rect)
        print(anchor_)

if __name__ == '__main__':
    base_size = 16
    ratio = (0.5, 1, 2)
    anchor = [0, 0, base_size - 1, base_size - 1]
    scales = (8 ,16, 32)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    plt.xlim((-400, 400))
    plt.ylim((-400, 400))
    w, h, x_c, y_c = trans(anchor)
    for i in scales:    #放大8,16,32倍
        ws = w * i
        hs = h * i
        tr_anchor = [ws, hs, x_c, y_c]
        anchor_ = makeanchors(tr_anchor, ratio, ax)
    plt.show()
'''
def generate_anchors(base_size,ratios ,scales):
    base_anchor = np.array([1,1,base_size,base_size]) -1
    print ('base anchors',base_anchor)
    ratio_anchor = _ratio_enum(base_anchor,ratios)
    print ('anchors after ratio',ratio_anchor)
    print(ratio_anchor.shape[0])
    anchors = np.vstack([_scale_enum(ratio_anchor[i, :],scales)
                         for i in range(ratio_anchor.shape[0])])
    print ("anchors after ration and scale",anchors)
    
    return anchors


def _ratio_enum(anchor,ratios):       #make anchor change by ratios
    w,h,x_ctr,y_ctr = _whctrs(anchor)
    size = w*h
    size_ratios = size/ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchor(ws,hs,x_ctr,y_ctr)
    return anchors
    
def _whctrs(anchor):     #transfer (x,y) to w,h
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    print(w,h,x_ctr,y_ctr)
    return w,h,x_ctr,y_ctr

def _scale_enum(anchor,scales):   #make anchor change by scales
    w,h,x_ctr,y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchor(ws,hs,x_ctr,y_ctr)
    return anchors
def _mkanchor(ws,hs,x_ctr,y_ctr):  #transfer w,h to (x,y)
    print(ws,hs,x_ctr,y_ctr)
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    #ws = ws[:,]
    #hs = hs[:,]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

if __name__ == '__main__':
    a = generate_anchors(16,[0.5,1,2],2**np.arange(3,6))
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    plt.xlim((-400, 400))
    plt.ylim((-400, 400))
    print (a)
    for anchor in a:
        rect = plt.Rectangle((anchor[0], anchor[1])
                             , anchor[2]-anchor[0], anchor[3]-anchor[1], fill=False)
        ax.add_patch(rect)
    plt.show()


