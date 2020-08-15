# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:35:54 2020

@author: 86182
"""
import cv2
import zipfile
import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]
valid_ids = COCO_IDS
cat_ids = {v: i for i, v in enumerate(valid_ids)}
#pylab.rcParams['figure.figsize'] = (8.0, 10.0)
root = 'F:\coco'  # 你下载的 COCO 数据集所在目录
#print(os.listdir(f'{root}/images'))  #展示该目录下的文件

Z = zipfile.ZipFile(f'{root}/images/train2014.zip')  #查看压缩文件里的文件列表
#print(Z.namelist())
#print(Z.namelist()[7])
#img_b = Z.read(Z.namelist()[7]) #读取图片由于 Z.read 函数返回的是 bytes，所以，我们需要借助一些其他模块来将图片数据转换为 np.uint8 形式
#print(img_b)
#print(type(img_b))

#img_flatten = np.frombuffer(img_b,'B')
#img_cv = cv2.imdecode(img_flatten, cv2.IMREAD_ANYCOLOR)
#print(img_cv.shape)
#def buffer2array(Z , image_name):
 # buffer = Z.read(image_name)
 # image = np.frombuffer(buffer,dtype = 'B')
 # img = cv2.imdecode(image,cv2.IMREAD_ANYCOLOR)
 # return img

#img = buffer2array(Z, Z.namelist()[8])
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.title('open cv')
#plt.axis('off')
datadir = root
datatype = 'train2014'
annFile = '{}/annotations/instances_{}.json'.format(datadir,datatype)
coco = COCO(annFile)

#cats = coco.loadCats(coco.getCatIds())
#nms = [cat['name'] for cat in cats]
#print('COCO categories: \n{}\n'.format(' '.join(nms)))
#nms = set([cat['supercategory'] for cat in cats])
#print('COCO supercategories: \n{}'.format(' '.join(nms)))
#,'dog','skatebord'
catIds = coco.getCatIds(catNms = ['person'])
imgIds = coco.getImgIds(catIds = catIds)
#print(imgIds)
imgIds = coco.getImgIds(imgIds = [262528])
#print(imgIds)
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
#print(img)
#I = buffer2array(Z,img['file_name'])
I = io.imread('{}/images/train2014/{}'.format(datadir,img['file_name']))


train = 'train'
data_dir = os.path.join(datadir, 'coco')
img_dir=os.path.join(data_dir, '%s2014' % train)
img_path = os.path.join(img_dir,coco.loadImgs(ids=[262528])[0]['file_name'])
ann_ids = coco.getAnnIds(imgIds=[262528])
annotations = coco.loadAnns(ids=ann_ids)
labels = np.array([cat_ids[anno['category_id']] for anno in annotations])
bboxes = np.array([anno['bbox'] for anno in annotations], dtype=np.float32)

for i  in range(0,len(bboxes)-1):
    x, y, w, h = bboxes[i]
    anno_image = cv2.rectangle(I, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)   

plt.axis('off')
plt.imshow(anno_image)
plt.show()
annFile = '{}/annotations/captions_{}.json'.format(datadir, datatype)
coco_caps = COCO(annFile)
annIds = coco_caps.getAnnIds(imgIds=img['id'])
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)