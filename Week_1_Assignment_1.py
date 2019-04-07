#!/usr/bin/env python
# coding: utf-8

import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


# # # Assignment_1 reimplement code

# In[180]:


img = cv2.imread('D:/demo.jpg')
img_gray = cv2.imread('D:/demo.jpg', 0)

print(img.shape)        # h, w, c
print(img_gray.shape)  # h, w


# In[ ]:


# image crop
h, w, _ = img.shape
_crop = img[: int(0.5 * h), int(0.2 * w): int(0.7 * w)]
cv2.imshow('_crop', _crop)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[ ]:


#gamma correction
def gamma_corr(img, gamma = 2):
    inv_gamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0)**inv_gamma)*255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(img, table)
cv2.imshow('img_brighter', gamma_corr(img, 2.0))
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[ ]:


#histogram
cv2.imshow('src', img)
plt.hist(img.flatten(), 256, [0,256])

b = cv2.equalizeHist(img[:, :, 0])
g = cv2.equalizeHist(img[:, :, 1])
r = cv2.equalizeHist(img[:, :, 2])
img = cv2.merge((b, g, r))
#plt.hist(img.flatten(), 256, [0,256])
cv2.imshow('hist_eq', img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[183]:


# scale+rotation+translation = similarity transform
def sim_trans(img):
    angle = random.randint(-180, 180)
    scale = random.randint(-2, 2)
    M = cv2.getRotationMatrix2D(random.random(1, 2), angle, scale)
    return cv2.warpAffine(img, M, (w, h))


# In[176]:


# Affine Transform
def aff_trans(img):
    h, w, _ = img.shape    #h = img.shape[0] #w = img.shape[1]
    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]]) #np.float32
    pts2 = np.float32([[w * 0.1, h * 0.2], [0.9 * (w - 1), 0.2 * h], [0.2 * w, 0.9 * (h - 1)]])
    
    M_aff = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M_aff, (w, h))


# In[181]:


# perspective transform
def persp_trans(img):
    h, w, _ = img.shape
    pts1 = np.float32([[0, 0], 
                       [w, 0], 
                       [0, h], 
                       [w, h]])
    pts2 = np.float32([[0.2 * w, 0.2 * h], 
                       [0.9 * w, 0.3 * h], 
                       [0,2 * w, 0.9 * h], 
                       [0.9 * w, 0.9 * h]])
    M_persp = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M_persp, (w, h))
