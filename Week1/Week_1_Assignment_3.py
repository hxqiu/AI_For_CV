#!/usr/bin/env python
# coding: utf-8
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('D:/demo.jpg')

#random crop
# img: source img
# x: width
# y: height
def random_crop(img, x, y):
    rand_x = random.randint(0, img.shape[1] - x)
    rand_y = random.randint(0, img.shape[0] - y)
    return img[rand_x: rand_x + x, rand_y : rand_y + y]

#random color shift
# img: source img
def random_color(img):
    b, g, r = cv2.split(img)
    rand_r = random.randint(-255, 255)
    rand_g = random.randint(-255, 255)
    rand_b = random.randint(-255, 255)
    rand_r = 0 if rand_r < 0 else rand_r
    rand_g = 0 if rand_g < 0 else rand_g
    rand_b = 0 if rand_b < 0 else rand_b

    b = b + rand_b
    g = g + rand_g
    r = b + rand_r
    return cv2.merge([b, g, r])

def random_pers_trans(img, margin = 0.2):
    w, h, c = img.shape
    x1 = random.randint(-w * margin, w * margin) #lb
    y1 = random.randint(-h * margin, h * margin)
    #rb
    x2 = random.randint(w * (1 - margin), w * (1 + margin))
    y2 = random.randint(-h * margin, h * margin)
    #rt
    x3 = random.randint(w * (1 - margin), w * (1 + margin))
    y3 = random.randint(h * (1 - margin), h * (1 + margin))
    #lt
    x4 = random.randint(-w * margin, w * margin)
    y4 = random.randint(h * (1 - margin), h * (1 + margin))
    
    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    M_pers = cv2.getPerspectiveTransform(pts1, pts2)
    img_pers = cv2.warpPerspective(img, M_aff, (w, h))
    return M_pers, img_pers

def random_rotation(img):
    rand_r = random.randint(-180, 180)
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rand_r, 1)
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img_rotate

def img_augment(img, x = 100, y = 100, persp_margin = 0.2):
    tmp_img = random_crop(img, x, y)
    tmp_img = random_color(tmp_img)
    tmp_img = random_rotation(tmp_img)
    _, tmp_img = random_pers_trans(tmp_img)
    return tmp_img
