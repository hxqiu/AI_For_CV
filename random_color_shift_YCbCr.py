#!/usr/bin/env python
# coding: utf-8

def random_color_YCrCb(img):
    ycb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    y = ycb[:,:,0]
    cr = ycb[:,:,1]
    cb = ycb[:,:,2]
 
    rand_y = random.randint(-255, 255)
    rand_cr = random.randint(-255, 255)
    rand_cb = random.randint(-255, 255)
    rand_y = 0 if rand_y < 0 else rand_y
    rand_cr = 0 if rand_cr < 0 else rand_cr
    rand_cb = 0 if rand_cb < 0 else rand_cb

    y = y + rand_y
    cr = cr + rand_cr
    cb = cb + rand_cb
    ret = np.dstack((y, cb, cr))
    return ret