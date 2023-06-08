# -*- coding: utf-8 -*-
"""
Created on Fri May 22 00:09:15 2020

@author: yuki
"""


# -*- coding: utf-8 -*-
"""
Created on Thu May 21 21:17:24 2020

@author: yuki
"""

#cvfave_1.1から改良

import cv2
import sys
import os.path
from time import sleep
import glob
import os





#オリジナル
def resizer1(filename):
    #global num
    image = cv2.imread(filename , cv2.IMREAD_COLOR)
    resize_image = cv2.resize(image,(128,128),interpolation = cv2.INTER_AREA)
    output_path = os.path.join(resize_output_dir, '{0}.jpg'.format(num))
    cv2.imwrite(output_path, resize_image)
    #num += 1
    


#ファイルネームをもとのに一致。マルチ用
def resizer(filename):
    print(filename)
    #global num
    image = cv2.imread(filename , cv2.IMREAD_COLOR)
    resize_image = cv2.resize(image,(128,128),interpolation = cv2.INTER_AREA)
    output_path = os.path.join(resize_output_dir, os.path.splitext(os.path.basename(filename))[0]+"_128.jpg")
    cv2.imwrite(output_path, resize_image)
    #num += 1





resize_output_dir = "./128resized_images"
files = glob.glob("./cut_images/*")



"""
for num, filename in enumerate(files):
    print(filename)
    resizer(filename)
    #sleep(1)
"""

"""
#普通の実行
num = 0
for filename in files:
    print(filename)
    resizer(filename)
    #sleep(1)
"""



"""
#速いが順番はぐちゃぐちゃ
numが同時に使用されてしまうので禁止

num = 0
from multiprocessing import Pool
if __name__ == '__main__':
    p = Pool()
    p.map(resizer, files)

"""

#最新版
from multiprocessing import Pool
if __name__ == '__main__':
    p = Pool()
    p.map(resizer, files)


