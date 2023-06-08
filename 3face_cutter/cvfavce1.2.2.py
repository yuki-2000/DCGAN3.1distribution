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
#https://qiita.com/mczkzk/items/fda37ac4f9ddab2d7f45

import cv2
import sys
import os.path
from time import sleep
import glob
import os




import numpy as np
import cv2










def detect1(filename, cascade_file = "./lbpcascade_animeface.xml"):
    #print(filename)
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)
    
    #global num

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread( filename , cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    #元のminsizeは24ｘ24だが、128でやるなら50x50は最低必要
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (50, 50))
    
    
    

    #顔の切り取り
    for i, (x,y,w,h) in enumerate(faces):
        # 一人ずつ顔を切り抜く
        face_image = image[y:y+h, x:x+w]
        output_path = os.path.join(face_output_dir,  '{0}-{1}.jpg'.format(num,i))

        
        #print(output_path)
        #output_path = os.path.join(output_dir,'{0}.jpg'.format(i))
        cv2.imwrite(output_path,face_image)
        
   
    #赤枠で囲った画像の保存
    #上のfor内でやると、赤線がカットされた画像に映り込む    
    for i, (x,y,w,h) in enumerate(faces):        
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(paint_output_dir + str(num) + ".jpg", image)
    

    #num += 1


    
        
    
    """
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
        #cv2.imshow("AnimeFaceDetect", image)
    cv2.waitKey(0)
    #cv2.imwrite("out.png", image)
    cv2.imwrite('{0}.jpg'.format(num), image)
    """
    

"""
if len(sys.argv) != 2:
    sys.stderr.write("usage: detect.py <filename>\n")
    sys.exit(-1)
"""    









#https://qiita.com/SKYS/items/cbde3775e2143cad7455
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None





#日本語パスを対応させた
#出力ファイル名はもとのに基づいた
def detect(filename, cascade_file = "./lbpcascade_animeface.xml"):

    print(filename)
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)
    
    #global num

    cascade = cv2.CascadeClassifier(cascade_file)
    image = imread( filename , cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    #元のminsizeは24ｘ24だが、128でやるなら50x50は最低必要
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (50, 50))
    
    
    

    #顔の切り取り
    for i, (x,y,w,h) in enumerate(faces):
        # 一人ずつ顔を切り抜く
        face_image = image[y:y+h, x:x+w]
        #output_path = os.path.join(face_output_dir,  '{0}-{1}.jpg'.format(num,i))
        output_path = os.path.join(face_output_dir, os.path.splitext(os.path.basename(filename))[0]+'-{0}.jpg'.format(i))
        #print(output_path)
        #print(output_path)
        #output_path = os.path.join(output_dir,'{0}.jpg'.format(i))
        cv2.imwrite(output_path,face_image)
"""        
        
    #赤枠で囲った画像の保存
    #上のfor内でやると、赤線がカットされた画像に映り込む    
    for i, (x,y,w,h) in enumerate(faces):        
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(paint_output_dir + str(num) + ".jpg", image)
    

    #num += 1

"""   












face_output_dir = "./cut_images"
paint_output_dir = "./painted_images/"
files = glob.glob("./input_images/*")





"""
#ver1よう
for num, filename in enumerate(files):
    print(filename)
    detect(filename)
    #sleep(1)
"""

"""
for file in files:
    detect(file)

"""
#新しいほう
from multiprocessing import Pool
if __name__ == '__main__':
    p = Pool()
    p.map(detect, files)





"""

#普通の実行
num = 0
for filename in files:
    print(filename)
    detect(filename)
    #sleep(1)

"""




"""
#使用不可　numが同時に使用されてしまい、スレッド数で割った数しかできない。
#速いが順番はぐちゃぐちゃ

num = 0
from multiprocessing import Pool
if __name__ == '__main__':
    p = Pool()
    p.map(detect, files)
"""
