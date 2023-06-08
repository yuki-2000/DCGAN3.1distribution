# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 21:19:25 2021

@author: yuki
"""
#https://teratail.com/questions/165488
#ここからパクりました。
#画像がカラーか白黒か判定します。



import cv2
import numpy as np
import os

# モノクロ画像に変換
def to_gray(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.splitext(path)[0] + '_gray.jpg', gray)

# 彩度を変更
def conv_saturation(path,sat):
    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s[:,:] = sat
    hsv = cv2.merge((h,s,v));
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('{}_sat{:03d}.jpg'.format(os.path.splitext(path)[0],sat), img)

# 提示ソースの判定用の値
def get_ratio(path):
    img = cv2.imread(path)

    b, g, r = cv2.split(img)

    r_g = np.count_nonzero(abs(r - g))
    r_b = np.count_nonzero(abs(r - b))
    g_b = np.count_nonzero(abs(g - b))
    diff_sum = float(r_g + r_b + g_b)

    return diff_sum / img.size # > 0.005

# カラー画像か判定
# Detect if image is color, grayscale or black and white with Python/PIL
# https://stackoverflow.com/questions/20068945/detect-if-image-is-color-grayscale-or-black-and-white-with-python-pil
def detect_color_image(file, MAYBE_COLOR = 300):
    print(file)

    from PIL import Image, ImageStat
    from functools import reduce

    MONOCHROMATIC_MAX_VARIANCE = 0.005
    COLOR = 1000

    v = ImageStat.Stat(Image.open(file)).var # =分散
    is_monochromatic = reduce(lambda x, y: x and y < MONOCHROMATIC_MAX_VARIANCE, v, True)
    if is_monochromatic:
        return "Monochromatic image"
    else:
        mes = '?'
        if len(v)==3: # color
            maxmin = abs(max(v) - min(v))
            if maxmin > COLOR:
                mes = "Color"
                image_copy(file, Color_path)
            elif maxmin > MAYBE_COLOR:
                mes = "Maybe color"
                image_copy(file, Maybe_color_path)
            else:
                mes = "grayscale"
                image_copy(file, grayscale_path)

            return '{}({})'.format(mes, maxmin)

        elif len(v)==1:
            image_copy(file, Black_and_white_path)
            return "Black and white"

        else:
            image_copy(file, unknown_path)
            return "Don't know..."



def image_copy(file, dst):
    import shutil
    shutil.copy(file, dst)



"""
# テスト画像生成
to_gray('lena.jpg')
conv_saturation('lena.jpg',1)
conv_saturation('lena.jpg',10)
conv_saturation('lena.jpg',100)
"""



#画像の入力元、出力先のパス設定
from pathlib import Path
import glob
#opencvはasciiのみ
base_path = "./input_images"
#paths = Path(base_path).glob("*")
paths = glob.glob(base_path + "/*")


Color_path = "./color_sorted/color"
Maybe_color_path = "./color_sorted/maybe_color"
grayscale_path = "./color_sorted/gray"
Black_and_white_path = "./color_sorted/bw"
unknown_path = "./color_sorted/unknown"


def folder_maker(saving_direcory_path):
    if not os.path.exists(saving_direcory_path):
        os.mkdir(saving_direcory_path)


folder_maker(Color_path)
folder_maker(Maybe_color_path)
folder_maker(grayscale_path)
folder_maker(Black_and_white_path)
folder_maker(unknown_path)


"""
# 各画像を判定
for path in paths:
    print(path)
    print(get_ratio(path)) # 提示ソースの判定値
    print(detect_color_image(path))
    print()
"""    
    
    
from multiprocessing import Pool
if __name__ == '__main__':
    p = Pool()
    p.map(detect_color_image, paths)

"""
# 閾値を徐々にあげる
path = 'lena_sat010.jpg'
print(path)
for maybe_color in [100,200,300]:
    print(detect_color_image(path,maybe_color),maybe_color)
"""