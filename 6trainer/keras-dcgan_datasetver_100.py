
from keras.models import Sequential
from keras.layers import Dense, Conv2DTranspose, Dropout
from keras.layers import Reshape, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
#from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Flatten
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
import glob
import random
import argparse
#import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img,img_to_array,load_img
#from keras.applications.vgg16 import VGG16
#from keras.layers import Input
#from keras.layers.core import Dropout
#from keras.models import Model



#https://qiita.com/MuAuan/items/85db7176574bdf979061
#これがもと
#----.py --mode trainが最低
#----.py --mode train --batch_size 512　--iteration　100いてれの値はモデルの保存頻度


#filesの中身がpngになっていたので修正
#100step沖だったのを10stepおきに変更
#tensorflow1.xで動かせ
#googledriveで学習するためにデータセットを作ってそこから学習するように変更


#ノイズがなぜか2次元だったので100に変更
#savenum=0の時は厚みを読み込まないようにした



n_colors = 3



#オリジナル
def generator_model():
    model = Sequential()

    #noise dimention!!
    model.add(Dense(8*8*128, input_shape=(100,))) #1024,100 10
    #model.add(Activation('tanh'))

    #model.add(Dense(128 * 16 * 16)) #128
    model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Reshape((8, 8, 128)))
    model.add(Conv2DTranspose(128, (5, 5), activation='tanh', strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, (5, 5), activation='tanh', strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(32, (5, 5), activation='tanh', strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(n_colors,(5, 5), activation='tanh', strides=2, padding='same'))
    #model.add(BatchNormalization())

    return model






#バッチノーマライゼーションを活性化関数の前に挿入
#ノイズから顔に変わるのが速くなった！
def generator_model2():
    model = Sequential()

    #noise dimention!!
    model.add(Dense(8*8*128, input_shape=(100,))) #1024,100 10
    #model.add(Activation('tanh'))

    #model.add(Dense(128 * 16 * 16)) #128
    #model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Reshape((8, 8, 128)))
    model.add(Conv2DTranspose(128, (5, 5),  strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2DTranspose(64, (5, 5),  strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2DTranspose(32, (5, 5),  strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2DTranspose(n_colors,(5, 5), activation='tanh', strides=2, padding='same'))
    #model.add(BatchNormalization())

    return model



#2からさらに最後もバッチを挟んだ
#カラフルビビッドになった。いいのかわからない
#http://elix-tech.github.io/ja/2017/02/06/gan.html
#によると良くないらしい
def generator_model3():
    model = Sequential()

    #noise dimention!!
    model.add(Dense(8*8*128, input_shape=(100,))) #1024,100 10
    #model.add(Activation('tanh'))

    #model.add(Dense(128 * 16 * 16)) #128
    model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Reshape((8, 8, 128)))
    model.add(Conv2DTranspose(128, (5, 5),  strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2DTranspose(64, (5, 5),  strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2DTranspose(32, (5, 5),  strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2DTranspose(n_colors,(5, 5),  strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))

    return model





#ドロップアウト追加

def generator_model4():
    model = Sequential()

    #noise dimention!!
    model.add(Dense(8*8*128, input_shape=(100,))) #1024,100 10
    #model.add(Activation('tanh'))

    #model.add(Dense(128 * 16 * 16)) #128
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #model.add(Dropout(0.5))

    model.add(Reshape((8, 8, 128)))
    model.add(Conv2DTranspose(128, (5, 5),  strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    
    model.add(Conv2DTranspose(64, (5, 5),  strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    
    model.add(Conv2DTranspose(32, (5, 5),  strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #model.add(Dropout(0.5))
    
    model.add(Conv2DTranspose(n_colors,(5, 5), activation='tanh', strides=2, padding='same'))
    #model.add(BatchNormalization())

    return model





#オリジナル
def discriminator_model():
    model = Sequential()
    
    model.add(Conv2D(16, (5, 5), input_shape=(128, 128, n_colors), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
        
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model



#バッチノーマライゼーション追加メモリが足りなくなる
#画像が全く生成されない
#realとfakeをノーマライズは良くないみたいだ
def discriminator_model2():
    model = Sequential()
    
    model.add(Conv2D(16, (5, 5), input_shape=(128, 128, n_colors), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
        
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model




#最後のflattenをglobal_average_poolingにしてみた
#計算量が減るらしいパラメータが2/3になった
#画像は甲乙つけがたいからいいのかな
#https://qiita.com/mine820/items/1e49bca6d215ce88594a
#https://www.machinecurve.com/index.php/2020/01/30/what-are-max-pooling-average-pooling-global-max-pooling-and-global-average-pooling/
#https://qiita.com/Phoeboooo/items/f188eb2426afc8757272
def discriminator_model3():
    model = Sequential()
    
    model.add(Conv2D(16, (5, 5), input_shape=(128, 128, n_colors), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
        
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128))
    model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


#3をもとにmaxpoolingからaveに変えてみた
#ぼやっとした画像になってしまった却下
def discriminator_model4():
    model = Sequential()
    
    model.add(Conv2D(16, (5, 5), input_shape=(128, 128, n_colors), padding='same'))
    model.add(Activation('tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
        
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128))
    model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model




#3をもとにglobal_average_poolingでなくGlobalMaxPooling2D
#悪くないが、averageのほうがいいかも
def discriminator_model5():
    model = Sequential()
    
    model.add(Conv2D(16, (5, 5), input_shape=(128, 128, n_colors), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
        
    model.add(GlobalMaxPooling2D())
    model.add(Dense(128))
    model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model



#3をもとに層を2倍に
#バッチサイズは半分に
#遅いうえに目が現れない
def discriminator_model6():
    model = Sequential()
    
    model.add(Conv2D(16, (5, 5), input_shape=(128, 128, n_colors), padding='same'))
    model.add(Activation('tanh'))
    model.add(Conv2D(16, (5, 5), input_shape=(128, 128, n_colors), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
        
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128))
    model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model






def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    #discriminator.trainable = False
    model.add(discriminator)
    return model

def image_batch(batch_size):
    files = glob.glob("./in_images/**/*.jpg", recursive=True)
    files = random.sample(files, batch_size)
    # print(files)
    res = []
    for path in files:
        img = Image.open(path)
        img = img.resize((128, 128))  #(64, 64)
        arr = np.array(img)
        arr = (arr - 127.5) / 127.5
        arr.resize((128, 128, n_colors)) #(64, 64)
        res.append(arr)
    return np.array(res)






def combine_images(generated_images, cols=5, rows=5):
    shape = generated_images.shape
    h = shape[1]
    w = shape[2]
    image = np.zeros((rows * h,  cols * w, n_colors))
    for index, img in enumerate(generated_images):
        if index >= cols * rows:
            break
        i = index // cols
        j = index % cols
        image[i*h:(i+1)*h, j*w:(j+1)*w, :] = img[:, :, :]
    image = image * 127.5 + 127.5
    image = Image.fromarray(image.astype(np.uint8))
    return image
    
#自分で作りました    
def combine_and_make_gif(generated_images, filename, dir_name="./gen_unique", duration=1000/15):
    from  time import sleep
    from PIL import Image
    import os, glob
    import sys
    import cv2
    shape = generated_images.shape
    h = shape[1]
    w = shape[2]
    ims = []
    for image in generated_images:
        image = image * 127.5 + 127.5
        image = Image.fromarray(image.astype(np.uint8))
        ims.append(image)
        
    
    
    #10, 15, 20がいい
    duration = 1000/15



    #保存した画像をもとにgifを作成
    #frames = glob.glob(f'{dir_name}/*.png')
    #frames.sort(key=os.path.getmtime, reverse=False)
    #ims = []
    #for frame in frames:

    #ims.append(Image.open(frame))



    ims[0].save(f'{dir_name}/{filename}.gif', save_all=True, append_images=ims[1:],
    optimize=False, duration=duration, loop=0)
    print(f'{dir_name}/{filename}.gif') 
     
 
#自分で作りました    
def make_gif(img100, filename, dir_name="./gen_unique", duration=1000/15):
    from  time import sleep
    from PIL import Image
    import os, glob
    import sys
    import cv2
    import numpy
    #shape = img100.shape
    #h = shape[1]
    #w = shape[2]

    #10, 15, 20がいい
    duration = 1000/15

    img100[0].save(f'{dir_name}/{filename}.gif', save_all=True, append_images=img100[1:],
    optimize=False, duration=duration, loop=0)
    print(f'{dir_name}/{filename}.gif')  
    
    
    
    #動画の作成
    #https://daeudaeu.com/pil_cv2_tkinter/#PIL_OpenCV2
    #pil>cv2に変換しないと
    
    #パラメーター
    fps = 1000 / duration
    height = 1280
    width = 1280
    
    
    
    # encoder(for mp4)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # output file name, encoder, fps, size(fit to image size)
    video = cv2.VideoWriter(f'{dir_name}/{filename}.mp4',fourcc, fps, (width, height))
    
    for img in img100:
        pil_image_array = numpy.array(img)
        # RGB -> BGR によりCV2画像オブジェクトに変換
        cv2_image = cv2.cvtColor(pil_image_array, cv2.COLOR_RGB2BGR)
        video.write(cv2_image)
    
    video.release()
    print('written')
 
 
 
 
 
    
    


def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

#オリジナル
def main0(BATCH_SIZE=55, ite=1000):
    batch_size = BATCH_SIZE
    ite=ite
    discriminator = discriminator_model3()
    generator = generator_model2()

    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    set_trainable(discriminator, False)
    opt =Adam(lr=0.0001, beta_1=0.5) 
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=opt)
    
    print('generator.summary()---')
    generator.summary()
    print('discriminator_on_generator.summary()---')
    discriminator_on_generator.summary()

    set_trainable(discriminator, True)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt)
    print('discriminator.summary()---')
    discriminator.summary()
    #generator.load_weights('./gen_images/generator_11000.h5', by_name=True)
    #discriminator.load_weights('./gen_images/discriminator_11000.h5', by_name=True)



#https://qiita.com/skyfish20ch/items/ef8b7e0db4a6903c730b
    f=np.load("./gan.npz")
    X_train=f["x_train"]
    # Rescale -1 to 1
    print(X_train.shape)

    X_train = X_train / 127.5 - 1.



    
    for i in range(0 * 1000,31 * 1000):
        print(i)
        
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        batch_images = X_train[idx]
        #noise dimention!!
        noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0) #32*32 10
        generated_images = generator.predict(noise)
        X = np.concatenate((batch_images, generated_images))
        y = [1] * batch_size + [0] * batch_size
        d_loss = discriminator.train_on_batch(X, y)
        #noise dimention!!
        noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0) ##32*32
        g_loss = discriminator_on_generator.train_on_batch(noise, [1] * batch_size)
        
        #ここの数値で画像生成頻度が変わるね
        # make 5x5image 
        if i % 10 == 0:
            print("step %d d_loss, g_loss : %g %g" % (i, d_loss, g_loss))
            image = combine_images(generated_images)
            #os.system('mkdir -p ./gen_images')
            os.makedirs(os.path.join(".", "gen_images"), exist_ok=True)
            image.save("./gen_images/gen%05d.png" % i)
            #save model
            if i % ite == 0:
                generator.save_weights('./gen_images/generator_%d.h5' % i, True)
                discriminator.save_weights('./gen_images/discriminator_%d.h5' % i, True)

#saveした重みをロードできるようにした
def main(BATCH_SIZE=55, ite=1000, savenum=0):
    batch_size = BATCH_SIZE
    ite=ite
    savenum = savenum
    discriminator = discriminator_model3()
    generator = generator_model2()

    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    set_trainable(discriminator, False)
    opt =Adam(lr=0.0001, beta_1=0.5) 
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=opt)
    
    print('generator.summary()---')
    generator.summary()
    print('discriminator_on_generator.summary()---')
    discriminator_on_generator.summary()

    set_trainable(discriminator, True)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt)
    print('discriminator.summary()---')
    discriminator.summary()
    if savenum != 0:
        generator.load_weights('./gen_images/generator_%d.h5' % savenum, by_name=True)
        discriminator.load_weights('./gen_images/discriminator_%d.h5' % savenum, by_name=True)






    
    for i in range(savenum,31 * 10000):
        print(i)
        batch_images = image_batch(batch_size)

        #noise dimention!!
        noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0) #32*32 10
        generated_images = generator.predict(noise)
        X = np.concatenate((batch_images, generated_images))
        y = [1] * batch_size + [0] * batch_size
        d_loss = discriminator.train_on_batch(X, y)
        #noise dimention!!
        noise = np.random.uniform(size=[batch_size, 100], low=-1.0, high=1.0) ##32*32
        g_loss = discriminator_on_generator.train_on_batch(noise, [1] * batch_size)
        
        #ここの数値で画像生成頻度が変わるね
        #if i % 100 == 0:
        if i % (ite//10) == 0:
            print("step %d d_loss, g_loss : %g %g" % (i, d_loss, g_loss))
            image = combine_images(generated_images)
            #os.system('mkdir -p ./gen_images')
            os.makedirs(os.path.join(".", "gen_images"), exist_ok=True)
            image.save("./gen_images/gen%05d.png" % i)
            if i % ite == 0:
                generator.save_weights('./gen_images/generator_%d.h5' % i, True)
                discriminator.save_weights('./gen_images/discriminator_%d.h5' % i, True)



            
def generate0(BATCH_SIZE=55, ite=10000, nice=False):
    g = generator_model2()
    g.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    g.load_weights('./gen_images/generator_%d.h5'%ite)
    if nice:
        d = discriminator_model3()
        d.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5)) #optimizer="SGD"
        d.load_weights('./gen_images/discriminator_%d.h5'%ite)
        #noise dimention!!
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100)) ##32*32
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        for i in range(10):
            #noise dimention!!
            noise = np.random.uniform(size=[BATCH_SIZE, 100], low=-1.0, high=1.0) ##32*32 10x10
            
            #print('noise[0]',noise[0])
            
            #おそらく特徴量の可視化だな
            #for j in range(25):
                #noise[j]=np.array((-1+j/25, -1+i/10))  #(-1+i/10, -1+j/25)
            for j in range(25):
                noise[j][i]=-1+j/12.5
            
            print('noise[0]',noise[0])
            ##noise dimention!!
            plt.imshow(noise[0].reshape(10,10)) #32,32　100,1
            plt.pause(0.01)
            generated_images = g.predict(noise)
            #600x480までしか表示できないらしい
            plt.imshow(generated_images[0])
            plt.pause(0.01)
            #ここ２行最初コメントアウト
            ##noise dimention!!10x10=100
            image_noise = combine_images(noise.reshape(BATCH_SIZE,10,10,1))
            image_noise.save("./gen_images/generate_noise_%05d_%d.png" % (ite,i))
            image = combine_images(generated_images)
            
            image.save("./gen_images/generate4_%05d_%d.png" % (ite,i))
            image.resize((400,400))
            plt.imshow(image)
            print(i)
    #os.system('mkdir -p ./gen_images')
    os.makedirs(os.path.join(".", "gen_images"), exist_ok=True)
    image.save("./gen_images/generate%05d.png" % ite)


#gifでそれぞれの特徴量を変化させたgif100個作る
def generate1(BATCH_SIZE=55, ite=10000, nice=False):
    #ここはドロップアウトなしのモデルを使った方がいいかな
    g = generator_model2()
    g.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    g.load_weights('./gen_images/generator_%d.h5'%ite)
    if nice:
        d = discriminator_model3()
        d.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5)) #optimizer="SGD"
        d.load_weights('./gen_images/discriminator_%d.h5'%ite)
        #noise dimention!!
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100)) ##32*32
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        for i in range(100):
            #noise dimention!!
            noise = np.random.uniform(size=[BATCH_SIZE, 100], low=-1.0, high=1.0) ##32*32 10x10
            
            
            
            
            #全て同じノイズを使用する
            for line in range(BATCH_SIZE):
                noise[line] = noise[0]
                
            
            #print('noise[0]',noise[0])
            
            #おそらく特徴量の可視化だな
            #for j in range(25):
                #noise[j]=np.array((-1+j/25, -1+i/10))  #(-1+i/10, -1+j/25)
            for j in range(BATCH_SIZE):
                #noise[j][i]=-1+j*2/BATCH_SIZE
                noise[j][i]=-100+(j*200)/BATCH_SIZE
                
            
            #print('noise[0]',noise[0])
            ##noise dimention!!
            plt.imshow(noise[0].reshape(10,10)) #32,32　100,1
            plt.pause(0.01)
            generated_images = g.predict(noise)
            #600x480までしか表示できないらしい

            plt.imshow(generated_images[0])
            print("generateの0番目")
            plt.pause(0.01)
            #ここ２行最初コメントアウト
            ##noise dimention!!10x10=100
            #image_noise = combine_images(noise.reshape(BATCH_SIZE,10,10,1))
            #image_noise.save("./gen_unique/generate_noise_%05d_%d.png" % (ite,i))
            filename = ("generate4_%05d_%d" % (ite,i))
            
            combine_and_make_gif(generated_images, filename, dir_name="./gen_unique", duration=1000/BATCH_SIZE)
            
            image = combine_images(generated_images, cols=5, rows=5)
            #image.save("./gen_unique/generate4_%05d_%d.png" % (ite,i))
            plt.imshow(image)
            print(i)
    #os.system('mkdir -p ./gen_images')
    os.makedirs(os.path.join(".", "gen_images"), exist_ok=True)
    image.save("./gen_images/generate%05d.png" % ite)


#100x100のgifを作る
def generate(BATCH_SIZE=55, ite=10000, nice=False):
    z=100
    
    g = generator_model2()
    g.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    g.load_weights('./gen_images/generator_%d.h5'%ite)
    if nice:
        d = discriminator_model3()
        d.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5)) #optimizer="SGD"
        d.load_weights('./gen_images/discriminator_%d.h5'%ite)
        #noise dimention!!
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100)) ##32*32
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        img100=[]
        noise = np.random.uniform(size=[z, z], low=-1.0, high=1.0) ##32*32 10x10
        #全て同じノイズを使用する
        for line in range(z):
            noise[line] = noise[0]
            
        #このjは何段階で作るか？ バッチサイズの意味変わっちゃうよね   
        for j in range(BATCH_SIZE):
            
        
            #noise dimention!!

            
            #print('noise[0]',noise[0])
            
            #おそらく特徴量の可視化だな
            #for j in range(25):
                #noise[j]=np.array((-1+j/25, -1+i/10))  #(-1+i/10, -1+j/25)
            #このi:100は特徴量の数
            for i in range(z):
                #noise[j][i]=-1+j*2/BATCH_SIZE
                noise[i][i]=-100+(j*200)/BATCH_SIZE

                
            
            #print('noise[0]',noise[0])
            ##noise dimention!!
            #plt.imshow(noise[0].reshape(10,10)) #32,32　100,1
            #plt.pause(0.01)
            generated_images = g.predict(noise)
            #600x480までしか表示できないらしい

            #plt.imshow(generated_images[0])
            #print("generateの0番目")
            #plt.pause(0.01)
            #ここ２行最初コメントアウト
            ##noise dimention!!10x10=100
            #image_noise = combine_images(noise.reshape(BATCH_SIZE,10,10,1))
            #image_noise.save("./gen_unique/generate_noise_%05d_%d.png" % (ite,i))

            
            image = combine_images(generated_images, cols=10, rows=10)
            #print("この画像のタイプは")
            #print(type(image))


            img100.append(image)
            #image.save("./gen_unique/gentest_%05d_%d.png" % (ite,j))
            #plt.imshow(image)
            print(j)
            
    
        filename = ("generate100_%05d" % ite)
        make_gif(img100, filename, dir_name="./gen_unique", duration=100)
        
    #os.system('mkdir -p ./gen_images')
    #os.makedirs(os.path.join(".", "gen_images"), exist_ok=True)
    #image.save("./gen_images/generate%05d.png" % ite)





#original    
def get_args0():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--iteration", type=int, default=100)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args
    
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--iteration", type=int, default=100)
    parser.add_argument("--savenum", type=int, default=0)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

#original
"""
if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        main(BATCH_SIZE=args.batch_size,ite=args.iteration)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size,ite=args.iteration, nice=args.nice)
"""

#savenumを追加
if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        main(BATCH_SIZE=args.batch_size,ite=args.iteration, savenum=args.savenum)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size,ite=args.iteration, nice=args.nice)



