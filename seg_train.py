# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:56:16 2020
@author: Leo
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential, load_model
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical  
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, PReLU,Input,LeakyReLU
from keras.layers import UpSampling2D,BatchNormalization,Reshape,Permute,Activation  
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate,Add,Subtract,Average,UpSampling2D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def preprocess_img(img):
    img = image.img_to_array(img)
    img = 2/255.0 * img - 1
    return img # -1~1
def augmentation_img(img1,img2):
    aug = np.random.randint(2,size=5)
    img_a1, img_a2 = np.zeros_like(img1) , np.zeros_like(img2)
    if aug[0]:
        # 旋转
        img_a1 = image.random_rotation(img1,30,0,1,2)
        img_a2 = image.random_rotation(img2,30,0,1,2)
    if aug[1]:
        # 错位
        img_a1 = image.random_shear(img1,30,0,1,2)
        img_a2 = image.random_shear(img2,30,0,1,2)
    if aug[2]:
        # 缩放
        img_a1 = image.random_zoom(img_a1,(0.8,1.2),0,1,2) 
        img_a2 = image.random_zoom(img_a2,(0.8,1.2),0,1,2) 
    if aug[3]:
        # 亮度
        img_a1 = image.random_brightness(img_a1, (0.5,1.5))
    if aug[4]:
        # 噪声
        img_a1 += np.random.uniform(0,32,(np.shape(img_a1)))
        img_a1 = np.clip(img_a1,0,255)
    # 转成 0-255 整数
    img_a1 = np.array(img_a1,'uint8')
    img_a2 = np.array(img_a2,'uint8')
    return img_a1, img_a2

def seg_model(shape,n_label): 
    model = Sequential()  
    #encoder  
    model.add(Conv2D(32,(3,3),strides=(1,1),input_shape=shape,padding='same',activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    #(128,128)  
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(64,64)  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(32,32)  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(16,16)  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(8,8)  
    #decoder  
    model.add(UpSampling2D(size=(2,2)))  
    #(16,16)  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(32,32)  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(64,64)  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(128,128)  
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(256,256)  
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same'))  
    model.add(Reshape((shape[0]*shape[1],n_label)))

    model.add(Activation('softmax'))  
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])  
    model.summary() 
    return model
def conv2d_layers(x,n_kernal,pool=False):
    x = Conv2D(n_kernal, (3,3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    if pool:
        x = MaxPooling2D(pool_size=(2,2))(x)
    return x
def conv2d_decode(x1,x2,n_kernal):
    x = concatenate([x1,x2],axis=3)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(n_kernal, (3,3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    return x
    
def build_model(input_shape,n_label):
    input_img = Input(shape=input_shape)
    # encoder
    # 256
    block1 = conv2d_layers(input_img,32)
    block1 = conv2d_layers(block1,32,True)
    # 128
    block2 = conv2d_layers(block1,64)
    block2 = conv2d_layers(block2,64,True)
    # 64
    block3 = conv2d_layers(block2,128)
    block3 = conv2d_layers(block3,128)
    block3 = conv2d_layers(block3,128,True)
    # 32
    block4 = conv2d_layers(block3,256)
    block4 = conv2d_layers(block4,256)
    block4 = conv2d_layers(block4,256,True)
    # 16
    block5 = conv2d_layers(block4,256)
    block5 = conv2d_layers(block5,256)
    block5 = conv2d_layers(block5,256,True)
    # 8
    # decoder
    block6 = UpSampling2D(size=(2, 2))(block5)
    block6 = conv2d_layers(block6,256)
    block6 = conv2d_layers(block6,256)
    block6 = conv2d_layers(block6,256)
    # 16
    block7 = conv2d_decode(block6,block4,256)
    block7 = conv2d_layers(block7,256)
    block7 = conv2d_layers(block7,256)
    # 32
    block8 = conv2d_decode(block7,block3,128)
    block8 = conv2d_layers(block8,128)
    # 64
    block9 = conv2d_decode(block8,block2,64)
    block9 = conv2d_layers(block9,64)
    # 128
    block10 = conv2d_decode(block9,block1,32)
    block10 = conv2d_layers(block10,32)
    # 256
    conv = Conv2D(n_label,(1, 1),strides=(1, 1), padding='same')(block10)
    conv_reshape = Reshape((input_shape[0]*input_shape[1],n_label))(conv)
    outputs = Activation('softmax')(conv_reshape)

    model = Model(inputs=input_img, outputs=outputs)
    adam = Adam(lr=0.0002, epsilon=1e-08, decay=1e-5, amsgrad=True)
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy']) 
    model.summary()  
    return model

def batch_generater(data_path, data_list, batch_size, shape, n_label):
    offset = 0
    while True:
        train_list = data_list[0]
        test_list = data_list[1]
        X = np.zeros((batch_size, *shape))
        Y = np.zeros((batch_size, shape[0]*shape[1], n_label))
        for i in range(batch_size):
            img_x_path = data_path[0] + '/' + train_list[i + offset]
            img_y_path = data_path[1] + '/' + test_list[i + offset]
            
            img_x = image.load_img(img_x_path, target_size = shape[0:2])
            img_y = image.load_img(img_y_path, target_size = shape[0:2])
            img_x = image.img_to_array(img_x)
            img_y = image.img_to_array(img_y)
            img_y = img_y[...,0:1] # label图三个通道是一样的，只留一个
            #img_x, img_y= augmentation_img(img_x, img_y) # 扰动
            
            img_x = preprocess_img(img_x) # 归一化
            img_y = np.array(img_y).flatten() # 展平
            img_y = to_categorical(img_y, 5) # one-hot
            
            X[i,...] = img_x
            Y[i,...] = img_y
            if i+offset >= len(train_list)-1:
                data_list = shuffle(data_list)
                offset = 0
        yield (X, Y)
        offset += batch_size


if __name__ == '__main__':
    input_shape = (256, 256, 3)
    batch_size = 8
    epoch = 15
    n_label = 4 + 1
    model_savepath = './model/seg_model.h5'
    x_path = './data/remote_sensing_image/train/src' # 值域范围 0-255
    y_path = './data/remote_sensing_image/train/label' # 值域范围 0-255
    train_list = os.listdir(x_path)
    test_list  = os.listdir(y_path)
    X_train, X_test, y_train, y_test = train_test_split(train_list, test_list, test_size=0.2, random_state=42)
    
    # 搭建模型 + 加载权重
    model = build_model(input_shape,n_label)

    if os.listdir('./model/'):
        try:
            #model.load_weights(model_savepath)
            model = load_model(model_savepath)
            print('Model weights loaded!')
        except:
            print('Model weights load filed!')

    # 模型设置，设置ckpt，设置提前结束
    save_best = ModelCheckpoint(model_savepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    h = model.fit_generator(generator=batch_generater((x_path,y_path),(X_train,y_train),batch_size,input_shape,n_label), 
                            steps_per_epoch = len(X_train)//batch_size, 
                            epochs=epoch, verbose=1, callbacks=[save_best, early_stop], 
                            validation_steps = len(X_test)//32,
                            validation_data=batch_generater((x_path,y_path),(X_test,y_test),32,input_shape,n_label))
    
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('train_val_loss_.jpg',dpi=600)
    