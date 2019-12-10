# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:30:09 2019
@author: Leo
"""
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D,Input
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential,Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

epochs = 50              # 迭代次数
input_shape = (28,28,1)  # 输如形状
nb_classes = 10

# 加载数据
(X_train,y_train),(X_test,y_test) = mnist.load_data()
print('X_train shape:',X_train.shape)
print('X_test shape:',X_test.shape)

# 数据预处理
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
print('X_train shape:',X_train.shape)
print('X_test shape:',X_test.shape)

# 转换成独热编码
Y_train = np_utils.to_categorical(y_train,nb_classes)
Y_test  = np_utils.to_categorical(y_test,nb_classes)
print('Y_train:',Y_train.shape)

# 传统卷积
def base_model():
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.2))
    # 参数量：3*3*32+32
    model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.25))
    # 参数量：32*64*3*3+64
    model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.25))
    # 参数量：64*64*3*3+64
    model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.25))
    # 参数量：64*128*3*3+128
    model.add(Flatten())
    model.add(Dense(128,activation='relu')) # 参数量：(128+1)*128
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(10,activation='softmax')) # 参数量：(128+1)*10
    model.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adam(),
                 metrics=['accuracy'])
    model.summary()
    return model

def eff_block(x_in, ch_in, ch_out):
    x = Conv2D(ch_in,kernel_size=(1, 1),activation='relu',padding='same',use_bias=False)(x_in)
    x = keras.layers.DepthwiseConv2D(kernel_size=(1, 3),activation='relu',padding='same',use_bias=False)(x)
    x = MaxPool2D(pool_size=(2, 1),strides=(2, 1))(x)
    x = keras.layers.DepthwiseConv2D(kernel_size=(3, 1),activation='relu',padding='same',use_bias=False)(x)
    x = Conv2D(ch_out,kernel_size=(2, 1),strides=(1, 2),activation='relu',padding='same',use_bias=False)(x)
    x = Dropout(0.25)(x)
    return x
def effnet():
    x_in = Input(shape=(28,28,1))
    x = eff_block(x_in, 16, 32)
    x = eff_block(x, 32, 64)
    x = eff_block(x, 32, 64)
    x = eff_block(x, 64, 128)
    #x = Conv2D(128,kernel_size=(3,3),activation='relu')(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=x_in, outputs=x)
    model.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adam(lr=1e-2),
                 metrics=['accuracy'])
    model.summary()
    return model

datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)
datagen.fit(X_train)

#model = base_model()
model = effnet()

#保存效果最好的模型
filepath = 'eff_model.hdf5'
checkpointer = ModelCheckpoint(filepath,monitor='val_acc',save_best_only=True,mode='max')
h = model.fit_generator(datagen.flow(X_train,Y_train,batch_size=1000),
                       steps_per_epoch=len(X_train)/1000,epochs=epochs,
                       validation_data=datagen.flow(X_test,Y_test,batch_size=len(X_test)),
                       validation_steps=1,callbacks=[checkpointer])
history = h.history
print(history.keys())

accuracy     = history['acc']
val_accuracy = history['val_acc']
epochs = range(len(accuracy))

plt.plot(epochs,accuracy,label='Training Accuracy')
plt.plot(epochs,val_accuracy,label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
#plt.show()
plt.savefig('eff_model.jpg',dpi=600)

score = model.evaluate(X_test,Y_test)
print("Test accuracy:",score[1])
