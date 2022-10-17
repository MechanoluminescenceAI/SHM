
import glob
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.models import load_model
import random
import pandas as pd
from pandas import DataFrame
from keras.layers import Dense
from keras.layers import Flatten,Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import os
import keras
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping


np.random.seed(0)



def set_image(image):
    image=image+'/images/*.png'
    gan_images=glob.glob(image)
    real_list=[]
    fake_list=[]   
    targets=[]
    outputs=[]
    for i in gan_images:
        if 'targets' in i:
            targets.append(i)
        elif 'outputs' in i:
            outputs.append(i)
    for fname in targets:
        img=Image.open(fname)
        x=np.asarray(img)
        real_list.append(x)
    for fname in outputs:
        img=Image.open(fname)
        x=np.asarray(img)
        fake_list.append(x)
    return np.asarray(real_list),np.asarray(fake_list)

def saveGraph(K,RealK,FakeK,file_name):
    fig=plt.gcf()
    plt.plot(K,'ro-',label='Real K')
    plt.plot(RealK,'bs-',label='Real Image Predict K')
    plt.plot(FakeK,'go-',label='Fake Image Predict K')
    fig.savefig(file_name+".png")
    plt.cla()

def saveExcel(K,RealK,FakeK,file_name):
    frame=DataFrame()
    start=[i+700 for i in range(len(K))]
    frame['frame']=start
    frame['Real K']=K
    frame['Real Image Predict K']=RealK
    frame['Fake Image Predict K']=FakeK
    frame.to_csv(file_name+'Excel.csv')

def create_model(img_shape):
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(11,11),padding='valid',activation='relu',
                         input_shape=img_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(8,kernel_size=(5,5),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16,kernel_size=(5,5),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(192,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(320,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(192,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model
    
def train(output_dir,train_img,train_K,K_type,train_rate=100):
    index=0
    if train_K=='ML':
        index=2
    else:
        index=5
    if K_type=='Kp':
        index=index
    elif K_type=='K1':
        index=index+1
    else:
        index=index+2
    if not os.path.exists(output_dir+'Learning_Graph'):
        os.makedirs(output_dir+'Learning_Graph')
    count=0
    real_img,fake_img=set_image(train_img)
    model=create_model(real_img[0].shape)
    K=pd.read_csv('K(0810).csv').values[1:,index]
    K=np.array(K,dtype='f')
    train_rate=int((train_rate*len(real_img))/100)
    hist=model.fit(
            real_img[0:train_rate],K[0:train_rate],
            epochs=1,
            verbose=2,
            validation_data=(fake_img[0:train_rate],K[0:train_rate]))
    Real_Image_K=model.predict(real_img)
    Fake_Image_K=model.predict(fake_img)
    saveGraph(K,Real_Image_K,Fake_Image_K,output_dir+'Learning_Graph/'+str(count))
    while count<300:
        count+=1
        hist=model.fit(
                real_img[:train_rate],K[:train_rate],
                epochs=1,
                verbose=2,
                validation_data=(fake_img[:train_rate],K[:train_rate]))
        print(train_K,"의 ",K_type,str(count)+"번째 학습 중입니다")
        Real_Image_K=model.predict(real_img)
        Fake_Image_K=model.predict(fake_img)
        saveGraph(K,Real_Image_K,Fake_Image_K,output_dir+'Learning_Graph/'+str(count))
    saveExcel(K,Real_Image_K,Fake_Image_K,output_dir)
    model.save(output_dir+'CNN.h5')

def test(output_dir,test_img,train_K,K_type,model_name):
    index=0
    if train_K=='ML':
        index=2
    else:
        index=5
    if K_type=='Kp':
        index=index
    elif K_type=='K1':
        index=index+1
    else:
        index=index+2
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    real_img,fake_img=set_image(test_img)
    K=pd.read_csv('K(0810).csv').values[1:,index]
    K=np.array(K,dtype='f')
    model=load_model(model_name)
    Real_Image_K=model.predict(real_img)
    Fake_Image_K=model.predict(fake_img)
    saveGraph(K,Real_Image_K,Fake_Image_K,output_dir+'Graph')
    saveExcel(K,Real_Image_K,Fake_Image_K,output_dir)

def img_location(gan_Learning,img_Type,gan_epochs):
    GAN=''
    LOCATION=''
    if gan_Learning==100:
        GAN='GAN_Image/GAN_Learning_100%/'
    else:
        GAN='GAN_Image/GAN_Learning_90%/'
        
    if img_Type=='FEM':
        LOCATION=GAN+'FEM/FEM('+str(gan_epochs)+')'
    else:
        LOCATION=GAN+'ML/ML('+str(gan_epochs)+')'
        
    return LOCATION
# Kp K1 J
e1=['ML','FEM']
k1=['Kp','K1']
val=[100,90]
val2=[100,90]
gan_epochs=[100,200,300,400,500]
for ex in e1:
    for Kt in k1:
        for cnn in val2:
                train(output_dir='train/'+ex+'('+str(cnn)+')/'+Kt+'/',train_img=img_location(cnn,ex,300),train_K=ex,K_type=Kt,train_rate=cnn)

for ex in e1: # ML FEM
    for v2 in val2: #CNN RATE
        for Kt in k1: #K TYPE
            for v1 in val: # GAN RATE
                for ge in gan_epochs: #GAN EPOCHS
                    print(ex,'CNN',v2,Kt,'GAN',v1,ge)
                    test('test/'+ex+'/CNN('+str(v2)+')/K('+Kt+')/GAN('+str(v1)+')('+str(ge)+')/',img_location(v1,ex,ge),ex,Kt,'train/'+ex+'('+str(v2)+')/'+Kt+'/CNN.h5')
