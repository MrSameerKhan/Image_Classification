# -*- coding: utf-8 -*-
import os, cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import np_utils
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, adam, RMSprop
from keras.models import Sequential

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras import backend as K
K.set_image_dim_ordering('th') 

img_rows=128
img_cols=128
num_channel=1
num_epoch=1

# will choose path from the current working directory
data_path= "/home/sameer/Desktop/Classification/train/training_data"
data_dir = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir:
     img_list=os.listdir(data_path+'/'+dataset)
     print("Loaded the images of dataset "+"{}\n".format(dataset))
     for img in img_list:
         input_img =cv2.imread(data_path+'/'+dataset+'/'+img)
         input_img =cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
         input_img_resize= cv2.resize(input_img, (128,128))
         img_data_list.append(input_img_resize)

#print("Image data list", img_data_list)
img_data = np.array(img_data_list)
#print("Array of image data",img_data)
img_data=img_data.astype("float32")
#print("Float of image arrat", img_data)
img_data= img_data/255
#print("Normalize of image data", img_data)

print("Shape of Image data", img_data.shape)

if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)		
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)

num_classes = 4

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:202]=0
labels[202:404]=1
labels[404:606]=2
labels[606:]=3
	  
names = ['cats','dogs']

Y= np_utils.to_categorical(labels, num_classes)

x,y=shuffle(img_data, Y, random_state=2)


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=2)

img1=img_data.shape
print("Image shape", img1)
num_of_samples2 = img_data.shape[0]
print("Number of samples", num_of_samples2)
input_shape= img_data[0].shape
print("Image row , column",input_shape)



model = Sequential()


model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))
model.add(Activation('relu'))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])

hist = model.fit(X_train, Y_train, batch_size=16, nb_epoch=1, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

# print('Test Loss:', score[0])
# print('Test accuracy:', score[1])

test_image = X_test[0:1]
print (test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(Y_test[0:1])

test_image = cv2.imread('/test/cat.1000.jpg')
test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
   
if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=3) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
else:
	if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
# Predicting the test image
print((model.predict(test_image)))
print(model.predict_classes(test_image))
