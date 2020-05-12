#!/usr/bin/env python
# coding: utf-8

# In[4]:


from keras.preprocessing .image import ImageDataGenerator
from keras.models.models import Sequential
from keras.layers import Conv2D,Maxpooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras import backend as k
import numpy as np
from keras.preprocessing import image 
img_width,img_height=150*150
train_data_dir = "data/train"
validation_data_dir = "data/validation"
nb_train_samples = 1000
nb_validation_samples = 100
epochs = 50
batch_size = 20
if k.image_data_format()=='channels_first':
    input_shape = (3,img_width,img_height)
    else:
        input_shape = (img_width,img_height,3)
        train_datagen = ImageDataGenerator(
        rescale = 1. /255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)
        test_datagen = ImageDataGenetor(rescale = 1. /255)
        
        train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img-_widh img_height),
        batch_size=batch_size,
        class_mode='binary')
        
        
        model=Sequential()
        model.add(Conv2D(32,(3,3), input_shape = input_shape))
        model.add(Activation('mini'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.summary()
        model.add(Conv2D(32(3,3)))
        model.add(Activation('mini'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Conv2D(64,(3,3)))
        model.add(Activation('mini'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('mini'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        model.summary()
        
        model.compile(Loss='binary_crossentropy',
                     optimizer = 'rmsprop',
                     metrices = ['accuracy'])
        
        #this is the augmentation cpnfiguration we will use for training
        
        model.fit_generator()
        train_generator,
        steps_per_epoch=nb_train_samples  // batch size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples  // batch_size)
        
        model.save_weights('first_try.h5')
        
        img_pred = image.load_img('data/validation/Eastern_Towhee/Eastern_Towhee_0079_22690.jpg', target_size= (150,150))
        img_pred = image.img_to_array(img_pred)
        img_pred = np.expand_dims(img_pred, axis = 0)
        
        
        
        rslt = model.predict(img_pred)
        print(rslt)
        if rslt[0][0] == 1:
            prediction = "Eastern_Towhee"
        else:
            prediction = "Someother bird"
            
        print(prediction)


# In[5]:


from keras.preprocessing .image import ImageDataGenerator
from keras.models.models import Sequential
from keras.layers import Conv2D,Maxpooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras import backend as k
import numpy as np
from keras.preprocessing import image 
img_width,img_height=150*150
train_data_dir = "data/train"
validation_data_dir = "data/validation"
nb_train_samples = 1000
nb_validation_samples = 100
epochs = 50
batch_size = 20
if k.image_data_format()=='channels_first':
    input_shape = (3,img_width,img_height)
    else:
        input_shape = (img_width,img_height,3)
        train_datagen = ImageDataGenerator(
        rescale = 1. /255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)
        test_datagen = ImageDataGenetor(rescale = 1. /255)
        
        train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img-_widh img_height),
        batch_size=batch_size,
        class_mode='binary')
        
        
        model=Sequential()
        model.add(Conv2D(32,(3,3), input_shape = input_shape))
        model.add(Activation('mini'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.summary()
        model.add(Conv2D(32(3,3)))
        model.add(Activation('mini'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Conv2D(64,(3,3)))
        model.add(Activation('mini'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('mini'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        model.summary()
        
        model.compile(Loss='binary_crossentropy',
                     optimizer = 'rmsprop',
                     metrices = ['accuracy'])
        
        #this is the augmentation cpnfiguration we will use for training
        
        model.fit_generator()
        train_generator,
        steps_per_epoch=nb_train_samples  // batch size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples  // batch_size)
        
        model.save_weights('first_try.h5')
        
        img_pred = image.load_img('data/validation/Eastern_Towhee/Eastern_Towhee_0079_22690.jpg', target_size= (150,150))
        img_pred = image.img_to_array(img_pred)
        img_pred = np.expand_dims(img_pred, axis = 0)
        
        
        
        rslt = model.predict(img_pred)
        print(rslt)
        if rslt[0][0] == 1:
            prediction = "Eastern_Towhee"
        else:
            prediction = "Someother bird"
            
        print(prediction)


# In[ ]:




