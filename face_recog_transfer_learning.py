#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator

from keras.applications import MobileNet

img_rows, img_cols = 224, 224 
MobileNet = MobileNet(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))

for layer in MobileNet.layers:
    layer.trainable = False


# In[2]:


def layer_add(bottom_model):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(1,activation='sigmoid')(top_model)
    return top_model


# In[4]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

FC_Head = layer_add(MobileNet)

model = Model(inputs = MobileNet.input, outputs = FC_Head)

print(model.summary())


# In[5]:


from keras.preprocessing.image import ImageDataGenerator

train_data_dir = '/home/invincible/Desktop/cnn_dataset/train'
validation_data_dir = '/home/invincible/Desktop/cnn_dataset/test'
# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
batch_size = 32
 


# In[6]:


train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,)
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,)


# In[15]:


from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

"""model.compile(loss = 'binary_crossentropy',
              optimizer = Adam(learning_rate= 0.0001),
              metrics = ['accuracy'])"""


# In[19]:


epochs = 5
batch_size = 32
nb_train_samples = 1498
nb_validation_samples = 206
history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)


# In[20]:


model.save('my_face_recog.hdf5')


# In[21]:


ar=model.predict(validation_generator)


# In[7]:


from keras.preprocessing import image
import os
from keras.models import load_model


# In[8]:


model=load_model('my_face_recog.hdf5')


# In[9]:


di='/home/invincible/Desktop/cnn_dataset/validation'


# In[10]:


a=os.listdir(di)


# In[11]:


test_image = image.load_img(os.path.join(di,a[0]), 
               target_size=(224,224))


# In[12]:


test_image


# In[13]:


test_image = image.img_to_array(test_image)


# In[14]:


test_image.shape


# In[15]:


import numpy as np 
test_image = np.expand_dims(test_image, axis=0)


# In[16]:


test_image.shape


# In[17]:


res=model.predict(test_image)


# In[25]:


res[0][0]


# In[ ]:





# In[ ]:




