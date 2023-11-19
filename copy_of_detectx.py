import os, sys
import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline
from keras import backend as K
from keras.layers import Dropout, Dense, BatchNormalization, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
from sklearn.metrics import precision_recall_fscore_support
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

train_data_dir = 'C:\Users\yunus\Downloads\sample'

!ls 'C:\Users\yunus\Downloads\sample'

"""### Visualize some images"""

import os
import random
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

image_per_class = 2
classes = os.listdir(train_data_dir)
subplot = len(classes)*100 + image_per_class*10 + 1
fig = plt.figure(figsize=(8,8))

for each_class in classes:
    files = os.listdir(os.path.join(train_data_dir, each_class))
    files_to_print = random.sample(files, image_per_class)

    for image_file in files_to_print:
        img = image.load_img(os.path.join(train_data_dir, each_class, image_file), target_size=(150,150))
        plt.subplot(subplot)
        plt.imshow(img)
        plt.title(each_class)
        subplot += 1

plt.show()

rescale = 1./255
target_size = (150, 150)
batch_size = 16
class_mode = "categorical"
#class_mode = "binary"

train_datagen = ImageDataGenerator(rescale=rescale,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data

# Checkpoint
model_dir = 'models'
model_file = model_dir + '/detectx-mobilenet-{epoch:02d}.hdf5'
checkpoint = ModelCheckpoint(
    model_file,
    monitor='acc',
    period=1)

callbacks = [checkpoint]

def get_mobilenet():
    base_mobilenet_model = MobileNet(
        input_shape=(150, 150, 3),
        include_top=False,
        weights=None
    )
    model = Sequential()
    model.add(base_mobilenet_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model

optimizer = Adam(lr=0.001)
# For more than two classes use categorical_crossentropy or sparse_categorical_crossentropy
loss = 'categorical_crossentropy'
metrics = ['accuracy']
steps_per_epoch = 19
validation_steps = 5
epochs = 50

model = get_mobilenet()
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)

model.summary()

epochs = 50
history = model.fit_generator(
    train_generator,
    steps_per_epoch = steps_per_epoch,
    validation_data = validation_generator,
    validation_steps = validation_steps,
    epochs=epochs,
    callbacks = callbacks,
  verbose=1)

"""Clearly the model at 40th epoch is performing the best in terms of validation loss and accuracy.<br>
Lets load this model...
"""

from keras.models import load_model

model_epoch = load_model('models/detectx-mobilenet-40.hdf5')
model_epoch.summary()

y_pred = model_epoch.predict_generator(validation_generator)
y_pred = y_pred.argmax(axis=-1)
print(y_pred)

"""Now, lets validate the model"""

y_true = validation_generator.classes
y_true = y_true.reshape((y_true.shape[0], 1))
print(classification_report(y_true, y_pred))

"""Due to sparsity of data, the performance of the model is bad. To improve this we are trying to gather more data."""