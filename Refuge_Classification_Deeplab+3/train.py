
# coding: utf-8

# In[1]:


# USAGE
# python train_network.py --dataset images --model santa_not_santa.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import Deeplabv3
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau


# In[2]:


modelName='MICCAI2018_classi_DeepLabv3_21_201808261436.hdf5'

dataset='data/train/'
# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 50
INIT_LR = 1e-4
BS = 10

image_width=256
image_high=256


# In[3]:


# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (image_width, image_high))
	image = img_to_array(image)
	image[:,:,0] -= 107.546
	image[:,:,1] -= 60.8877
	image[:,:,2] -= 29.6568
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[5]
	cla = 0 if label[0:1] == "n" else 1
	#print(label+':'+str(cla))
	labels.append(cla)


# In[4]:


# scale the raw pixel intensities to the range [0, 1]

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.30, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


# In[5]:


# initialize the model
print("[INFO] compiling model...")
model =  Deeplabv3(input_shape=(image_width,image_high,3), classes=2)


# In[6]:



opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
model_checkpoint = ModelCheckpoint(modelName, monitor='val_acc',verbose=1, save_best_only=True)
reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,
                                       verbose=1, epsilon=1e-4, mode='min')
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1,callbacks=[model_checkpoint,reduce_lr_loss])

# save the model to disk
#print("[INFO] serializing network...")
#model.save(modelName)


# In[7]:



# In[8]:


#plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Glaucoma/Not Glaucoma")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')
 

