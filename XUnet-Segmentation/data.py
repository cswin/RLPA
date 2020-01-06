from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# from resizeimage import resizeimage
#import cv2
#from libtiff import TIFF
 

class dataProcess(object):

	def __init__(self, out_rows, out_cols, data_path = "../data/train/train_segmentation_800_cropped_aug/",
              label_path = "../data/train/Masks_800_cropped_aug/",
              test_path = "../data/train/trainImage_save_path_800/", npy_path = "../data/npydata",
              img_type = "jpg", label_type = "bmp"):

		"""

		"""
         
		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
        
		self.test_path = test_path
		self.npy_path = npy_path
		self.label_type = label_type
		self.temppath ='temp'

	def create_train_data(self):
		i = 0
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		imgs = glob.glob(self.data_path+"/*."+self.img_type)
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		imgs=sorted(imgs)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			print(imgname)
			lbl_name = imgname[imgname.rindex("/")+1:imgname.rindex(".")]+'.'+self.label_type
#			img = load_img(self.data_path + "/" + midname,grayscale = False,target_size=(self.out_rows,self.out_cols))
            
#			plt.imshow(img)
#			plt.title('img_orignal')
#			plt.show() 
			#label_ori = load_img(self.label_path + "/" + lbl_name,grayscale = True,target_size=(self.out_rows,self.out_cols))
			#plt.imshow(label_ori)
			#plt.title('label_orignal')
			#plt.show() 
            #img = img.thumbnail((1022,712), Image.ANTIALIAS) 
			#label = label.thumbnail((1022,712), Image.ANTIALIAS) 
			#img = img_to_array(img) 
			#label = img_to_array(label) 

			img = cv2.imread(self.data_path + "/" + midname)
			img = cv2.resize(img, (self.out_rows,self.out_cols))
			img = img_to_array(img)
#			img[:,:,0] -= 107.546
#			img[:,:,1] -= 60.8877
#			img[:,:,2] -= 29.6568
#			img_temp = array_to_img(img)
#			img_temp.save(self.temppath + "/" + midname)

#			plt.imshow(img)
#			plt.title('img')
#			plt.show() 
            
			label = cv2.imread(self.label_path + "/" + lbl_name,cv2.IMREAD_GRAYSCALE)
			label = cv2.resize(label, (self.out_rows,self.out_cols))
#			plt.imshow(label)
#			plt.title('label')
#			plt.show()
			label = img_to_array(label)
  
			#img = np.array([img])
			#label = np.array([label])
 
            
			imgdatas[i] = img  
			imglabels[i] = label 
			#if i % 100 == 0:
			print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_train_cropped_800_128.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train_cropped_800_128.npy', imglabels)
		print('Saving to .npy files done.')

	def create_test_data(self):
		i = 0
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		imgs = glob.glob(self.test_path+"/*."+self.img_type)
		print(len(imgs))
		imgs=sorted(imgs)
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			#img = load_img(self.test_path + "/" + midname,grayscale = False,target_size=(self.out_rows,self.out_cols))
			#img = img.thumbnail((1022,712), Image.ANTIALIAS) 
			img = cv2.imread(self.test_path + "/" + midname)
			img = cv2.resize(img, (self.out_rows,self.out_cols))
			img = img_to_array(img) 
#			img[:,:,0] -= 107.546
#			img[:,:,1] -= 60.8877
#			img[:,:,2] -= 29.6568
			#img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = np.array([img])
			imgdatas[i] = img
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_test_cropped_800_128.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')

	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train_cropped_800_128.npy")
		imgs_train = imgs_train.astype('float32')

#		meanR = imgs_train.mean(axis = 0)
#		meanG = imgs_train.mean(axis = 1)
#		meanB = imgs_train.mean(axis = 2)
#		imgs_train[:,:,:,0] -= 34.4804
#		imgs_train[:,:,:,1] -= 68.8604
#		imgs_train[:,:,:,2] -= 118.389
		imgs_train /= 255
        
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train_cropped_800_128.npy")
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_mask_train /= 255
		#imgs_mask_train[imgs_mask_train > 0.5] = 1
		#imgs_mask_train[imgs_mask_train <= 0.5] = 0
		return imgs_train,imgs_mask_train

	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test_cropped_800_128.npy")
		imgs_test = imgs_test.astype('float32')

#		meanR = imgs_test.mean(axis = 0)
#		meanG = imgs_test.mean(axis = 1)
#		meanB = imgs_test.mean(axis = 2)
#		imgs_test[:,:,:,0] -= 34.4804
#		imgs_test[:,:,:,1] -= 68.8604
#		imgs_test[:,:,:,2] -= 118.389
		imgs_test /= 255
		return imgs_test

if __name__ == "__main__":

	#aug = myAugmentation()
	#aug.Augmentation()
	#aug.splitMerge()
	#aug.splitTransform()
	#mydata = dataProcess(2848,4288)
	mydata = dataProcess(128,128)
	mydata.create_train_data()
	mydata.create_test_data()
	#imgs_train,imgs_mask_train = mydata.load_train_data()
	#print imgs_train.shape,imgs_mask_train.shape
