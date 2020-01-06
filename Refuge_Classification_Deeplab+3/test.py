
# coding: utf-8

# In[11]:


from keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import imutils
import cv2
import os
#from utils import save_images
from imutils import paths
import csv
from model import Deeplabv3
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import Model_DiscSeg as DiscModel
from skimage.transform import resize
from utils_Mnet import pro_process, BW_img, disc_crop,save_images
import skimage.measure
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from skimage.io import imsave
from keras.preprocessing import image


# In[12]:



#modelName='MICCAI2018_classi_DeepLabv3_09_201808231055_99907.hdf5'
modelName=['MICCAI2018_classi_DeepLabv3_21_201808261436.hdf5']


#dataset='../xunet/data/Validation400/validation_800'
#save_path='./results/DeepLabv3_09_validation'
#dataset='../xunet/data/train/train_segmentation_600_400_cropped/'

dataset='../xunet/data/train/Original/Original_all/'
#dataset = '../xunet/data/Validation400/REFUGE-Validation400/'
Region_save_path='./ROI_DeepLabv3_21_train/'
Output_save_Path='./results/DeepLabv3_21_train/'

if not os.path.exists(Region_save_path):
   os.makedirs(Region_save_path)

if not os.path.exists(Output_save_Path):
   os.makedirs(Output_save_Path)

image_width=256
image_high=256

DiscROI_size = 600
DiscSeg_size = 640


# In[13]:


DiscSeg_model = DiscModel.DeepModel(size_set=DiscSeg_size)
DiscSeg_model.load_weights('Model_DiscSeg_ORIGA_pretrain_Gen_ROI.h5')


# In[14]:


imagePaths = sorted(list(paths.list_images(dataset)))
# load the trained convolutional neural network
print("[INFO] loading network...")

model_1=Deeplabv3(input_shape=(image_width,image_high,3), classes=2)
model_1.load_weights(modelName[0])




# In[15]:


header = ['FileName','Glaucoma Risk']
with open('result_DeepLabv3_21_train_ROI600.csv', 'wb') as csvfile:
    # loop over the input images
    #writer = csv.DictWriter(csvfile, fieldnames =header)
    #writer.writeheader()
    for imagePath in imagePaths:

        test_image = np.asarray(image.load_img(imagePath))
        midname = imagePath[imagePath.rindex("/")+1:]
        org_img = test_image.copy()   
       
        
        test_image = resize(test_image, (DiscSeg_size, DiscSeg_size, 3)) 
        test_image = np.reshape(test_image, (1,) + test_image.shape)*255
        [prob_6, prob_7, prob_8, prob_9, prob_10] = DiscSeg_model.predict([test_image])
        #image = np.expand_dims(image, axis=0)
        
        org_img_disc_map = BW_img(np.reshape(prob_10, (DiscSeg_size, DiscSeg_size)), 0.5)
        #plt.imshow(org_img_disc_map)
        #plt.title('org_img_disc_map')
        #plt.show()       
        
        regions = regionprops(label(org_img_disc_map))
      
        C_x = int(regions[0].centroid[0] * org_img.shape[0] / DiscSeg_size)
        C_y = int(regions[0].centroid[1] * org_img.shape[1] / DiscSeg_size)
        org_img_disc_region, err_coord, crop_coord = disc_crop(org_img, DiscROI_size, C_x, C_y)
        
        imsave(os.path.join(Region_save_path, '%s' % midname),org_img_disc_region)    
    
        # pre-process the image for classification
        input_image = cv2.imread(os.path.join(Region_save_path, '%s' % midname))
        input_image = cv2.resize(input_image, (image_width, image_high))
        input_image = img_to_array(input_image)
        #plt.imshow(input_image)
        #plt.title('input_image')
        #plt.show()        
        input_image = np.expand_dims(input_image, axis=0)
       
        ##the below is the mean of 600x600 cropped training images
        input_image[:,:,:,0] -= 107.546        
        input_image[:,:,:,1] -= 60.8877
        input_image[:,:,:,2] -= 29.6568

        input_image = input_image.astype("float") / 255.0
    
        # classify the input image
        (notGlau_1, Glau_1) = model_1.predict(input_image)[0]
     
        
        notGlau_array=np.array([notGlau_1])
        notGlau = np.mean(notGlau_array,dtype=np.float64)
        
        Glau_array=np.array([Glau_1])
        Glau = np.mean(Glau_array,dtype=np.float64)
        
        # build the label
        label_name = "Glaucoma" if Glau > notGlau else "Not Glaucoma"
        proba = Glau if Glau > notGlau else notGlau
        label_name = "{}: {:.2f}%".format(label_name, proba * 100)
        print(midname+':'+label_name)
        
        # draw the label on the image
        output = imutils.resize(org_img, width=400)
        cv2.putText(output, label_name, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
        	0.7, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(Output_save_Path, '%s' % midname ),output)
    
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([midname, Glau])
        cv2.destroyAllWindows()
        

