
# coding: utf-8

# In[1]:


import numpy as np
import scipy.misc
from keras.preprocessing import image
from keras.preprocessing.image import  img_to_array
from skimage.transform import rotate, resize
from skimage.measure import label, regionprops
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
from skimage.color import gray2rgb
from skimage.color import rgb2gray
from time import time
from utils_Mnet import pro_process, BW_img, disc_crop,save_images
#import matplotlib.pyplot as plt
#from skimage.io import imsave
from train import myUnet
#from PIL import Image
import pydensecrf.densecrf as dcrf
import cv2

#import cv2
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import Model_DiscSeg as DiscModel


# In[7]:


DiscROI_size = 700
DiscSeg_size = 640
ModelInput_size = 512

test_data_type = '.jpg'
mask_data_type ='.bmp'

Original_test_img_path = '../xunet/data/Validation400/REFUGE-Validation400/'

maskDisc_save_path='maskDisc_save_path_20180910/'
maskCup_save_path='maskCup_save_path_20180910/'
seg_result_save_path = 'result_vali400_xunet_0142_512_20180910_ROI1024T800/'
temp_files_save_path='temp/'


if not os.path.exists(seg_result_save_path):
    os.makedirs(seg_result_save_path)

if not os.path.exists(maskDisc_save_path):
    os.makedirs(maskDisc_save_path)     

if not os.path.exists(maskCup_save_path):
    os.makedirs(maskCup_save_path)   

if not os.path.exists(temp_files_save_path):
    os.makedirs(temp_files_save_path)      


# In[8]:


##Original_image = Image which has to labelled
##Mask image = Which has been labelled by some technique..
def crf(original_image, mask_img):
    
    # Converting annotated image to RGB if it is Gray scale
    if(len(mask_img.shape)<3):
        mask_img = gray2rgb(mask_img)

#     #Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)
    
#     # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = len(set(labels.flat)) 
    
    #Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 10 steps 
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0],original_image.shape[1]))


# In[10]:


file_test_list = sorted([file for file in os.listdir(Original_test_img_path) if file.lower().endswith(test_data_type)])
print(str(len(file_test_list)))
 

DiscSeg_model = DiscModel.DeepModel(size_set=DiscSeg_size)
DiscSeg_model.load_weights('../Refuge_Classification_Deeplab+3/RModel_DiscSeg_ORIGA_pretrain.h5')


model0142 = myUnet(ModelInput_size)
model0142.load_weights('xunet_0142_512_200e_ROI1024T700V_v1.hdf5')


# In[11]:


for lineIdx in range(0, len(file_test_list)):
    temp_txt = [elt.strip() for elt in file_test_list[lineIdx].split(',')]
    #print(' Processing Img: ' + temp_txt[0])
    # load image
#    org_img = np.asarray(image.load_img(Original_test_img_path + temp_txt[0]))
    
    
    org_img = cv2.imread(Original_test_img_path + temp_txt[0])
#    org_img = cv2.resize(org_img, (DiscSeg_size, DiscSeg_size))
    org_img = img_to_array(org_img)
    #plt.imshow(org_img)
    #plt.title('org_img')
    #plt.show()
    #imsave('org_img.jpg',org_img)              
      
    # Disc region detection by U-Net
    temp_org_img = cv2.resize(org_img, (DiscSeg_size, DiscSeg_size)) 
    #plt.imshow(temp_org_img)
    #plt.title('temp_org_img')
    #plt.show()
                
    
    temp_org_img = np.reshape(temp_org_img, (1,) + temp_org_img.shape)*255

    [prob_6, prob_7, prob_8, prob_9, prob_10] = DiscSeg_model.predict([temp_org_img])
    
    #plt.imshow(np.squeeze(np.clip(prob_10*255,0,255).astype('uint8')))
    #plt.title('temp_img')
    #plt.show()
   

    

    org_img_disc_map = BW_img(np.reshape(prob_10, (DiscSeg_size, DiscSeg_size)), 0.5)
   
    regions = regionprops(label(org_img_disc_map))

    
    C_x = int(regions[0].centroid[0] * org_img.shape[0] / DiscSeg_size)
    C_y = int(regions[0].centroid[1] * org_img.shape[1] / DiscSeg_size)
    org_img_disc_region, err_coord, crop_coord = disc_crop(org_img, DiscROI_size, C_x, C_y)


    #plt.imshow(org_img_disc_region)
    #plt.title('org_img_disc_region')
    #plt.show()
   

    # Disc and Cup segmentation by M-Net
#    run_time_start = time()

    
    temp_img = pro_process(org_img_disc_region, ModelInput_size)
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    
        
    temp_img = temp_img.astype('float32')
    temp_img /= 255
 
    #feed the test data into model
    result = model0142.predict({'inputs_modality1': temp_img,
                                'inputs_modality2': temp_img,
                                'inputs_modality3': temp_img},
                                           batch_size=1, verbose=1)

#    run_time_end = time()

    # Extract mask
    result = np.reshape(result, (result.shape[1], result.shape[2]))
    result = scipy.misc.imresize(result, (DiscROI_size, DiscROI_size))
#    imsave('disc_result_000.jpg',disc_result)    

#    result=crf(org_img_disc_region,result)
    #plt.imshow(result)
    #plt.title('result_CRF')
    #plt.show()  

    result[result<50]=20
    result[(result<200) & (result>50)]=128
    result[result>200]=255
   
    
    result = result.astype('float32')
  
    #plt.imshow(result)
    #plt.title('result')
    #plt.show()

 

 
 
    ROI_result =result.astype('uint8')
     
    #plt.imshow(ROI_result)
    #plt.title('ROI_result')
    #plt.show()     
 
    
    Img_result = np.zeros((org_img.shape[0],org_img.shape[1]), dtype=int)
    Img_result[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ] = ROI_result[err_coord[0]:err_coord[1], err_coord[2]:err_coord[3], ]

    #plt.imshow(Img_result)
    #plt.title('Img_result')
    #plt.show() 
    

    Img_result[Img_result==0]=255
    Img_result[Img_result==20]=0
    
    #plt.imshow(Img_result)
    #plt.title('Img_result')
   # plt.show() 
    nameLen=len(temp_txt[0])
#    imsave(seg_result_save_path+temp_txt[0][:nameLen-4]+mask_data_type,Img_result)
    imsave(seg_result_save_path+temp_txt[0][:nameLen-4]+mask_data_type,Img_result)    

