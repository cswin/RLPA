import gc
import os
import sys
import glob
from keras.preprocessing.image import   img_to_array, load_img
import numpy as np
 
from PIL import Image
from keras import losses

 
from keras import backend as K


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#https://github.com/ternaus/kaggle_dstl_submission/blob/master/src/unet_buildings.py
smooth = 1e-12
def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + losses.binary_crossentropy(y_pred, y_true)


w = 0.01
def custom_loss_dice_mbe(y_true, y_pred):
             loss1=losses.mean_absolute_error(y_true,y_pred)
             loss2=dice_coef_loss(y_true, y_pred)
             return loss1*0.98+loss2*0.02  
         
def custom_loss_jac_mbe(y_true, y_pred):
             loss1=losses.mean_absolute_error(y_true,y_pred)
             loss2=-K.log(jaccard_coef(y_true, y_pred)) 
             return loss1*0.98+loss2*0.02       
            
def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


 
def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation, also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # References
    Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
    What is a good evaluation measure for semantic segmentation?.
    IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.
    https://en.wikipedia.org/wiki/Jaccard_index
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
 

 


def soft_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=axis)
    area_true = K.sum(y_true, axis=axis)
    area_pred = K.sum(y_pred, axis=axis)
    return (2 * intersection + smooth) / (area_true + area_pred + smooth)
    
def hard_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    return soft_sorensen_dice(y_true_int, y_pred_int, axis, smooth)

sorensen_dice = hard_sorensen_dice

def sorensen_dice_loss(y_true, y_pred, weights):
    # Input tensors have shape (batch_size, height, width, classes)
    # User must input list of weights with length equal to number of classes
    #
    # Ex: for simple binary classification, with the 0th mask
    # corresponding to the background and the 1st mask corresponding
    # to the object of interest, we set weights = [0, 1]
    batch_dice_coefs = soft_sorensen_dice(y_true, y_pred, axis=[1, 2])
    dice_coefs = K.mean(batch_dice_coefs, axis=0)
    w = K.constant(weights) / sum(weights)
    return 1 - K.sum(w * dice_coefs)

def soft_jaccard(y_true, y_pred, axis=None, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=axis)
    area_true = K.sum(y_true, axis=axis)
    area_pred = K.sum(y_pred, axis=axis)
    union = area_true + area_pred - intersection
    return (intersection + smooth) / (union + smooth)

def hard_jaccard(y_true, y_pred, axis=None, smooth=1):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    return soft_jaccard(y_true_int, y_pred_int, axis, smooth)

jaccard = hard_jaccard

def jaccard_loss(y_true, y_pred, weights):
    batch_jaccard_coefs = soft_jaccard(y_true, y_pred, axis=[1, 2])
    jaccard_coefs = K.mean(batch_jaccard_coefs, axis=0)
    w = K.constant(weights) / sum(weights)
    return 1 - K.sum(w * jaccard_coefs)

def weighted_categorical_crossentropy(y_true, y_pred, weights, epsilon=1e-8):
    ndim = K.ndim(y_pred)
    ncategory = K.int_shape(y_pred)[-1]
    # scale predictions so class probabilities of each pixel sum to 1
    y_pred /= K.sum(y_pred, axis=(ndim-1), keepdims=True)
    y_pred = K.clip(y_pred, epsilon, 1-epsilon)
    w = K.constant(weights) * (ncategory / sum(weights))
    # first, average over all axis except classes
    cross_entropies = -K.mean(y_true * K.log(y_pred), axis=tuple(range(ndim-1)))
    return K.sum(w * cross_entropies)

def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        return np.array(im).reshape(1, im.size[1], im.size[0], 1)
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 1))
    return data


def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8')).convert('L')
    im.save(filepath, 'bmp')

def test_model(myxunet):
    modelpath="./"
    modelName='deeplabv3_0116.hdf5'
    data_path="../data/test/train/"
    save_path="../results/0116"
    out_rows=512 
    out_cols=512
    
    
    # init dataPorcess object class
    #mydata = dataProcess(256,512)
    ## load the test data
    #imgs_test = mydata.load_test_data()
    
    
    imgs = glob.glob(data_path+"*.jpg")
    print(len(imgs))
    imgs_test = np.ndarray((len(imgs),out_rows,out_cols,3), dtype=np.uint8)
    imgs=sorted(imgs)
    i = 0
    for imgname in imgs:
        midname = imgname[imgname.rindex("/")+1:]
        img = load_img(data_path + "/" + midname,grayscale = False)
    
        img = img_to_array(img) 
        imgs_test[i] = img  
    
        print('Done: {0}/{1} images'.format(i, len(imgs)))
        i += 1
    print('loading done')
    np.save(save_path + '/imgs_test.npy', imgs_test)
    
    print('Saving to imgs_test.npy files done.')
    
    
    # init our network
    model = myxunet.get_unet()
    #load the trained model weights
    model.load_weights(modelpath+modelName)
    
    #load test data
    imgs_test_data = np.load(save_path+"/imgs_test.npy")
    imgs_test_data = imgs_test_data.astype('float32')
    imgs_test_data[:,:,:,0] -= 68.9289
    imgs_test_data[:,:,:,1] -= 40.9241 
    imgs_test_data[:,:,:,2] -= 20.571 
    imgs_test_data /= 255
    #feed the test data into model
    imgs_mask_test = model.predict({'inputs_modality1': imgs_test_data,
                                    'inputs_modality2': imgs_test_data,
                                    'inputs_modality3': imgs_test_data},
                                           batch_size=1, verbose=1)
    #save the results
    np.save(save_path+'/imgs_test_results.npy', imgs_mask_test)

    
    #get the results as images format
    print("array to image")
    results = np.load(save_path+'/imgs_test_results.npy')#load the saved result
    for i in range(results.shape[0]):
    		img=np.squeeze(results[i]) 
#    		img[img < 0.5] = 1
    		img = img * 255
    		img = np.clip(img,0,255).astype('uint8')
    		midname = imgs[i][imgname.rindex("/")+1:]
    		print(midname)
    		midname=midname[0:midname.index('.')]
    		save_images(os.path.join(save_path, '%s.bmp' % midname ), img)
            
 