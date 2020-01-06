import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout,BatchNormalization,Activation,advanced_activations
from keras.layers import Conv2DTranspose,Concatenate 
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from data import dataProcess
from utils import save_images,test_model
#from non_local import non_local_block
from keras.layers.merge import concatenate, add
from se import squeeze_excite_block

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
save_path="../results/0142/ROI1024T700V_validation_v2/"

modelName='xunet_0142_128_200e_ROI1024T700V_v1.hdf5'
class myUnet(object):
 
	def __init__(self, img_rows = 128, img_cols = 128):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_mask_train = mydata.load_train_data()
		imgs_test = mydata.load_test_data()
		return imgs_train, imgs_mask_train, imgs_test


	def get_unet(self):
        
        # ***********3 inputs***********************
		inputs_modality1 = Input((self.img_rows, self.img_cols, 3), name='inputs_modality1')
		conv_m1_01 = Conv2D(64, 3, activation='relu', padding='same',
                      kernel_initializer='he_normal')(inputs_modality1)
		conv_m1_02 = Conv2D(64, 3, activation='relu', padding='same',
                            kernel_initializer='he_normal')(conv_m1_01)

		inputs_modality2 = Input((self.img_rows, self.img_cols, 3), name='inputs_modality2')
		conv_m2_01 = Conv2D(64, 3, activation='relu', padding='same',
                            kernel_initializer='he_normal')(inputs_modality2)
		conv_m2_02 = Conv2D(64, 3, activation='relu', padding='same',
                            kernel_initializer='he_normal')(conv_m2_01)

		inputs_modality3 = Input((self.img_rows, self.img_cols, 3), name='inputs_modality3')
		conv_m3_01 = Conv2D(64, 3, activation='relu', padding='same',
                            kernel_initializer='he_normal')(inputs_modality3)
		conv_m3_02 = Conv2D(64, 3, activation='relu', padding='same',
                            kernel_initializer='he_normal')(conv_m3_01)
		# ***********3 inputs end***********************

		# ***********Feature blending ***********************#
		FB_00 = Concatenate()([conv_m1_02, conv_m2_02, conv_m3_02])
		pool1 = MaxPooling2D(pool_size=(2, 2))(FB_00)


		conv2 = Conv2D(64, 3, padding='same',
                       kernel_initializer='he_normal')(pool1)
		init=conv2

		conv2 = BatchNormalization(axis=-1, epsilon=1e-3)(conv2)
		conv2 = Activation('relu')(conv2)   

		conv2 = Conv2D(64, 3,  padding='same',
                       kernel_initializer='he_normal')(conv2)
		# squeeze and excite block
		conv2 = squeeze_excite_block(conv2)
		conv2 = add([init, conv2])
		conv2 = BatchNormalization(axis=-1, epsilon=1e-3)(conv2)
		conv2 = Activation('relu')(conv2)   

		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

###


		conv3 = Conv2D(128, 3, padding='same',
                       kernel_initializer='he_normal')(pool2)
		init=conv3
		conv3 = BatchNormalization(axis=-1, epsilon=1e-3)(conv3)
		conv3 = Activation('relu')(conv3)   

		conv3 = Conv2D(128, 3,  padding='same',
                       kernel_initializer='he_normal')(conv3)
		# squeeze and excite block
		conv3 = squeeze_excite_block(conv3)
		conv3 = add([init, conv3])
		conv3 = BatchNormalization(axis=-1, epsilon=1e-3)(conv3)
		conv3 = Activation('relu')(conv3)   


#        

		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

###



		conv4 = Conv2D(256, 3, padding='same',
                       kernel_initializer='he_normal')(pool3)
		init=conv4
		conv4 = BatchNormalization(axis=-1, epsilon=1e-3)(conv4)
		conv4 = Activation('relu')(conv4)

		conv4 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv4)
		# squeeze and excite block
		conv4 = squeeze_excite_block(conv4)
		conv4 = add([init, conv4])


		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
###
		conv5 = Conv2D(512, 3, padding='same',
                       kernel_initializer='he_normal')(pool4)
		init=conv5
		conv5 = BatchNormalization(axis=-1, epsilon=1e-3)(conv5)
		conv5 = Activation('relu')(conv5)

		conv5 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv5)
		# squeeze and excite block
		conv5 = squeeze_excite_block(conv5)
		conv5 = add([init, conv5])
		conv5 = BatchNormalization(axis=-1, epsilon=1e-3)(conv5)
		conv5 = Activation('relu')(conv5)        
#		drop4 = Dropout(0.5)(conv5)

        
#		conv5 = Conv2D(512, 1, padding='same',
#                       kernel_initializer='he_normal')(conv5)
#		conv5 = BatchNormalization(axis=-1, epsilon=1e-3)(conv5)        
#		conv5 = Activation('relu')(conv5)
		pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
###
        
        
		up6 = Conv2D(512, 3, padding='same',
                     kernel_initializer='he_normal')(pool5)

		up6 = BatchNormalization(axis=-1, epsilon=1e-3)(up6)        
		up6 = Activation('relu')(up6)
        
		up6 = Conv2DTranspose(512, 3,  strides=(2, 2), padding='same',kernel_initializer='he_normal')(up6)
		init=up6
		up6 = BatchNormalization(axis=-1, epsilon=1e-3)(up6)
		up6 = Activation('relu')(up6) 
		up6 = Conv2D(512, 1, padding='same',
                       kernel_initializer='he_normal')(up6)
		# squeeze and excite block
		up6 = squeeze_excite_block(up6)
		up6 = add([init, up6])
		up6 = BatchNormalization(axis=-1, epsilon=1e-3)(up6)        
		up6 = Activation('relu')(up6)
        

		merge6 = Concatenate()([conv5, up6])
        
        
        
        
		conv6 = Conv2D(256, 3,  padding='same',
                       kernel_initializer='he_normal')(merge6)

		conv6 = BatchNormalization(axis=-1, epsilon=1e-3)(conv6)
		conv6 = Activation('relu')(conv6)         
        
 

		up7 = Conv2DTranspose(256, 3,  strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv6)
		init=up7
		up7 = BatchNormalization(axis=-1, epsilon=1e-3)(up7)
		up7 = Activation('relu')(up7)    
		up7 = Conv2D(256, 1, padding='same',
                       kernel_initializer='he_normal')(up7)
		# squeeze and excite block
		up7 = squeeze_excite_block(up7)
		up7 = add([init, up7])
		up7 = BatchNormalization(axis=-1, epsilon=1e-3)(up7)        
		up7 = Activation('relu')(up7)
        
		merge7 = Concatenate()([conv4, up7])
        
        
        
		conv7 = Conv2D(128, 3,  padding='same',
                       kernel_initializer='he_normal')(merge7)
		conv7 = BatchNormalization(axis=-1, epsilon=1e-3)(conv7)
		conv7 = Activation('relu')(conv7)   
   
		up8 = Conv2DTranspose(128, 3,  strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv7)
		init=up8     
 
		up8 = BatchNormalization(axis=-1, epsilon=1e-3)(up8)
		up8 = Activation('relu')(up8)  
		up8 = Conv2D(128, 1, padding='same',
                       kernel_initializer='he_normal')(up8)
		# squeeze and excite block
		up8 = squeeze_excite_block(up8)
		up8 = add([init, up8])
		up8 = BatchNormalization(axis=-1, epsilon=1e-3)(up8)        
		up8 = Activation('relu')(up8)

        
		merge8 = Concatenate()([conv3, up8])
        
        
		conv8 = Conv2D(64, 3,  padding='same',
                       kernel_initializer='he_normal')(merge8)
		conv8 = BatchNormalization(axis=-1, epsilon=1e-3)(conv8)
		conv8 = Activation('relu')(conv8)  
		up9 = Conv2DTranspose(64, 3,  strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv8)
		init=up9
		up9 = BatchNormalization(axis=-1, epsilon=1e-3)(up9)
		up9 = Activation('relu')(up9)  
		up9 = Conv2D(64, 1, padding='same',
                       kernel_initializer='he_normal')(up9)
		# squeeze and excite block
		up9 = squeeze_excite_block(up9)
		up9 = add([init, up9])
		up9 = BatchNormalization(axis=-1, epsilon=1e-3)(up9)        
		up9 = Activation('relu')(up9)        
        
        
		merge9 = Concatenate()([conv2, up9])
        
        
		conv9 = Conv2D(32, 3,  padding='same',
                       kernel_initializer='he_normal')(merge9)
		conv9 = BatchNormalization(axis=-1, epsilon=1e-3)(conv9)
		conv9 = Activation('relu')(conv9) 
		up10 = Conv2DTranspose(32, 3,  strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv9)
		init=up10    

		up10 = BatchNormalization(axis=-1, epsilon=1e-3)(up10)
		up10 = Activation('relu')(up10)  
		up10 = Conv2D(32, 1, padding='same',
                       kernel_initializer='he_normal')(up10)
		# squeeze and excite block
		up10 = squeeze_excite_block(up10)
		up10 = add([init, up10])
		up10 = BatchNormalization(axis=-1, epsilon=1e-3)(up10)        
		up10 = Activation('relu')(up10)        
        
        
		merge9 = Concatenate()([FB_00, up10])
        


		# *************************************u-net*****************************

		# *************************************outputs***************************
		conv10_m1 = Conv2D(32, 3, activation='relu', padding='same',
                          kernel_initializer='he_normal')(merge9)
		conv10_m1 = Conv2D(32, 3, activation='relu', padding='same',
                          kernel_initializer='he_normal')(conv10_m1)
		conv10_m1 = Conv2D(1, 1,   name='conv10_m1')(conv10_m1)


		# *************************************outputs******************************

		model = Model(input=[inputs_modality1, inputs_modality2, inputs_modality3], 
                      output=[conv10_m1])
#		model.load_weights(pretrainedModel)
		model.compile(optimizer = Adam( lr = 1e-4), loss = 'mean_absolute_error', metrics = ['accuracy'])
#		model.compile(optimizer = Adam(lr = 1e-4), loss = bce_dice_loss, metrics = ['accuracy'])
  
		return model



	def train(self):

		print("loading data")
		imgs_train, imgs_mask_train, imgs_test = self.load_data()
		print("loading data done")
		model = self.get_unet()
		print("got unet")
		early_stopping = EarlyStopping(patience=10, verbose=1)
		reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
		model_checkpoint = ModelCheckpoint(modelName, monitor='val_loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit({'inputs_modality1': imgs_train, 'inputs_modality2': imgs_train, 'inputs_modality3': imgs_train},
            {'conv10_m1': imgs_mask_train}, batch_size=5, nb_epoch=200, verbose=1,  validation_split=0.3,
            shuffle=True, callbacks=[reduce_lr,model_checkpoint,early_stopping])

#		print('predict test data')
# 
#		imgs_mask_test = model.predict({'inputs_modality1': imgs_test, 'inputs_modality2': imgs_test, 'inputs_modality3': imgs_test},
#                                       batch_size=1, verbose=1)
#        
#		np.save(save_path+'/imgs_mask_test.npy', imgs_mask_test)

	def save_img(self):

		print("array to image")
		imgs = np.load(save_path+'/imgs_mask_test.npy')
		for i in range(imgs.shape[0]):
			#img = imgs[i]
			img = np.squeeze(imgs[i]) * 255
			img = np.clip(img,0,255).astype('uint8')
			save_images(os.path.join(save_path, '%d.jpg' % (i)), img)
			#img = array_to_img(img)
			#img.save("./results/%d.jpg"%(i))




if __name__ == '__main__':
	myunet = myUnet()
	myunet.train()
#	myunet.save_img()
	print('predict test model')
	test_model(myunet)

