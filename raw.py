
import numpy as np
from numpy.random import randint
from tensorflow.keras.layers import Activation, Conv2D, Dropout, LeakyReLU, Input, Concatenate, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization 


def discriminatorModel(imageShape):
    # weight initialization
    model = Sequential()
    model.add(Input(shape=imageShape))
    model.add(Conv2D(64, (4, 4), strides=(2,2), padding='same',input_shape=imageShape))
    model.add(InstanceNormalization(axis=-1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (4, 4), strides=(2,2), padding='same'))
    model.add(InstanceNormalization(axis=-1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, (4, 4), strides=(2,2), padding='same'))
    model.add(InstanceNormalization(axis=-1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(512, (4, 4), strides=(2,2), padding='same'))
    model.add(InstanceNormalization(axis=-1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(512, (4, 4), strides=(1,1), padding='same'))
    model.add(InstanceNormalization(axis=-1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Conv2D(1, (4,4)))
    # We need to slow down the rate at which the descriminator learns
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'],loss_weights=[0.5])
    return model
model = discriminatorModel((256,256,3))    
model.summary()

def residualBlock(n_filters, input_layer):
    init = RandomNormal(stddev=0.02)
    resTensor = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
    resTensor =InstanceNormalization(axis=-1)(resTensor)
    resTensor = Activation('relu')(resTensor)
    resTensor = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(resTensor)
    resTensor =InstanceNormalization(axis=-1)(resTensor)
    resTensor = Concatenate()([resTensor,input_layer])
    return resTensor

def generatorModel(LImageShape, LResnet=9):
    # init = RandomNormal(stddev=0.02)
    
    in_shape = Input(shape=LImageShape)
    genTensor = Conv2D(64, (7,7), padding = 'same',)(in_shape)
    genTensor = InstanceNormalization(axis=-1)(genTensor)
    genTensor = Activation('relu')(genTensor)
    genTensor = Conv2D(128, (3,3), strides = (2,2), padding = 'same')(genTensor)
    genTensor = InstanceNormalization(axis=-1)(genTensor)
    genTensor = Activation('relu')(genTensor)
    genTensor = Conv2D(258, (3,3), strides = (2,2), padding = 'same')(genTensor)
    genTensor = InstanceNormalization(axis=-1)(genTensor)
    genTensor = Activation('relu')(genTensor)
    
    for _ in range(LResnet):
        genTensor = residualBlock(256, genTensor)
    
    genTensor = Conv2DTranspose(128, (3,3), strides = (2,2), padding = 'same')(genTensor)
    genTensor = InstanceNormalization(axis=-1)(genTensor)
    genTensor = Conv2DTranspose(64, (3,3), strides = (2,2), padding = 'same')(genTensor)
    genTensor = InstanceNormalization(axis=-1)(genTensor)
    genTensor = Conv2D(3, (7,7), padding='same', activation='tanh',)(genTensor)
    genTensor = InstanceNormalization(axis=-1)(genTensor)
    
    #TO-DO: This part
    
    out_image = Activation('tanh')(genTensor)
    
    # define model
    model = Model(in_shape, out_image)
    return model



model = generatorModel((256,256,3)) 
model.summary() 


# Definition of a composite function model
def compositeModel(g_model_1, d_model, g_model_2, image_shape):
    g_model_1.trainable = True
    d_model.trainable = False
    g_model_2.trainable = False

    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)

    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)

    output_f = g_model_2(gen1_out)
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)

    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer='adam')
    return model

 
# input shape
image_shape = (256,256,3)
# generator: A -> B
g_model_AtoB = generatorModel(image_shape)
# generator: B -> A
g_model_BtoA = generatorModel(image_shape)
# discriminator: A -> [real/fake]
d_model_A = discriminatorModel(image_shape)
# discriminator: B -> [real/fake]
d_model_B = discriminatorModel(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = compositeModel(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = compositeModel(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, patch_shape, patch_shape, 1))
	return X, y


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
	print(dataset.shape)
	# generate fake instance
	X = g_model.predict(dataset)
	# create 'fake' class labels (0)
	y = np.zeros((len(X), patch_shape, patch_shape, 1))
	return X, y


# Import libraries needed for loading data
from os import listdir
from  PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image


# 
path_to_dataset = '/home/beks/Documents/School/AI/AI-Group-Project/images/'
path_to_paintings = path_to_dataset+"trainA/"
path_to_photos = path_to_dataset+"trainB/"
paths_to_training_data = [path_to_paintings,path_to_photos]


# Funtion to load images from DB
# Returns a list of paintings and photos

def load_data():
    dataset = []
    for index,path  in enumerate(paths_to_training_data):
        # create a mini list of data
        photos,paintings = ([],)*2
        if index == 0:
            # Load paintings
            for file in listdir(path):
                image_data = image.imread(path+file)
                paintings.append(image_data)
        else:
            # Load photos
            for file in listdir(path):
                image_data = image.imread(path+file)
                photos.append(image_data)
        dataset = [paintings[0:1],photos[0:1]]
    return dataset


# Load images
dataset = load_data()

# Update image pool
def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif np.random() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return np.asarray(selected)



# Define a function for training our models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
	# define properties of the training run
	n_epochs, n_batch, = 100, 1
	# determine the output square shape of the discriminator
	n_patch = d_model_A.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	trainA = np.asarray(dataset[0])
	trainB = np.asarray(dataset[1])
	# prepare image pool for fakes
	poolA, poolB = list(), list()
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
		X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
		X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
		# update fakes from pool
		X_fakeA = update_image_pool(poolA, X_fakeA)
		X_fakeB = update_image_pool(poolB, X_fakeB)
		# update generator B->A via adversarial and cycle loss
		g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
		# update discriminator for A -> [real/fake]
		dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
		dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
		# update generator A->B via adversarial and cycle loss
		g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
		# update discriminator for B -> [real/fake]
		dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
		dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
		# summarize performance
		print(f"{i+1}\nDisA loss:{dA_loss1}\nDisA loss2: {dA_loss2}\nDisB loss: {dB_loss1}\nDisB loss2: {dB_loss2}\nGen loss: {g_loss1}\nGen loss2: {g_loss2}")
	
	# Save model after training 
	print("Saving model.....")
	g_model_AtoB.save("gen_AtoB")
	g_model_BtoA.save("gen_BtoA")
	d_model_A.save("d_model_A")
	d_model_B.save("d_model_B")



train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)
# Alert me
print("Creating notifier....")
import os
from time import sleep
for i in range(0,5):
    sleep(3)
    os.system('spd-say "Baah, Model trained, come back here now"')


# Loading model 
from tensorflow.keras.models import load_model

cust = {'InstanceNormalization': InstanceNormalization}
model = load_model('gen_AtoB', cust)