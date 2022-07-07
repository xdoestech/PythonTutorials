'''
file showing how to augment an image multiple ways to create a new set of images
allows you to make get more data quickly
good solution to overfitting
'''
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
#%matplotlib inline

#plot image function 
#https://www.tensorflow.org/tutorials/images/classification#visualize_training_images
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#creates images with the specified parameters
#https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15, zoom_range=0.1, 
    channel_shift_range=10., horizontal_flip=True)

#filename of randomly chosen dog image 
chosen_image = random.choice(os.listdir('CNNwithKeras/data/dogs-vs-cats/train/dog'))
#create path to image chosen above
image_path = 'CNNwithKeras/data/dogs-vs-cats/train/dog/' + chosen_image
#the image itself from path as np array
image = np.expand_dims(plt.imread(image_path),0)
#show the selected image
plt.imshow(image[0])
#generate batch of augmented images
#aug_iter = gen.flow(image)
    #save the augmented data
aug_iter = gen.flow(image, save_to_dir='CNNwithKeras/data/dogs-vs-cats/train/dog', save_prefix='aug-image-', save_format='jpeg')

#get 10 samples of augmented image
aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
#show the augmented images
plotImages(aug_images)
