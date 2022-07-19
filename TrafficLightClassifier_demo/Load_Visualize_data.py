import cv2
import helpers # helper functions

import random
import matplotlib 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%matplotlib inline

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

## TODO: Write code to display an image in IMAGE_LIST (try finding a yellow traffic light!)
## TODO: Print out 1. The shape of the image and 2. The image's label

# ------------------- Global Definitions -------------------

# Definition of the 3 possible traffic light states and theirs label
tl_states = ['red', 'yellow', 'green']
tl_state_red = 0
tl_state_yellow = 1
tl_state_green = 2
tl_state_count = 3
tl_state_red_string = tl_states[tl_state_red]
tl_state_yellow_string = tl_states[tl_state_yellow]
tl_state_green_string = tl_states[tl_state_green]

# Index of image and label in image set
image_data_image_index = 0
image_data_label_index = 1

# Normalized image size
default_image_size = 32

# ---------------- End of Global Definitions ---------------

fig = plt.figure(figsize=(20,40))

example_count = 24
if example_count>len(IMAGE_LIST):
    example_count = len(IMAGE_LIST)
    
chosen = set()

# print 24 random examples, prevent double choice
for example_index in range(example_count):
    tries = 0
    
    while tries<2:
        index = 0
        tries += 1
        if example_index==0: # first choice should be a yellow light
            for iterator in range(len(IMAGE_LIST)):
                if IMAGE_LIST[iterator][image_data_label_index]==tl_state_yellow_string:
                    index = iterator
                    break
        else: # all other choices are random
            index = random.randint(0, len(IMAGE_LIST)-1)
        
        if index in chosen: # try a second time if chosen already
            continue
        chosen.add(index)
        
    example_image = IMAGE_LIST[index][image_data_image_index]
    result = "{}, shape: {}".format(IMAGE_LIST[index][image_data_label_index],example_image.shape)
    ax = fig.add_subplot(example_count, 4, example_index+1, title=result)
    ax.imshow(example_image.squeeze())
    
fig.tight_layout(pad=0.7)

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

# TODO: Display a standardized image and its label

fig = plt.figure(figsize=(20,40))

# 12 example pairs
example_count = 12
if example_count>len(IMAGE_LIST):
    example_count = len(IMAGE_LIST)
total_count = example_count*2

chosen = set() # use set to prevent double random selection

for example_index in range(example_count):

    tries = 0
    index = 0
    
    # select next image
    while tries<2:
        tries += 1
        index = random.randint(0, len(IMAGE_LIST)-1)
        
        if index in chosen:
            continue
        chosen.add(index)
        
    eff_index = example_index*2
    
    # print original
    example_image = IMAGE_LIST[index][image_data_image_index]
    result = "{} {}".format(IMAGE_LIST[index][image_data_label_index],example_image.shape)
    ax = fig.add_subplot(total_count, 4, eff_index+1, title=result)
    ax.imshow(example_image.squeeze())
    
    # print standardized counterpiece
    eff_index += 1
    example_image = STANDARDIZED_LIST[index][image_data_image_index]
    result = "{} {}".format(STANDARDIZED_LIST[index][image_data_label_index],example_image.shape)
    ax = fig.add_subplot(total_count, 4, eff_index+1, title=result)
    ax.imshow(example_image.squeeze())

fig.tight_layout(pad=0.7)