'''
loading datasets
DEPENDENCIES: torchvision is not installed
pip install torchvision from cmd
'''

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

#Loading a Dataset
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

#Iterating and Visualizing the Dataset
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

'''
Creating a Custom Dataset for your files
 must implement three functions: __init__, __len__, and __getitem__
the FashionMNIST IMAGES are STORED in a DIRECTORY img_dir
LABELS are STORED separately in a CSV FILE annotations_file
'''
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
# The __init__ function is run once when instantiating the Dataset object. 
# We initialize the directory containing the images, the annotations file, and both transforms 
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
#returns number of samples in dataset
    def __len__(self):
        return len(self.img_labels) 
#     loads and returns a sample from the dataset at the given index idx
#     identifies the image’s location on disk, converts that to a tensor using read_image
#     retrieves the corresponding label from the csv data in self.img_labels
#     calls the transform functions on them (if applicable)
#     returns the tensor image and corresponding label in a tuple
# 
#info on transformations: https://pytorch.org/vision/stable/transforms.html
    #https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

#PREPARING YOUR DATA FOR TRAINING WITH DATALOADERS
#set batch size and and data shuffling 
from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

#ITERATE THROUGH THE DATALOADER
'''
Each iteration below returns a batch of train_features and train_labels
Because we specified shuffle=True, after we iterate over all batches the data is shuffled
'''
#fine tune data loading order: https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")