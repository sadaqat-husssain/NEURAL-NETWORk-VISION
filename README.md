# NEURAL-NETWORk-VISION
This is my first project on neural network vision
<br>
AUTHER:SADAQAT HUSSAIN
<br>
The first normalization code
<br>
import numpy as np
from scipy.io import loadmat

# Load the SVHN data using scipy.io.loadmat
train_data = loadmat('E:/PROJECT DATA/train_32x32.mat')
test_data = loadmat('E:/PROJECT DATA/test_32x32.mat')

# Extract the images and labels from the dataset
train_images = train_data['X']  # Images are in a (32, 32, 3, N) format
train_labels = train_data['y'].squeeze()  # Labels are in the 'y' field

test_images = test_data['X']
test_labels = test_data['y'].squeeze()

# Re-arrange axes of the image data to (N, 32, 32, 3)
train_images = np.transpose(train_images, (3, 0, 1, 2))
test_images = np.transpose(test_images, (3, 0, 1, 2))

# Normalize pixel values to [0, 1] by dividing by 255
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Check normalization output
print(f"First pixel value before normalization: 27")
print(f"First pixel value after normalization: {27 / 255}")


