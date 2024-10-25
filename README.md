# NEURAL-NETWORk-VISION
This is my first project on neural network vision
<br>
AUTHER:SADAQAT HUSSAIN
<br>
This code is run on google colab 
<br>
In this code all the things are here from training to testing
<br>
from google.colab import files
uploaded = files.upload()  # This will open a file dialog to upload the zip file

import os

# Define the name of the extracted folder
extracted_folder = 'FER_2013_dataset'  # Adjust this if you used a different name

# Print the contents of the extracted folder
print("Contents of extracted folder:")
for root, dirs, files in os.walk(extracted_folder):
    print(f"Root: {root}")
    for d in dirs:
        print(f"  Directory: {d}")
    for f in files:
        print(f"  File: {f}")

import zipfile

# Define the paths for the ZIP files
train_zip_path = 'FER_2013_dataset/New folder/train.zip'
test_zip_path = 'FER_2013_dataset/New folder/test.zip'

# Define the extraction directory
extraction_path = 'FER_2013_dataset/'

# Extract train.zip
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

# Extract test.zip
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

# Check the contents of the extraction folder again
print("Contents after extraction:")
for root, dirs, files in os.walk(extraction_path):
    print(f"Root: {root}")
    for d in dirs:
        print(f"  Directory: {d}")
    for f in files:
        print(f"  File: {f}")

# Set the data paths for the ImageDataGenerator
train_data_path = os.path.join(extraction_path, 'train')  # Adjust this based on the extracted structure
test_data_path = os.path.join(extraction_path, 'test')  # Adjust this based on the extracted structure

# Now, you can load the data using ImageDataGenerator as before
# Set the paths for your training and testing data
train_data_path = 'FER_2013_dataset/train'  # Adjust if necessary based on extracted structure
test_data_path = 'FER_2013_dataset/test'     # Adjust if necessary based on extracted structure
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set parameters
image_size = (48, 48)  # Size to which images will be resized
batch_size = 32 # Number of images to process at a time


# Image data generators
train_datagen = ImageDataGenerator(
     rescale=1./255,  # Normalize pixel values to be between 0 and 1
     rotation_range=20,  # Randomly rotate images in the range of 20 degrees
     width_shift_range=0.2,  # Randomly shift images horizontally
     height_shift_range=0.2,  # Randomly shift images vertically
     shear_range=0.2,  # Shear angle in counter-clockwise direction
     zoom_range=0.2,  # Randomly zoom in
     horizontal_flip=True,  # Randomly flip images
     fill_mode='nearest'  # Fill in new pixels with the nearest pixel values
)

#  Add brightness and contrast adjustments. Here's how to modify it:
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,  
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    shear_range=0.2,  
    zoom_range=0.2,  
    horizontal_flip=True,  
    brightness_range=[0.8, 1.2],  # Adjust brightness
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only normalize for testing

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # For multi-class classification
    shuffle=True
)

# Load testing data
test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

# First convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolutional layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output from the convolutional layers
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout to prevent overfitting

# Output layer for 7 classes
model.add(Dense(7, activation='softmax'))  # Adjust the number based on your dataset

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    epochs=30  # You can increase this for better performance
)
# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
import matplotlib.pyplot as plt

# Plot training and validation accuracy and loss
plt.figure(figsize=(8, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
from google.colab import files
uploaded = files.upload()  # This will open a file dialog to upload your image file
# Step 1: Load the necessary libraries
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load the uploaded image
img_path = 'PrivateTest_251881.jpg'  # Use the uploaded file name
img = image.load_img(img_path, target_size=(48, 48))  # Resize to match model input size

# Step 3: Preprocess the image
img_array = image.img_to_array(img)  # Convert to numpy array
img_array = img_array / 255.0  # Normalize the pixel values
img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

# Step 4: Make predictions
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions)  # Get the index of the highest predicted score

# Step 5: Interpret the results
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']  # Adjust as per your dataset
predicted_class_label = class_labels[predicted_class_index]

# Display the image and prediction result
plt.imshow(img)
plt.axis('off')  # Turn off axis
plt.title(f'Predicted Expression: {predicted_class_label}')  # Show the prediction
plt.show()

print(f'The predicted expression is: {predicted_class_label}')  # Print the predicted class





