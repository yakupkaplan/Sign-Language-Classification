# Sign Language Classification using Deep Learning with Sign Language MNIST Dataset

"""
This is a project to classify sign language using deep learning approaches.

Dataset to be used: Kaggle Sign Language MNIST: https://www.kaggle.com/datamunge/sign-language-mnist
    - The American Sign Language letter database of hand gestures represent a multi-class problem with 24 classes of letters (excluding J and Z which require motion)
    - Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions).
    - The training data (27,455 cases) and test data (7172 cases) are approximately half the size of the standard MNIST
        but otherwise similar with a header row of label, pixel1,pixel2….pixel784 which represent a single 28x28 pixel image with grayscale values between 0-255.

Steps to follow:
    - Import Dependencies, Read the Data
    - Data Preprocessing
    - Data Augmentation
    - Model Building: Create custom CNN model, Compile the model, Model summary, Plot model architecture, Using callbacks, Train the model and measure model speed
    - Performance Interpretation: Plot the training and validation accuracy and loss at each epoch to be able interpret the model performance and check for over- and underfitting
    - Making Preditions and Model Evaluation: Plotting and interpreting confusion matrix, Plot fractional incorrect misclassifications

References:
    - https://youtu.be/3hjsdfTVWRQ - Dr. Sreenivas Bhattiprolu
    - https://www.kaggle.com/razamh/sign-language-classification-98
    - https://www.kaggle.com/madz2000/cnn-using-keras-100-accuracy
    - https://www.kaggle.com/ranjeetjain3/deep-learning-using-sign-langugage
    - https://www.kaggle.com/sayakdasgupta/sign-language-classification-cnn-99-40-accuracy
    - https://www.kaggle.com/drvaibhavkumar/sign-language-classification-using-cnn-acc-99
"""

# Import Dependencies
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer

from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings("ignore")


# Define the working directory
import os
root = 'C:/Users/yakup/PycharmProjects/sign_language_classification'
os.chdir(root)
os.getcwd()

# Define data pathes
train_path = 'data/sign_mnist_train.csv'
test_path = 'data/sign_mnist_test.csv'

# Read the datasets
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# See how many labels we have in the dataset
train['label'].nunique()  # 24

# Let's check, if we have samples with label 9 and 25.
len(train.loc[train['label'] == 9, :])  # 0
len(train.loc[train['label'] == 25, :])  # 0


## Data Preprocessing

# Convert the datasets into numpy arrays for efficiency.
train_data = np.array(train, dtype='float32')
test_data = np.array(test, dtype='float32')

# Define class labels for easy interpretation
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
len(class_names)

# Sanity check - plot a few images and labels
i = random.randint(1, train.shape[0])
fig1, ax1 = plt.subplots(figsize=(2, 2))
plt.imshow(train_data[i, 1:].reshape((28, 28)), cmap='gray')
plt.show()
print("Label for the image is: ", class_names[int(train_data[i, 0])])

# Data distribution visualization -> Dataset seems to be fairly balanced. No balancing operation is needed.
plt.style.use("ggplot")
plt.figure(figsize=(15, 15))
sns.countplot(x='label', data=train)
plt.show()

# Define X_train  -- Normalize / scale X values
X_train = train_data[:, 1:]  # /255.
X_test = test_data[:, 1:]  # /255.

# Convert y to categorical if planning on using categorical_crossentropy. No need to do this if using sparse_categorical_crossentropy.
y_train = train_data[:, 0]
# y_train_cat = to_categorical(y_train, num_classes=24)
y_test = test_data[:, 0]
# y_test_cat = to_categorical(y_test, num_classes=24)

# Reshape for the neural network
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Converting the integer labels to binary form
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# Take a look at some samples from train dataset
plt.figure(figsize=(9, 7))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.xlabel(np.argmax(y_train[i]))
plt.show()

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=(1./255), rotation_range=30,
                                  width_shift_range=0.2, height_shift_range=0.2,
                                  shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=(1./255))


## Model Building

# Custom CNN Model

model = Sequential()

model.add(Conv2D(75, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=2, padding='same'))

model.add(Conv2D(50, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=2, padding='same'))

model.add(Conv2D(25, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=2, padding='same'))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(24, activation="softmax"))

# Model summary
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Plot model architecture
plot_model(model, to_file='results/model_plot.png', show_shapes=True, show_layer_names=True)


# Define callbacks to prevent overfitting and evaluate the model performance in a better way.
file_location = "models/custom_cnn_model_with_data_augmentation.{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}.h5"

# A feature of CallBack API that keeps track of performance of the model and stores the best performaces at different time steps.
checkpoint = ModelCheckpoint(file_location, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Define early stopping to prevent overfitting
# earlystop = EarlyStopping(monitor='val_acc', verbose=1, mode='max')

# Used for reducing learning rate when a metric has stopped improving.
reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, mode='max', min_lr=0.00001)

# List of callbacks used from CallBack APIs
callbacks_list = [checkpoint,
                  # earlystop,
                  reduce_lr]


# Train the model and measure model speed
start = time.time()

history = model.fit_generator(generator=train_datagen.flow(X_train, y_train, batch_size=128),
                              validation_data=val_datagen.flow(X_test, y_test),
                              epochs=15,
                              verbose=1,
                              callbacks=callbacks_list)

end = time.time()
print(f'The time taken to execute is {round(end-start,2)} seconds.')

# Check the model performance on test data
loss, acc = model.evaluate(val_datagen.flow(X_test, y_test))
print(f"Accuracy: {acc*100}")  # Accuracy: 99.19130206108093
print(f"Loss: {loss}")  # Loss: 0.03060048632323742

# 225/225 [==============================] - 1s 6ms/step - loss: 0.0306 - acc: 0.9919
# Accuracy: 99.19130206108093
# Loss: 0.03060048632323742


# Let's see the results
# Plot the training and validation accuracy and loss at each epoch to be able interpret the model performance and check for over- and underfitting
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# plt.savefig('results/loss_control.jpeg', dpi=fig.dpi)

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# plt.savefig('results/accuracy_control.png', dpi=fig.dpi)


# # Analysis after Model Training
#
# epochs = [i for i in range(20)]
# fig, ax = plt.subplots(1, 2)
# train_acc = history.history['acc']
# train_loss = history.history['loss']
# val_acc = history.history['val_acc']
# val_loss = history.history['val_loss']
# fig.set_size_inches(16, 9)
#
# ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
# ax[0].plot(epochs, val_acc, 'ro-', label='Testing Accuracy')
# ax[0].set_title('Training & Validation Accuracy')
# ax[0].legend()
# ax[0].set_xlabel("Epochs")
# ax[0].set_ylabel("Accuracy")
#
# ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
# ax[1].plot(epochs, val_loss, 'r-o', label='Testing Loss')
# ax[1].set_title('Testing Accuracy & Loss')
# ax[1].legend()
# ax[1].set_xlabel("Epochs")
# ax[1].set_ylabel("Loss")
# plt.show()


# Make predictions
predictions = model.predict_classes(X_test)
for i in range(len(predictions)):
    if(predictions[i] >= 9):
        predictions[i] += 1
predictions[:5]

# See the classification report
classes = ["Class " + str(i) for i in range(25) if i != 9]
print(classification_report(y_test, predictions, target_names=classes))

# Create confusion matrix
cm = confusion_matrix(y_test, predictions)
cm = pd.DataFrame(cm, index=[i for i in range(25) if i != 9], columns=[i for i in range(25) if i != 9])
plt.figure(figsize=(15,15))
sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='')
