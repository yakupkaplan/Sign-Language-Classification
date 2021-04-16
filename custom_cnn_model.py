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
    - Model Building: Create custom CNN model, Compile the model, Model summary, Plot model architecture, Using callbacks, Train the model and measure model speed
    - Performance Interpretation: Plot the training and validation accuracy and loss at each epoch to be able interpret the model performance and check for over- and underfitting
    - Making Preditions and Model Evaluation: Plotting and interpreting confusion matrix, Plot fractional incorrect misclassifications

References:
    - https://youtu.be/3hjsdfTVWRQ - Dr. Sreenivas Bhattiprolu
    - https://www.kaggle.com/razamh/sign-language-classification-98
"""

# Import Dependencies
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
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
train['label'].nunique()

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
fig = plt.figure(figsize=(18, 18))
# sns.set_theme(style = "darkgrid")
ax = sns.countplot(x="label", data=train)
ax.set_ylabel('Count')
ax.set_title('Label')
plt.show()

# Normalize / scale X values
X_train = train_data[:, 1:] /255.
X_test = test_data[:, 1:] /255.

# Convert y to categorical if planning on using categorical_crossentropy. No need to do this if using sparse_categorical_crossentropy.
y_train = train_data[:, 0]
# y_train_cat = to_categorical(y_train, num_classes=24)

y_test = test_data[:, 0]
# y_test_cat = to_categorical(y_test, num_classes=24)

# Reshape for the neural network
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))

# Take a look at some samples from train dataset
plt.figure(figsize=(9, 7))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.xlabel(np.argmax(y_train[i]))
plt.show()


## Model Building

# Custom CNN Model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(25, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Model summary
model.summary()

# Plot model architecture
plot_model(model, to_file='results/model_plot.png', show_shapes=True, show_layer_names=True)


# Define callbacks to prevent overfitting and evaluate the model performance in a better way.
file_location = "models/custom_cnn_model.{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}.h5"

# A feature of CallBack API that keeps track of performance of the model and stores the best performaces at different time steps.
checkpoint = ModelCheckpoint(file_location, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Used for reducing learning rate when a metric has stopped improving.
reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, mode='max', min_lr=0.00001)

# List of callbacks used from CallBack APIs
callbacks_list = [checkpoint, reduce_lr]


# Train the model and measure model speed
start = time.time()
history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    callbacks=callbacks_list)
end = time.time()
print(f'The time taken to execute is {round(end-start,2)} seconds.')


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


## Making Preditions and Model Evaluation

predictions = model.predict_classes(X_test)

accuracy = accuracy_score(y_test, predictions)
print('Accuracy Score = ', accuracy)  # Accuracy Score =  0.9440881204684886

# See the result for random samples
i = random.randint(1, len(predictions))
plt.imshow(X_test[i, :, :, 0])
print("Predicted Label: ", class_names[int(predictions[i])])
print("True Label: ", class_names[int(y_test[i])])
plt.show()

# Take some images and check for actuaal and predicted labels.
plt.figure(figsize=(12, 8))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    plt.xlabel(f"Actual: {y_test[i]}\n Predicted: {predictions[i]}")
plt.tight_layout()
plt.show()

# Print confusion matrix to interpret model performance on test_set
cm = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(figsize=(18, 18))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)
plt.show()
# plt.savefig('results/confusion_matrix.png', dpi=fig.dpi)

# Plot fractional incorrect misclassifications
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
fig, ax = plt.subplots(figsize=(12, 12))
plt.bar(np.arange(24), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')
plt.xticks(np.arange(24), class_names)
plt.show()
# plt.savefig('results/fractional_incorrect_misclassifications.png', dpi=fig.dpi)

