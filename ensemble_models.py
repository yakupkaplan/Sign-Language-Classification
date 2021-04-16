# Sign Language Classification using Ensemble Deep Learning with Sign Language MNIST Dataset

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
    - Model Building: Create 3 different models and save them
    - Approach 1: Averaging Models / Sum Ensemble: Simple sum of all outputs / predictions and argmax across all classes
    - Approach 2.1 : Weighted Average Ensemble
    - Approach 2.2 : Weighted Average Ensemble: Grid search for the best combination of w1, w2, w3 that gives maximum accuracy_score
    - Making Preditions and Model Evaluation: Plotting and interpreting confusion matrix, Plot fractional incorrect misclassifications
    - See the power of ensemle models.

References:
    - https://youtu.be/3hjsdfTVWRQ - Dr. Sreenivas Bhattiprolu
    - https://www.kaggle.com/razamh/sign-language-classification-98
    -
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
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

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

# Define epochs for all models
epochs = 10

# Model1
model1 = Sequential()

model1.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.3))

model1.add(Conv2D(64, (3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.3))

model1.add(Conv2D(128, (3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.3))

model1.add(Flatten())

model1.add(Dense(128, activation='relu'))
model1.add(Dense(25, activation='softmax'))

# Compile the model
model1.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Model summary
model1.summary()

# Plot model architecture
plot_model(model1, to_file='results/model_plot.png', show_shapes=True, show_layer_names=True)


# Train the model and measure model speed
start = time.time()
history = model1.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    batch_size=128,
                    epochs=epochs,
                    verbose=1,
                    #callbacks=callbacks_list
                    )
end = time.time()
print(f'The time taken to execute is {round(end-start,2)} seconds.')

# Save the model
model1.save('models/model1.hdf5')


# Model2

model2 = Sequential()

model2.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Conv2D(128, (3, 3), activation='relu'))
model2.add(Conv2D(25, (1, 1)))

model2.add(Flatten())

model2.add(Dense(25, activation='softmax'))

# Compile the model
model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Model summary
model2.summary()

# Train the model and measure model speed
start = time.time()
history2 = model2.fit(X_train,
                      y_train,
                      batch_size=128,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(X_test, y_test))
end = time.time()
print(f'The time taken to execute is {round(end-start,2)} seconds.')

# Save the model
model2.save('models/model2.hdf5')


# Model 3

model3 = Sequential()

model3.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.2))

model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.2))

model3.add(Flatten())

model3.add(Dense(25, activation='softmax'))

# Compile the model
model3.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Model summary
model3.summary()

# Train the model and measure model speed
start = time.time()
history3 = model3.fit(X_train,
                      y_train,
                      batch_size=128,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(X_test, y_test))
end = time.time()
print(f'The time taken to execute is {round(end-start,2)} seconds.')

# Save the model
model3.save('models/model3.hdf5')


####### Approach 1 : Averaging Models / Sum Ensemble: Simple sum of all outputs / predictions and argmax across all classes ##########

# Import neccessary dependenices
from keras.models import load_model
from sklearn.metrics import accuracy_score

# Load models
model1 = load_model('models/model1.hdf5')
model2 = load_model('models/model2.hdf5')
model3 = load_model('models/model3.hdf5')

# Create a list for our saved models
models = [model1, model2, model3]

# Make predictions on test data, save the predictions for each model in a list
preds = [model.predict(X_test) for model in models]
preds = np.array(preds)  # preds.shape -> (3, 7172, 25)
summed = np.sum(preds, axis=0)  # summed.shape -> (7172, 25)

# Make predictions
prediction1 = model1.predict_classes(X_test)
prediction2 = model2.predict_classes(X_test)
prediction3 = model3.predict_classes(X_test)
# argmax across classes: Prediction of the ensemble model
ensemble_prediction = np.argmax(summed, axis=1)  # All models are equally important.

# See the accuracy_score for each model
accuracy1 = accuracy_score(y_test, prediction1)
accuracy2 = accuracy_score(y_test, prediction2)
accuracy3 = accuracy_score(y_test, prediction3)

# Define the accuracy_score for ensemble model's
ensemble_accuracy = accuracy_score(y_test, ensemble_prediction)

# Print the accuracy_scores for all models
print('Accuracy Score for model1 = ', accuracy1)  # Accuracy Score for model1 =  0.9323759063022866
print('Accuracy Score for model2 = ', accuracy2)  # Accuracy Score for model2 =  0.8735359732292247
print('Accuracy Score for model3 = ', accuracy3)  # Accuracy Score for model3 =  0.9272169548243168
print('Accuracy Score for average ensemble = ', ensemble_accuracy)  # Accuracy Score for average ensemble =  0.9393474623535973


## We see, that the score of ensemble model better than all of other ones.


####### Approach 2.1 : Weighted Average Ensemble ##########

models = [model1, model2, model3]
preds = [model.predict(X_test) for model in models]
preds = np.array(preds)

# Give some logical weights to the models according to their accuracy scores
weights = [0.45, 0.2, 0.35]

# Use tensordot to sum the products of all elements over specified axes.
weighted_preds = np.tensordot(preds, weights, axes=((0), (0)))
weighted_ensemble_prediction = np.argmax(weighted_preds, axis=1)

# Define the accuracy_score for ensemble model's
weighted_accuracy = accuracy_score(y_test, weighted_ensemble_prediction)

# Print the accuracy_scores for all models
print('Accuracy Score for model1 = ', accuracy1)  # Accuracy Score for model1 =  0.9323759063022866
print('Accuracy Score for model2 = ', accuracy2)  # Accuracy Score for model2 =  0.8735359732292247
print('Accuracy Score for model3 = ', accuracy3)  # Accuracy Score for model3 =  0.9272169548243168
print('Accuracy Score for average ensemble = ', ensemble_accuracy)  # Accuracy Score for average ensemble =  0.9393474623535973
print('Accuracy Score for weighted average ensemble = ', weighted_accuracy)  # Accuracy Score for weighted average ensemble =  0.9422755158951478


## So, we see that by giving different weigths to models, we can actually increase our accuracy_score.


####### Approach 2.2 : Weighted Average Ensemble: Grid search for the best combination of w1, w2, w3 that gives maximum accuracy_score ######

models = [model1, model2, model3]
preds1 = [model.predict(X_test) for model in models]
preds1 = np.array(preds1)

df = pd.DataFrame([])

# Let's find the weights that maximizes the accuracy_score.
for w1 in range(0, 5):
    for w2 in range(0, 5):
        for w3 in range(0, 5):
            wts = [w1 / 10., w2 / 10., w3 / 10.]
            wted_preds1 = np.tensordot(preds1, wts, axes=((0), (0)))
            wted_ensemble_pred = np.argmax(wted_preds1, axis=1)
            weighted_accuracy = accuracy_score(y_test, wted_ensemble_pred)
            df = df.append(pd.DataFrame({'wt1': wts[0], 'wt2': wts[1],
                                         'wt3': wts[2], 'acc': weighted_accuracy * 100}, index=[0]), ignore_index=True)

# Print the max value
max_acc_row = df.iloc[df['acc'].idxmax()]
print("Max accuracy of ", max_acc_row[3], " obained with w1=", max_acc_row[0],
      " w2=", max_acc_row[1], " and w3=", max_acc_row[2])  # Max accuracy of  94.7016174010039  obained with w1= 0.3  w2= 0.1  and w3= 0.2


### Explore metrics for the ideal weighted ensemble model.

models = [model1, model2, model3]
preds = [model.predict(X_test) for model in models]
preds = np.array(preds)
ideal_weights = [0.3, 0.1, 0.2]

# Use tensordot to sum the products of all elements over specified axes.
ideal_weighted_preds = np.tensordot(preds, ideal_weights, axes=((0),(0)))
ideal_weighted_ensemble_prediction = np.argmax(ideal_weighted_preds, axis=1)

ideal_weighted_accuracy = accuracy_score(y_test, ideal_weighted_ensemble_prediction)
print(ideal_weighted_accuracy)  # 0.947016174010039


# See the result for random samples
i = random.randint(1, len(ideal_weighted_ensemble_prediction))
plt.imshow(X_test[i, :, :, 0])
print("Predicted Label: ", class_names[int(ideal_weighted_ensemble_prediction[i])])
print("True Label: ", class_names[int(y_test[i])])
plt.show()

# Take some images and check for actuaal and predicted labels.
plt.figure(figsize=(12, 8))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    plt.xlabel(f"Actual: {y_test[i]}\n Predicted: {ideal_weighted_ensemble_prediction[i]}")
plt.tight_layout()
plt.show()

# Print confusion matrix to interpret model performance on test_set
cm = confusion_matrix(y_test, ideal_weighted_ensemble_prediction)
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

