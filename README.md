# Sign-Language-Classification


This is a project to classify sign language using deep learning approaches.

__Dataset to be used__: Kaggle Sign Language MNIST: https://www.kaggle.com/datamunge/sign-language-mnist
  - The American Sign Language letter database of hand gestures represent a multi-class problem with 24 classes of letters (excluding J and Z which require motion)
  - Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions).
  - The training data (27,455 cases) and test data (7172 cases) are approximately half the size of the standard MNIST but otherwise similar with a header row of label, pixel1,pixel2….pixel784 which represent a single 28x28 pixel image with grayscale values between 0-255.

__Steps to follow:__
  - Creating Custom CNN Model for Classification
  - Using Ensemble Models for Classification
  - Creating Custom CNN Model and Implementing Data Augmentation

__References:__
  - https://youtu.be/3hjsdfTVWRQ - Dr. Sreenivas Bhattiprolu
  - https://www.kaggle.com/razamh/sign-language-classification-98
  - https://www.kaggle.com/madz2000/cnn-using-keras-100-accuracy
  - https://www.kaggle.com/ranjeetjain3/deep-learning-using-sign-langugage
  - https://www.kaggle.com/sayakdasgupta/sign-language-classification-cnn-99-40-accuracy
  - https://www.kaggle.com/drvaibhavkumar/sign-language-classification-using-cnn-acc-99
