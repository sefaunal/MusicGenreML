# MusicGenreML
A machine learning project that classifies music into various genres using Convolutional Neural Networks (CNN). The project leverages the GTZAN dataset and pre-trained models to predict music genres based on audio features.

<br>

## How To Use

### 1. CNN_Using_PreModel.py
Use this Python file to make genre predictions on audio files (MP3 or WAV format).  
*Note:* The required pre-trained model file is already included in this repository.

### 2. Model Training

#### Dataset_Preparation.py
This script extracts the necessary features from the music files. Before running it, you'll need to download the *GTZAN* dataset and set the DATASET_PATH variable to point to the correct directory where the dataset is stored.

#### CNN_Network.py
Once you’ve prepared the dataset (via Dataset_Preparation.py), use this script to train the neural network. It will generate and export the trained model in JSON format.

<br>

## Screenshots

Below are some visuals that show the accuracy of the network:

### Accuracy Value of the Network

![Network Accuracy](https://user-images.githubusercontent.com/83312431/170832239-12afd166-a860-4031-9fe5-95d8851843d9.png)

### Accuracy Value of the Network with K-Fold Cross Validation

![KFold Accuracy](https://user-images.githubusercontent.com/83312431/170832283-5d38abb3-1675-4472-952f-d7624b094178.png)

### Output from CNN_Using_PreModel.py File

![Model Prediction Output](https://user-images.githubusercontent.com/83312431/168431383-d6b4a1fd-8b57-4859-8ef6-f5aa6d727bc2.png)

