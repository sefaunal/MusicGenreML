# MusicGenreML
### How To Use:
CNN_Using_PreModel.py: You can use this python file to make predictions by selecting an mp3 or wav file (Required model file has already been included in this repository).

### To Train the Network:
1. Dataset_Preparation.py: With this file you can extract the necessary features from music files. But before running this file you need to download
GTZAN dataset and edit DATASET_PATH variable to make it match your folder's path.
2. CNN_Network.py: After you get your JSON file from Dataset_Preparation.py you can use this file to train the network and export the model.

----------

#### Screenshots
Accuracy value of the network:

![168686538-89d61312-5f7d-42d6-8b0e-339414e4bfba](https://user-images.githubusercontent.com/83312431/170832239-12afd166-a860-4031-9fe5-95d8851843d9.png)

Accuracy value of the network with KFold:

![Ekran görüntüsü 2022-05-28 183359](https://user-images.githubusercontent.com/83312431/170832283-5d38abb3-1675-4472-952f-d7624b094178.png)

An Output from CNN_Using_PreModel.py file:

![image](https://user-images.githubusercontent.com/83312431/168431383-d6b4a1fd-8b57-4859-8ef6-f5aa6d727bc2.png)
