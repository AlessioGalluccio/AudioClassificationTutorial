import pandas as pd
from pathlib import Path
from torch.utils.data import random_split
from data_loader import *
from model import *

# url guide https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

download_path = Path.cwd()/'UrbanSound8K'

# Read metadata file
metadata_file = download_path/'metadata'/'UrbanSound8K.csv'
df = pd.read_csv(metadata_file)
print(df.head())

# Construct file path by concatenating fold and file name and add it to df
df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)

# Take relevant columns
df = df[['relative_path', 'classID']]
print(df.head())

# SPLIT
data_path = "C:/Users/al.galluccio/source/repos/AudioClassificationTutorial/UrbanSound8K/audio"
myds = SoundDS(df, data_path)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
myModel = myModel.to(device)
num_epochs = 2   # Just for demo, adjust this higher.
training(myModel, train_dl, num_epochs)