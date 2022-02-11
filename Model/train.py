from ResNet import CNN
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import torch.optim as optim
import torch
import torch.nn as nn
from dataLoader import CustomDataset
from get_data import Get_Data
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import albumentations as A
import get_data 
import importlib
importlib.reload(get_data)

#load the data
imageH = 512
imageW = 512

key_points_path_train, key_points_path_test = Get_Data().patient_level_split()
X_train, y_train = Get_Data().json_to_data(key_points_path_train, imageH, imageW)
X_test, y_test = Get_Data().json_to_data(key_points_path_test, imageH, imageW)

#make a testloader
test_dataset = CustomDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

#check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#create the model
model = CNN(output_dim=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

#apply transformations
transform = A.Compose([
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.3)
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    )
#k-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf.get_n_splits(X_train)

for train_idx, val_idx in kf.split(X_train):
    #split the data
    X_train_fold = [X_train[i] for i in train_idx]
    y_train_fold = [y_train[i] for i in train_idx]
    X_val_fold = [X_train[i] for i in val_idx]
    y_val_fold = [y_train[i] for i in val_idx]

    #create the dataloader
    train_dataset = CustomDataset(X_train_fold, y_train_fold, transform=transform)
    val_dataset = CustomDataset(X_val_fold, y_val_fold, transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

    #train the model
    writer = SummaryWriter()
    for epoch in trange(1, 200):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.view(-1, 8).to(device)

            outputs = model(images)

            optimizer.zero_grad()            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        writer.add_scalar('Training loss: ', loss, epoch)
        print("Epoch: ", epoch, "Training Loss: ", loss.item())
        #validate the model
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                labels = labels.view(-1, 8).to(device)
                images = images.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)
                writer.add_scalar('Validation loss: ', loss, epoch)
        print("Epoch: ", epoch, "Validation Loss: ", loss.item())
    writer.close()

#save the model on cpu
model.cpu()

#test the model
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.view(-1, 8).to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
    print("Test Loss: ", loss.item())