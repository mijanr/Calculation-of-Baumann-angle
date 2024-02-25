from operator import mod
from pickletools import optimize
import optuna
import torch
import torch.nn as nn
from dataLoader import CustomDataset
from get_data import Get_Data
import albumentations as A
from ResNet import CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def training_data():
    #load the data
    imageH = 512
    imageW = 512

    key_points_path_train, key_points_path_test = Get_Data().patient_level_split()
    X_train, y_train = Get_Data().json_to_data(key_points_path_train, imageH, imageW)
    X_test, y_test = Get_Data().json_to_data(key_points_path_test, imageH, imageW)

    #apply transformations
    transform = A.Compose([
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.3)
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    )

    train_dataset = CustomDataset(X_train, y_train, transform=transform)
    val_dataset = CustomDataset(X_test, y_test, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)
    return train_loader, val_loader

def define_model(trial):
    return CNN(8).to(device)

def objective(trial):
    model = define_model(trial).to(device)
    #find the best optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    train_loader, val_loader = training_data()
    criterion = nn.MSELoss()
    #find the best epochs
    epochs = trial.suggest_categorical("epochs", [100, 200, 300, 400])
    #train the model
    model.train()
    for epoch in range(trial.suggest_categorical("epochs", [100, 200, 300, 400])):
        for i, (img, key_points) in enumerate(train_loader):
            img, key_points = img.to(device), key_points.to(device)
            #flatten the image
            #img = img.view(-1, img.shape[1] * img.shape[2] * img.shape[3])
            outputs = model(img)
            key_points = key_points.view(-1, 8)
            # print('Generated key points: ', outputs.shape)
            # print('Original key points: ', key_points.shape)
            loss_train = criterion(outputs, key_points)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        #print loss
        print('Epoch: ', epoch, 'Training Loss: ', loss_train.item())
        #validation
        model.eval()
        with torch.no_grad():
            for i, (img, key_points) in enumerate(val_loader):
                img, key_points = img.to(device), key_points.to(device)
                #flatten the image
                #img = img.view(-1, img.shape[1] * img.shape[2] * img.shape[3])
                outputs = model(img)
                key_points = key_points.view(-1, 8)
                loss_val = criterion(outputs, key_points)
            print('Epoch: ', epoch, 'Validation Loss: ', loss_val.item())

        trial.report(loss_val.item(), epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return loss_val.item()
    
if __name__ == '__main__':
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=500)
    df = study.trials_dataframe()
    df.to_csv('best_hyperParam_ResNet.csv')
    #print(df)


    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("Values:", trial.values)
    #save the best parameters
    torch.save(trial.values, 'best_hyperParam_ResNet.pt')
