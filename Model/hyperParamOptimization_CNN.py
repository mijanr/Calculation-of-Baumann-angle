from operator import mod
import optuna
import torch
import torch.nn as nn
from dataLoader import CustomDataset
from get_data import Get_Data
import albumentations as A

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
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    layers = []
    #add one convolutional layer with inchannel 1, rest from trial
    layers.append(nn.Conv2d(1, trial.suggest_int("conv1_out_channel", 16, 64), 3, padding=1))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    in_channel = trial.suggest_int("conv1_out_channel", 16, 64)
    for i in range(trial.suggest_int("num_layers", 1, 4)):
        layers.append(nn.Conv2d(in_channel, trial.suggest_int("conv{}_out_channel".format(i+2), 4, 64), 3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2, 2))
        in_channel = trial.suggest_int("conv{}_out_channel".format(i+2), 4, 64)
    layers.append(nn.Flatten())
    layers.append(nn.LazyLinear(trial.suggest_int("lazy_out_features", 512, 1024)))
    
    #in_features is the best of lazy_out_features

    in_features = trial.suggest_int("lazy_out_features", 512, 1024)
    n_layers = trial.suggest_int("n_layers", 1, 5)
    
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 512)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features
    layers.append(nn.Linear(in_features, 8))

    return nn.Sequential(*layers)

def objective(trial):
    model = define_model(trial).to(device)

    #optimizer name
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    train_loader, val_loader = training_data()

    #loss function for regression
    criterion = nn.MSELoss()

    #train the model
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
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
            if i % 10 == 0:
                print("Epoch: {}/{}, Traing Loss: {}".format(epoch, num_epochs, loss_train.item()))
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
                if i % 10 == 0:
                    print("Epoch: {}/{},Val Loss: {}".format(epoch, num_epochs, loss_val.item()))

        trial.report(loss_val.item(), epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return loss_val.item()

if __name__ == '__main__':
    #train, eval, test = train_data()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=500)
    df = study.trials_dataframe()
    df.to_csv('best_hyperParam_CNN.csv')
    #print(df)


    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("Values:", trial.values)
    #save the best parameters
    torch.save(trial.values, 'best_hyperParam_CNN.pt')