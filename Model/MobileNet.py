import os
import torch
import pandas as pd
import numpy as np
import torchvision
import torch.nn.functional as F
from PIL import Image
import argparse


from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold


from sklearn.preprocessing import MinMaxScaler
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import mobilenet_v2


class MyDataset(Dataset):

    def __init__(self, dataframe, label_transform =None,  transform=None):

        self.df =dataframe
        
        #self.df = self.df.sample(n=100, random_state=42).reset_index(drop=True)  
        
        #self.img_dir = img_dir
        self.label_transform= label_transform
        self.transform = transform
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df.loc[index, 'Images']
        img = Image.open(image_path).convert("RGB")
       
        img= np.array(img)
        
        label = self.df.loc[index,'365']
        patientsID= self.df.loc[index, 'PatientID']
        
        
        ## clinical 
        sCD25 = self.df.loc[index,'sCD25(IL-2Ra)']
        BB14 = self.df.loc[index,'4-1BB']
        CTLA = self.df.loc[index,'CTLA-4']
        PDL1 = self.df.loc[index,'PD-L1'] 
        PD = self.df.loc[index,'PD-1']
        Tim = self.df.loc[index,'Tim-3']
        #patientID = self.df.loc[index,'PatientID']
        
        
        tabular = [[sCD25, BB14,CTLA , PDL1, PD, Tim]]
        tabular = torch.FloatTensor(tabular)

        
                            
        #if label== 0 and self.transform is not None:
        if self.transform is not None:
            img = self.transform(img)


        return img, label

    
    
class MyDatasetTest(Dataset):

    def __init__(self, dataframe, label_transform =None,  transform=None):

        self.df =dataframe
        self.label_transform= label_transform
        self.transform = transform
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df.loc[index, 'Images']
        img = Image.open(image_path).convert("RGB")

        img= np.array(img)
        
        label = self.df.loc[index,'365']
        patientsID= self.df.loc[index, 'PatientID']
        
        
        ## clinical 
        sCD25 = self.df.loc[index,'sCD25(IL-2Ra)']
        BB14 = self.df.loc[index,'4-1BB']
        CTLA = self.df.loc[index,'CTLA-4']
        PDL1 = self.df.loc[index,'PD-L1'] 
        PD = self.df.loc[index,'PD-1']
        Tim = self.df.loc[index,'Tim-3']
        patientID = self.df.loc[index,'PatientID']
        
        
        tabular = [[sCD25, BB14,CTLA , PDL1, PD, Tim]]
        tabular = torch.FloatTensor(tabular)

        
                            
        if self.transform is not None:
            img = self.transform(img)


        return img, label, patientsID

def train_epoch(net, trainloader, criterion, optimizer):

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for i, data in enumerate(trainloader):
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)

        labels = labels.unsqueeze(1)

        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Calculate accuracy
        predictions = torch.round(torch.sigmoid(outputs))
        total += labels.size(0)
        correct += predictions.eq(labels).sum().item()

    avg_train_loss = train_loss / len(trainloader)
    avg_train_acc = correct / total * 100

    return avg_train_loss, avg_train_acc

def valid_epoch(net, valid_loader, criterion):

    net.eval()
    val_loss = 0
    correct_val_predictions = 0
    total_valid = 0
    df1=pd.DataFrame()

    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            inputs, val_labels, patientsID = data

            inputs, val_labels = inputs.to(device), val_labels.to(device)

            # No need to convert labels to torch.tensor
            outputs = net(inputs)

            val_labels = val_labels.unsqueeze(1)
            loss = criterion(outputs, val_labels.float())
            val_loss += loss.item()

            # Calculate accuracy
            val_predictions = torch.round(torch.sigmoid(outputs))
            total_valid += val_labels.size(0)
            #correct += predictions.eq(labels).sum().item()
            correct_val_predictions += (val_predictions == val_labels).sum().item()

            

            #######

            # print(predictions)

            # hr_lst  = list (zip(patientsID, predictions.detach().cpu().numpy()))
            # hr_df_1 = pd.DataFrame(hr_lst, columns = ['PatientID', 'Score'])
            # #print("Data Frame: ",hr_df_1.shape)
            # #print("Data Frame: ", hr_df_1.head())
            # df1 = df1.append(hr_df_1)

    avg_val_loss = val_loss / len(valid_loader)
    avg_val_acc = correct_val_predictions / total_valid * 100
    #print('Correct',correct_val_predictions)
    #print('Total',total_valid)

    #print(df1['Score'].value_counts())

    return avg_val_loss, avg_val_acc, df1


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                       help='path for csv')
    
    args = parser.parse_args()
    path = args.data

    dataset=pd.read_csv(path)
    scaler=MinMaxScaler()
    dataset[['sCD25(IL-2Ra)', '4-1BB', 'CTLA-4', 'PD-L1','PD-1', 'Tim-3']]= scaler.fit_transform(dataset[['sCD25(IL-2Ra)', '4-1BB', 'CTLA-4', 'PD-L1','PD-1', 'Tim-3']])

    k = 4
    # Initialize GroupKFold object
    group_kfold = GroupKFold(n_splits=k)
    num_epochs=50

    kfold_train_acc=[]
    kfold_test_acc =[]
    X = dataset.drop(columns=['PatientID'])  # Assuming 'patient_id' is your patient ID column
    y = dataset['365']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.1391, 0.1391, 0.1391],[0.1779, 0.1779, 0.1779]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-40, 40)),
        ])
    test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        #transforms.CenterCrop((224, 224)),
        transforms.Normalize([0.1391, 0.1391, 0.1391],[0.1779, 0.1779, 0.1779]),
       ])
       
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    # Store results
    results = {}

    for lr in learning_rates:
        
        print(f"Training with learning rate: {lr}")
        
        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'PatientID': []}
        kfold_train_acc = []
        kfold_test_acc = []


        for train_index, test_index in group_kfold.split(X, y, groups=dataset['PatientID']):
            
            train_data, test_data = dataset.iloc[train_index], dataset.iloc[test_index]
            train_data = train_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)

            training_labels = train_data['365'].tolist()

            # Calculate weights for each class based on the class distribution
            training_class_counts = torch.tensor([training_labels.count(label) for label in set(training_labels)])
            training_class_weights = 1.0 / training_class_counts.float()

            # Assign weights to each sample based on its class
            training_sample_weights = [training_class_weights[label] for label in training_labels]

            # Create a WeightedRandomSampler
            training_sampler = WeightedRandomSampler(weights=training_sample_weights, num_samples=len(training_sample_weights), replacement=True)

            #####################################

            validation_labels = test_data['365'].tolist()

            # Calculate weights for each class based on the class distribution
            validation__class_counts = torch.tensor([validation_labels.count(label) for label in set(validation_labels)])
            validation_class_weights = 1.0 / validation__class_counts.float()

            # Assign weights to each sample based on its class
            validation_sample_weights = [validation_class_weights[label] for label in validation_labels]

            # Create a WeightedRandomSampler
            validation_sampler = WeightedRandomSampler(weights=validation_sample_weights, num_samples=len(validation_sample_weights), replacement=True)

            #################

            train_dataset = MyDataset(
            dataframe=train_data,
            transform= train_transform) 

            test_dataset = MyDatasetTest(
                dataframe=test_data,
                transform=test_transform)

            train_loader = DataLoader(train_dataset, batch_size=16, sampler= training_sampler)
            test_loader = DataLoader(test_dataset, batch_size=16, sampler= validation_sampler)

            model = mobilenet_v2(pretrained=False)
            n_inputs = model.classifier[1].in_features
            model.classifier[1] = nn.Sequential(
                nn.Linear(n_inputs, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                #nn.BatchNorm1d(64),
                nn.Linear(64, 1)
            )
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            criterion = nn.BCEWithLogitsLoss()

            for epoch in range(num_epochs):
                epoch_train_accuracy = []
                epoch_test_accuracy = []

                train_loss, train_correct = train_epoch(model, train_loader, criterion, optimizer)
                test_loss, test_correct, predictedDataFrame = valid_epoch(model, test_loader, criterion)

                train_loss = train_loss
                train_acc = train_correct
                test_loss = test_loss
                test_acc = test_correct

                epoch_train_accuracy.append(train_acc)
                epoch_test_accuracy.append(test_acc)

                print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                                            num_epochs,
                                                                                                                                            train_loss,
                                                                                                                                            test_loss,
                                                                                                                                            train_acc,
                                                                                                                                            test_acc))
                history['train_loss'].append(train_loss)
                history['test_loss'].append(test_loss)
                history['train_acc'].append(train_acc)
                history['test_acc'].append(test_acc)
                history['PatientID'].append(test_acc)

            kfold_train_acc.append(sum(epoch_train_accuracy) / len(epoch_train_accuracy))
            kfold_test_acc.append(sum(epoch_test_accuracy) / len(epoch_test_accuracy))

        results[lr] = {
            'train_acc': sum(kfold_train_acc) / len(kfold_train_acc),
            'test_acc': sum(kfold_test_acc) / len(kfold_test_acc)
        }

    # File to save results
    result_file = "/root/code/thesis/codeFolder/LatestDataInUse/results/MRT4SequencesImageOnly224MobileNetBatch16Epoch50Validationsampeler.txt"

    # Open file for writing
    with open(result_file, 'w') as f:
        # Loop over learning rates and results
        for lr, result in results.items():
            # Write learning rate and corresponding results to file
            f.write(f"Learning Rate: {lr}, Train Accuracy: {result['train_acc']}, Test Accuracy: {result['test_acc']}\n")


    # # Print results
    # for lr, result in results.items():
    #     print(f"Learning Rate: {lr}, Train Accuracy: {result['train_acc']}, Test Accuracy: {result['test_acc']}")



