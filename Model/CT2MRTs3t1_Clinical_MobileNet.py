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
from torchvision import models
from torchvision.models import mobilenet_v2
from sklearn.metrics import f1_score, classification_report

class MyDataset(Dataset):

    def __init__(self, dataframe, label_transform =None,  transform=None):

        self.df =dataframe
        self.label_transform= label_transform
        self.transform = transform
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        image_path_CT= self.df.loc[index, 'CT']
        image_path_MRI = self.df.loc[index, 'MRI']
       




        img_CT = Image.open(image_path_CT).convert("RGB")
        img_CT= np.array(img_CT)

        img_MRI = Image.open(image_path_MRI).convert("RGB")
        img_MRI= np.array(img_MRI)

        
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
            img_CT = self.transform(img_CT)
            img_MRI=self.transform(img_MRI)


        return img_CT, img_MRI, tabular, label

    
    
class MyDatasetTest(Dataset):

    def __init__(self, dataframe, label_transform =None,  transform=None):

        self.df =dataframe
        self.label_transform= label_transform
        self.transform = transform
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        image_path_CT= self.df.loc[index, 'CT']
        image_path_MRI = self.df.loc[index, 'MRI']
       




        img_CT = Image.open(image_path_CT).convert("RGB")
        img_CT= np.array(img_CT)

        img_MRI = Image.open(image_path_MRI).convert("RGB")
        img_MRI= np.array(img_MRI)

        
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

        
                            
        #if label== 0 and self.transform is not None:
        if self.transform is not None:
            img_CT = self.transform(img_CT)
            img_MRI=self.transform(img_MRI)


        return img_CT, img_MRI, tabular, label, patientID


class CustomModel(nn.Module):
    def __init__(self, num_classes=1):
        super(CustomModel, self).__init__()

        # Load the pre-trained MobileNetV2 model
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=False)

        # Modify the classifier of MobileNetV2
        num_ftrs = self.mobilenet_v2.classifier[1].in_features
        self.mobilenet_v2.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Add dropout for regularization
        )

        # Add an embedding layer to extend features to size 64
        self.embedding = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # Final linear layer for classification
        self.classifier = nn.Sequential(
            nn.Linear(64 + 64 + 64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, image_CT, image_MRI, features):
        # Forward pass through MobileNetV2 for CT image
        x_ct = self.mobilenet_v2(image_CT)

        # Forward pass through MobileNetV2 for MRI image
        x_mri = self.mobilenet_v2(image_MRI)

        # Flatten the features
        features = features.view(features.size(0), -1)

        # Apply the embedding layer to extend features to size 64
        extended_features = self.embedding(features)

        # Concatenate the MobileNetV2 outputs and extended features
        x = torch.cat([x_ct, x_mri, extended_features], dim=-1)

        # Forward pass through the classifier
        x = self.classifier(x)

        return x

def train_epoch(net, trainloader, criterion, optimizer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for i, data in enumerate(trainloader):
        img_ct,img_mri, clinical_data, labels = data

        img_ct,img_mri, clinical_data, labels = img_ct.to(device),img_mri.to(device), clinical_data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(img_ct, img_mri,clinical_data)

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
    val_f1_scores = []  # To store F1 scores

    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            val_ct, val_mri, val_clinical_data, val_labels, patientsID = data

            val_ct, val_mri, val_clinical_data,val_labels = val_ct.to(device),val_mri.to(device), val_clinical_data.to(device),val_labels.to(device)

            # No need to convert labels to torch.tensor
            outputs = net(val_ct,val_mri, val_clinical_data)

            val_labels = val_labels.unsqueeze(1)
            loss = criterion(outputs, val_labels.float())
            val_loss += loss.item()

            # Calculate accuracy
            val_predictions = torch.round(torch.sigmoid(outputs))
            total_valid += val_labels.size(0)
            correct_val_predictions += (val_predictions == val_labels).sum().item()

            # Calculate F1 score
            val_f1 = f1_score(val_labels.cpu(), val_predictions.cpu(), average='binary')  # Assuming binary classification
            val_f1_scores.append(val_f1)

    avg_val_loss = val_loss / len(valid_loader)
    avg_val_acc = correct_val_predictions / total_valid * 100
    avg_val_f1 = sum(val_f1_scores) / len(val_f1_scores)

    return avg_val_loss, avg_val_acc, avg_val_f1, df1


def test_epoch(net, valid_loader, criterion):
    net.eval()

    survival_df=pd.DataFrame()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            test_ct,test_mri, test_clinical_data, test_labels, patientsID= data

            test_ct, test_mri, test_clinical_data, test_labels = test_ct.to(device), test_mri.to(device), test_clinical_data.to(device),test_labels.to(device)


            # No need to convert labels to torch.tensor
            outputs = net(test_ct,test_mri, test_clinical_data)

            test_labels = test_labels.unsqueeze(1)
            
            test_predictions = torch.round(torch.sigmoid(outputs))


            #Classification Report
            # Convert tensors to numpy arrays
            y_true.extend(test_labels.cpu().numpy())
            y_pred.extend(test_predictions.cpu().numpy())


            #Survival Probabaility calclation
            hr_lst  = list (zip(patientsID, test_predictions.detach().cpu().numpy()))
            hr_df_1 = pd.DataFrame(hr_lst, columns = ['PatientID', 'Score'])
            survival_df = survival_df.append(hr_df_1)

    report = classification_report(y_true, y_pred)

    return report, survival_df



# Define a function to save the model
def save_model(model, lr, k, save_dir):
    model_file = os.path.join(save_dir, f"best_model_lr_{lr}_k_{k}.pt")
    torch.save(model, model_file)



# Define a function to save the training and validation curves
def save_training_curves(history, save_dir, lr, k):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves (LR={lr}, K={k})')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"loss_curves_lr_{lr}_k_{k}.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Curves (LR={lr}, K={k})')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"accuracy_curves_lr_{lr}_k_{k}.png"))
    plt.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                       help='path for csv')
    
    args = parser.parse_args()
    path = args.data


    dataset=pd.read_csv(path)
    scaler=MinMaxScaler()
    dataset[['sCD25(IL-2Ra)', '4-1BB', 'CTLA-4', 'PD-L1','PD-1', 'Tim-3']]= scaler.fit_transform(dataset[['sCD25(IL-2Ra)', '4-1BB', 'CTLA-4', 'PD-L1','PD-1', 'Tim-3']])

    k = 3
    # Initialize GroupKFold object
    group_kfold = GroupKFold(n_splits=k)
    num_epochs= 50

    kfold_train_acc=[]
    kfold_test_acc =[]

    X = dataset.drop(columns=['PatientID'])  # Assuming 'patient_id' is your patient ID column
    y = dataset['365']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        #transforms.Normalize([0.0498, 0.0498, 0.0498],[0.1235, 0.1235, 0.1235]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-40, 40)),
        ])
    test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        #transforms.Normalize([0.0498, 0.0498, 0.0498],[0.1235, 0.1235, 0.1235]),
       ])
       
    learning_rates = [0.001]

    # Directory to save the best performing models
    model_save_dir = "/root/code/thesis/codeFolder/LatestDataInUse/results/models/MobileNet/CT2+s3t1_Resnet/Multimodal"
    os.makedirs(model_save_dir, exist_ok=True)  # Ensure directory exists



    results={}
    curve_save_dir = "/root/code/thesis/codeFolder/LatestDataInUse/results/curves/MobileNet/CT2+s3t1_Resnet/Multimodal"
    os.makedirs(curve_save_dir, exist_ok=True)  

    for lr in learning_rates:
        
        print(f"Training with learning rate: {lr}")
        
        #history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'PatientID': [], 'learningRate':[], 'K':[]}
        kfold_train_acc = []
        kfold_test_acc = []

        kfold_number=1
        best_accuracy= 0
        best_model_state_dict = None  # Initialize best model state dictionary
        best_model_fold_number = None  # Initialize best model fold number
        best_learning_rate = None 
        


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
            if len(validation_class_weights) == 1:
                validation_sample_weights = [validation_class_weights[0] for _ in validation_labels]
            else:
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

            model = CustomModel()
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            criterion = nn.BCEWithLogitsLoss()

            history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'PatientID': [], 'learningRate':[], 'K':[]}

            for epoch in range(num_epochs):
                epoch_train_accuracy = []
                epoch_test_accuracy = []

                train_loss, train_correct = train_epoch(model, train_loader, criterion, optimizer)
                test_loss, test_correct, val_f1_score, predictedDataFrame = valid_epoch(model, test_loader, criterion)

                train_loss = train_loss
                train_acc = train_correct
                test_loss = test_loss
                test_acc = test_correct
                
                epoch_train_accuracy.append(train_acc)
                epoch_test_accuracy.append(test_acc)


                if test_acc > best_accuracy:
                    best_model_state_dict= model.state_dict()
                    best_model_fold_number = kfold_number
                    best_learning_rate=lr
                    report_classification, survival_dataframe = test_epoch(model, test_loader, criterion)

                print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} % f1 score {:.2f} %".format(epoch + 1,
                                                                                                                                            num_epochs,
                                                                                                                                            train_loss,
                                                                                                                                            test_loss,
                                                                                                                                            train_acc,
                                                                                                                                            test_acc, val_f1_score))
                                                                                                                                        
                history['train_loss'].append(train_loss)
                history['test_loss'].append(test_loss)
                history['train_acc'].append(train_acc)
                history['test_acc'].append(test_acc)
                history['PatientID'].append(test_acc)
                history['learningRate'].append(lr)
                history['K'].append(k)

            kfold_train_acc.append(np.max(epoch_train_accuracy))
            kfold_test_acc.append(np.max(epoch_test_accuracy))

            # Save training and validation curves
            save_training_curves(history, curve_save_dir, lr, kfold_number)

            kfold_number=kfold_number+1
    

        results[lr] = {
            'train_acc': sum(kfold_train_acc) / len(kfold_train_acc),
            'test_acc': sum(kfold_test_acc) / len(kfold_test_acc),
            'min_test_acc' : np.min(kfold_test_acc),
            'max_test_acc' : np.max(kfold_test_acc),
        }

        save_model(best_model_state_dict, best_learning_rate, best_model_fold_number, model_save_dir)


        # best_model = CustomModel()  # Assuming CustomModel is your model class
        # best_model.load_state_dict(best_model_state_dict)
        # best_model.to(device)
        # report_classification, survival_dataframe = test_epoch(best_model, test_loader, criterion)




        # Save survival dataframe to file
        uniquePatients = survival_dataframe['PatientID'].unique()
        survival_results = []

        for patient in uniquePatients:
 
            patientDataFrame = survival_dataframe[survival_dataframe['PatientID'] == patient]
            patientDataFrame = patientDataFrame.reset_index(drop=True)
            ones = patientDataFrame['Score'].sum()
            survival_rate = ones / len(patientDataFrame)

            # Append patient's survival rate to results
            survival_results.append({'PatientID': patient, 'SurvivalRate': survival_rate})

        
        # Convert survival results to DataFrame
        survival_results_df = pd.DataFrame(survival_results)

        # Save survival results DataFrame to file
        survival_results_df.to_csv(f'/root/code/thesis/codeFolder/LatestDataInUse/results/survival/MobileNet/CT2+s3t1_Resnet/Multimodal/survival_results_lr_{best_learning_rate}.csv', index=False)
        # Save classification report to file
        with open(f'/root/code/thesis/codeFolder/LatestDataInUse/results/reports/MobileNet/CT2+s3t1_Resnet/Multimodal/classification_report_lr_{best_learning_rate}.txt', 'w') as f:
            f.write(report_classification)



                    
    # File to save results
    result_file = "/root/code/thesis/codeFolder/LatestDataInUse/results/CT2+s3t1__MultimodalMobileNet.txt"

    # Open file for writing
    with open(result_file, 'w') as f:
        # Loop over learning rates and results
        for lr, result in results.items():
            # Write learning rate and corresponding results to file
            f.write(f"Learning Rate: {lr}, Train Accuracy: {result['train_acc']}, Test Accuracy: {result['test_acc']}, Minimum Test Accuracy: {result['min_test_acc']}, Maximum Test Accuracy: {result['max_test_acc']}\n")