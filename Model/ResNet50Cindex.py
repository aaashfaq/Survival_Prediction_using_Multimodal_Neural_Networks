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
from torchvision.models import resnet50
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import precision_score, recall_score, roc_auc_score


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
        #img = Image.open(image_path).convert("RGB")

        img = np.load(image_path)
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
       
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
        #img = Image.open(image_path).convert("RGB")
        img = np.load(image_path)
        img = Image.fromarray(img.astype('uint8')).convert('RGB')

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

def concordance_index(y_true, y_pred):
    """
    Calculate concordance index.
    
    Parameters:
        y_true (numpy.array): True labels.
        y_pred (numpy.array): Predicted labels or scores.
    
    Returns:
        float: Concordance index.
    """
    n = len(y_true)
    total = 0
    pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[j] > y_true[i]:
                pairs += 1
                if y_pred[j] > y_pred[i]:
                    total += 1
                elif y_pred[j] == y_pred[i]:
                    total += 0.5

    if pairs == 0:
        return 0.0

    return total / pairs


def valid_epoch(net, valid_loader, criterion):
    net.eval()
    val_loss = 0
    correct_val_predictions = 0
    total_valid = 0
    df1=pd.DataFrame()
    val_f1_scores = []  # To store F1 scores
    val_predictions_all = []
    val_labels_all = []

    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            val_inputs, val_labels, patientsID = data

            val_inputs,val_labels = val_inputs.to(device),val_labels.to(device)

            # No need to convert labels to torch.tensor
            outputs = net(val_inputs)

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

            # Collect predictions and labels for calculating precision, recall, and AUC
            val_predictions_all.append(val_predictions.cpu().numpy())
            val_labels_all.append(val_labels.cpu().numpy())

    avg_val_loss = val_loss / len(valid_loader)
    avg_val_acc = correct_val_predictions / total_valid * 100
    avg_val_f1 = sum(val_f1_scores) / len(val_f1_scores)

    # Concatenate predictions and labels
    val_predictions_all = np.concatenate(val_predictions_all)
    val_labels_all = np.concatenate(val_labels_all)

    # Calculate precision, recall, and AUC
    precision = precision_score(val_labels_all, val_predictions_all)
    recall = recall_score(val_labels_all, val_predictions_all)
    auc = roc_auc_score(val_labels_all, val_predictions_all)
    c_index = concordance_index(np.squeeze(val_labels_all), np.squeeze(val_predictions_all))

    return avg_val_loss, avg_val_acc, avg_val_f1, precision, recall, auc, df1,c_index


def test_epoch(net, valid_loader, criterion):
    net.eval()

    survival_df=pd.DataFrame()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            test_inputs, test_labels, patientsID= data

            test_inputs,test_labels = test_inputs.to(device),test_labels.to(device)


            # No need to convert labels to torch.tensor
            outputs = net(test_inputs)

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

    k = 4
    # Initialize GroupKFold object
    stratified_group_kfold = StratifiedGroupKFold(n_splits=k)
    num_epochs= 50


    X = dataset.drop(columns=['PatientID'])  # Assuming 'patient_id' is your patient ID column
    y = dataset['365']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.0381, 0.0381, 0.0381],[0.1041, 0.1041, 0.1041]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-40, 40)),
        ])
    test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize([0.0381, 0.0381, 0.0381],[0.1041, 0.1041, 0.1041]),
       ])
       
    learning_rates = [0.01,0.001,0.0001]

    # Directory to save the best performing models
    model_save_dir = "/root/code/thesis/codeFolder/LatestDataInUse/results/models/Resnet/CT2Sequences_numpy_tumor/Image"
    os.makedirs(model_save_dir, exist_ok=True)  # Ensure directory exists



    results={}
    curve_save_dir = "/root/code/thesis/codeFolder/LatestDataInUse/results/curves/Resnet/CT2Sequences_numpy_tumor/Image"
    os.makedirs(curve_save_dir, exist_ok=True)  

    for lr in learning_rates:
        
        print(f"Training with learning rate: {lr}")
        
        #history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'PatientID': [], 'learningRate':[], 'K':[]}
        kfold_train_acc = []
        kfold_test_acc = []
        kfold_precision= []
        kfold_recall = []
        kfold_auc = []
        kfold_f1_score = []
        kfold_c_index_score= []

        kfold_number=1
        best_accuracy= 0
        best_model_state_dict = None  # Initialize best model state dictionary
        best_model_fold_number = None  # Initialize best model fold number
        best_learning_rate = None 
        


        for train_index, test_index in stratified_group_kfold.split(X, y, groups=dataset['PatientID']):

            
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

            model = resnet50(pretrained=False)

            # Get the number of input features for the classifier
            n_inputs = model.fc.in_features

            # Modify the classifier of ResNet50
            model.fc = nn.Sequential(
                nn.Linear(n_inputs, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, 1)  # Assuming it's a binary classification
            )

            # Transfer the model to the appropriate device (CPU/GPU)
            model = model.to(device)


            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            criterion = nn.BCEWithLogitsLoss()

            history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'PatientID': [], 'learningRate':[], 'K':[], 'precision':[],'recall':[],'auc':[]}

            for epoch in range(num_epochs):
                epoch_train_accuracy = []
                epoch_test_accuracy = []
                epoch_precision = []
                epoch_recall = []
                epoch_auc = []
                epoch_f1_score = []
                epoch_c_index = []

                train_loss, train_correct = train_epoch(model, train_loader, criterion, optimizer)
                test_loss, test_correct,val_f1_score, precision, recall, auc, predictedDataFrame, c_index_score = valid_epoch(model, test_loader, criterion)
                #test_loss, test_correct,val_f1_score, precision, recall, auc, predictedDataFrame

                train_loss = train_loss
                train_acc = train_correct
                test_loss = test_loss
                test_acc = test_correct
                
                epoch_train_accuracy.append(train_acc)
                epoch_test_accuracy.append(test_acc)
                epoch_precision.append(precision)
                epoch_recall.append(recall)
                epoch_auc.append(auc)
                epoch_f1_score.append(val_f1_score)
                epoch_c_index.append(c_index_score)


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
                history['precision'].append(precision)
                history['recall'].append(recall)
                history['auc'].append(auc)



            kfold_train_acc.append(np.max(epoch_train_accuracy))
            kfold_test_acc.append(np.max(epoch_test_accuracy))
            kfold_precision.append(np.max(epoch_precision))
            kfold_recall.append(np.max(epoch_recall))
            kfold_auc.append(np.max(epoch_auc))
            kfold_f1_score.append(np.max(epoch_f1_score))
            kfold_c_index_score.append(np.max(epoch_c_index))

            # Save training and validation curves
            save_training_curves(history, curve_save_dir, lr, kfold_number)

            kfold_number=kfold_number+1
    

        results[lr] = {
            'train_acc': sum(kfold_train_acc) / len(kfold_train_acc),
            'test_acc': sum(kfold_test_acc) / len(kfold_test_acc),
            'min_test_acc' : np.min(kfold_test_acc),
            'max_test_acc' : np.max(kfold_test_acc),

            
            'precision': sum(kfold_precision) / len(kfold_precision),
            'min_precision' : np.min(kfold_precision),
            'max_precision' : np.max(kfold_precision),

            'recall': sum(kfold_recall) / len(kfold_recall),
            'min_recall' : np.min(kfold_recall),
            'max_recall' : np.max(kfold_recall),

            'auc': sum(kfold_auc) / len(kfold_auc),
            'min_auc' : np.min(kfold_auc),
            'max_auc' : np.max(kfold_auc),

            'f1_score': sum(kfold_f1_score) / len(kfold_f1_score),
            'min_f1_score' : np.min(kfold_f1_score),
            'max_f1_score' : np.max(kfold_f1_score),

            'c1_score': sum(kfold_c_index_score) / len(kfold_c_index_score),
            'min_c1_score' : np.min(kfold_c_index_score),
            'max_c1_score' : np.max(kfold_c_index_score),
        }

        save_model(best_model_state_dict, best_learning_rate, best_model_fold_number, model_save_dir)


        # best_model = resnet50(pretrained=False)

        # # Get the number of input features for the classifier
        # n_inputs = best_model.fc.in_features

        # # Modify the classifier of ResNet50
        # best_model.fc = nn.Sequential(
        #     nn.Linear(n_inputs, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(64, 1)  # Assuming it's a binary classification
        # )
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
        survival_results_df.to_csv(f'/root/code/thesis/codeFolder/LatestDataInUse/results/survival/ResNet50/CT2Sequences_numpy_tumor/Image/survival_results_lr_{best_learning_rate}.csv', index=False)
        # Save classification report to file

        # Save classification report to file
        with open(f'/root/code/thesis/codeFolder/LatestDataInUse/results/reports/Resnet/CT2Sequences_numpy_tumor/Image/classification_report_lr_{best_learning_rate}.txt', 'w') as f:
            f.write(report_classification)



                    
    # File to save results
    result_file = "/root/code/thesis/codeFolder/LatestDataInUse/results/CT2Sequences_numpy_tumor_ImageResnet50.txt"

    # Open file for writing
    with open(result_file, 'w') as f:
        # Loop over learning rates and results
        for lr, result in results.items():
            # Write learning rate and corresponding results to file
             f.write(f"Learning Rate: {lr}, Train Accuracy: {result['train_acc']}, Test Accuracy: {result['test_acc']}, Minimum Test Accuracy: {result['min_test_acc']}, Maximum Test Accuracy: {result['max_test_acc']},"
                f"Average Precision: {result['precision']}, Minimum Precision: {result['min_precision']}, Maximum Precision: {result['max_precision']},"
                f"Average Recall: {result['recall']}, Minimum Recall: {result['min_recall']}, Maximum Recall: {result['max_recall']},"
                f"Average AUC: {result['precision']}, Minimum AUC: {result['min_auc']}, Maximum AUC: {result['max_auc']},"
                f"Average F1 score: {result['f1_score']}, Minimum F1 score: {result['min_f1_score']}, Maximum F1 score: {result['max_f1_score']},"
                f"Average C1 score: {result['c1_score']}, Minimum C1 score: {result['min_c1_score']}, Maximum C1 score: {result['max_c1_score']}\n")




