# Master-thesis
## "Survival Rate Prediction using diagnostic images and immunoprofiles of Cancer Patients"

In this Master thesis work,  Multimodal Neural Netoworks were developed using immunoprofiles, MRI scans and CT scans for predicting the survival rate of liver cancer patient for one year.

The Immunoprofiles of the Liver cancer patients were combined with CT and MRI dataset to measure the increase or decrease in performance compared to image only models using MRI or CT dataset.

Three different Multimodal Neural Networks were developed. Immunoprofiles and CT Multimodal Neural Network, Immunoprofiles and MRI Multimodal Neural Network, Immunoprofiles, CT and MRI Multimodal Neural Network.

The designed multimodal neural network consists of ResNet-50 model for extracting the image features,feature embedding layer to extend the immunoprofile dataset to extracted image feature size, data fusion layer to concatenate the image and immunoprofiles feature together and a fully connected linear layer with one output unit for predicting one year survival prediction

Preprocessing folder contains the code used for preapring the data for the model.

Model folder contained the designed Multimodal neural Netowrks.

Results folder contains the evaluation metrics that were recorded after during training and testing of the model.
