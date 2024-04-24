# Welcome to FCS1 Team 1's Chest X-Ray Repository

### About

This is the repository for FCS1 Team 1's mini project. Our project focused on image recognition and classification, specifically diagnosis of chest diseases given an x-ray

### Contributers
- @corneliusacf: Insights and recommendations
- @harryjenniskens: Data cleaning and analysis
- @jonkoong: Machine learning techniques

![image](https://github.com/harryjenniskens/sc1015MiniProject/assets/167991732/6ef9965a-8931-4ef8-86a9-cef6ab407fa5)

### Dataset
NIH Chest X-rays from [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) 

### Problem definition
The current average Chest X-Ray Diagnosis Accuracy of a radiologist is 94%. How can we use machine learning to aid doctors and enhance that accuracy?

### Models used
- 3 Layer Convolutional Neural Network (CNN)
- 4 Layer CNN
- ResNet-50
- ResNet-50 with weights
- ResNet-50 single classification

### Conclusion & what we learned from this project
- We learned the importance of understanding our dataset and how to better identify certain challenges it may present.
- We used a pre-trained model that took 224x224 sized images, while our images were 512x512, meaning we lost nearly 81% of the available visual information when using the ResNet-50. Next time we should find and use a model that could cater to our image size.
- Our dataset also came with about a thousand images that contained bounding boxes on visual indicators of diseases, but we did not implement this in our models due to complexity.
- We also chose not to incorporate the patient's report and medical history into our prediction model. This and the bounding boxes could have potentially led to drastically more accurate results for our model, and is something we will look into incorporating for future projects.

### Summary
This is team 1 from SC1015 lab group FCS1. This is the github containing all our code and other resources for our mini project. Our group has chosen a dataset comprising of 120000 chest x-ray images and the diseases each is labeled with. There are 15 labels, 14 diseases and 1 "no finding" label. Patients can have any combination of the 15 labels, meaning they can have 1-15 different labels or diagnosis. 

We have trained and used 5 different models. We initially trained our own simple 3-layer convolutional neural network (CNN) with 32, 64, 64 filters per layer respectively, however only achieved 54% accuracy on the validation set. In our attempt to raise the accuracy, we then increased our convolutional layers to 4 and double the filters per layer. However, our accuracy dropped to 53%, which we suspected was due to the epoch or training cycle (1st model had 5, the 2nd only 3).

We then decided to switch gears and use the ResNet-50, which is a pre-trained model with 48 convolutional layers. However, our accuracy was still at around 53%. Since we were experiencing similar accuracies with different models, we took another look at our dataset. We discovered that we had significant class imbalances across our dataset, which could introduce bias into our model and reduce its accuracy. We attempted to solve this issue with class specific weights, and retrained our ResNet-50 model, along with some other tweaks. This yieled an accuracy of 57.66%.

Finally, to experiment, we only used single label x-rays which increased our accuracy to 65%. This was expected as the task is much easier. 

