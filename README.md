# Welcome to FCS1 Team 1's Chest X-Ray Repository

### 📃 About

This is the repository for FCS1 Team 1's mini project. Our project focused on image recognition and classification, specifically diagnosis of chest diseases given an x-ray

### 🗂 Repo Organisation
- Archived Codes: Collection of jupyter notebooks
- Original Dataset: The files extracted from Kaggle. Includes:
  - Sample chest x-rays
  - .csv files
- FCS1_team1_KoongAngJenniskens.ipynb: Our final jupyter notebook, containing all code and models used in our project
- Results.csv: each CNN model's metrics over epochs
- Accuracy.csv: each CNN model's accuracy metrics


### 👨‍💻 Contributers
- @corneliusacf: Insights and recommendations
- @harryjenniskens: Data cleaning and analysis
- @jonkoong: Machine learning techniques

![image](https://github.com/harryjenniskens/sc1015MiniProject/assets/167991732/6ef9965a-8931-4ef8-86a9-cef6ab407fa5)

### 📚 Dataset
NIH Chest X-rays from [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) 
Over 110,000 chest x-rays from over 30,000 unique patients

### 🧐 Problem definition
The current average Chest X-Ray Diagnosis Accuracy of a radiologist is 94%. How can we use machine learning to aid doctors and enhance that accuracy?

### 📊 Models used
| Machine Learning Model                  | Accuracy On Test Set |
|-----------------------------------------|----------|
| 3 Layer Convolutional Neural Network    | 54%      |
| 4 Layer CNN                             | 53%      |
| ResNet-50                               | 53%      |
| ResNet-50 with weights                  | 58%      |
| 3 Layer CNN (Single Classification)     | 65%      |


### 📈 Conclusion & what we learned from this project
- We learned the importance of understanding our dataset and how to better identify certain challenges it may present.
- We used a pre-trained model that took 224x224 sized images, while our images were 512x512, meaning we lost nearly 81% of the available visual information when using the ResNet-50. Next time we should find and use a model that could cater to our image size.
- Our dataset also came with about a thousand images that contained bounding boxes on visual indicators of diseases, but we did not implement this in our models due to complexity.
- We also chose not to incorporate the patient's report and medical history into our prediction model. This and the bounding boxes could have potentially led to drastically more accurate results for our model, and is something we will look into incorporating for future projects.

### 📝 Summary
- Each chest x-ray has 15 potential labels, 14 for diseases and 1 "no finding" label. Patients can have any combination of the 15 labels, meaning they can have 1-15 different labels or diagnosis. 

- We have trained and used 5 different models.
    1. We initially trained our own simple 3-layer convolutional neural network (CNN) with 32, 64, 64 filters per layer respectively, however only achieved 54% accuracy on the validation set.
    2. In our attempt to raise the accuracy, we then increased our convolutional layers to 4 and double the filters per layer. However, our accuracy dropped to 53%, which we suspected was due to the epoch or training cycle (1st model had 5, the 2nd only 3).
    3. We then decided to switch gears and use the ResNet-50, which is a pre-trained model with 48 convolutional layers. However, our accuracy was still at around 53%. Since we were experiencing similar accuracies with different models, we took another look at our dataset.
      - We discovered that we had significant class imbalances across our dataset, which could introduce bias into our model and reduce its accuracy.
    4. We attempted to solve this issue with class specific weights, and retrained our ResNet-50 model, along with some other tweaks. This yieled an accuracy of 57.66%.
    5. Finally, to experiment, we only used single label x-rays which increased our accuracy to 65%. This was expected as the task is much easier.
 
  ![image](https://github.com/harryjenniskens/sc1015MiniProject/assets/167991732/17988147-e5ff-49f7-b172-017063946aba) ![image](https://github.com/harryjenniskens/sc1015MiniProject/assets/167991732/36c5f647-9a7d-44e4-aaf9-19c60f97b68a)



### 🗄️ References
- https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345
- https://realpython.com/python-ai-neural-network/
- https://www.mdpi.com/2072-6694/15/13/3267
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9759648/
- https://api.isic-archive.com/collections/?pinned=true
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8382232/
- https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
- https://www.mdpi.com/2072-6694/15/13/3267
- https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-022-00793-7
