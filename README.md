# CS542_Finalproject

## Project Introduction

This is a Kaggle Competetion project.<br>
The Data is downloaded from Kaggle, because of the secret protocol, we cannot upload the dataset to thsi repo, but you can get the data from the link if you need.<br>
Link: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/overview<br>

The risk of fraud exists everywhere, but companies that offer advertisements online are especially at risk for click fraud. Click fraud is misleading click data and waste money for advertisers. In this project, the objective is to predict whether a user will download an app after clicking a mobile app ad. TalkingData has provided a huge dataset. For this purpose, we initially extracted more features from the limited raw dataset from TalkingData. Then we implemented different types of models to fit the data and obtain a high accuracy of the prediction based on the new dataset structure. The main training model we choose is LightGBM and XGBoost which are two popular decision tree training library for python. The two reliable library benefits our project as we do not need to debug and focus more on machine learning problems. Finally, the two models figure out the excellent result which is  97% on the test set for LigntGBM and 95% for XGBoost.<br>


## Model diagram
![image](https://github.com/mdche001/EC542_Finalproject/blob/master/Image/Blank%20Diagram.png)

## Data Process
To prepare the project,  the team do the data visualization to acquire a deeper comprehension of the dataset, then the team uses the train_sample as the source. The team tried to split the click time and acquire the total click amount in each day, each hour. The data visualization result state that the click frequency is unbalanced. The day frequency shows click amount increase rapidly after the first day, and that might state that the click fraud. <br>

## LightGBM 
LightGBM is a new GBDT algorithm with GOSS and EFB[1].Gradient-based One-Side Sampling (GOSS) can achieve a good balance between reducing the number of data instances and keeping the accuracy for learned decision trees. Exclusive Feature Bundling (EFB) effectively reduces the number of features. The combination of two methods enables LightGBM with fast training speed and high efficiency, lowe memory usage, compatibility with Large Datasets and parallel learning supported.

## XGBoost
The idea of the algorithm is to constantly add trees, constantly transforming features to grow a tree, and adding a tree each time is actually learning a new function to fit the residual of the last prediction. When we get k trees in training, we need to predict the score of a sample. In fact, according to the characteristics of this sample, we will fall into the corresponding leaf node in each tree. Each leaf node corresponds to a score, and finally only the score corresponding to each tree needs to be added to the predicted value of the sample. 

## Usage
### Prerequisites

Python Version: 3.6.5<br>	
matplotlib.pyplot<br>
gc<br>
seaborn<br>
pandas<br>
Lightgbm<br>
numpy<br>
scipy<br>
xgboost<br>

### Data visualization


The Data visualization work in the DataVisualization.ipynb. This notebook include the analysis graph.
![image](https://github.com/mdche001/EC542_Finalproject/blob/master/Image/Click_time.png)
![image](https://github.com/mdche001/EC542_Finalproject/blob/master/Image/downloadPercent.png)

### Data processing

In the FeatureCreation.py file, the new features construction is ccompleted. Provided more features for the model training.
![image](https://github.com/mdche001/CS542_Finalproject/blob/master/Image/datafeature.png)

### Running the model
#### LightGBM

#### XGBoost
 To get the XGBoost model, first run the XGBoost.py file, then u will get a model named xgboost.model, which is only 2MB. With the model, you can run the testxgboost.py, then we can get the final result xgb_subnew.csv file, which is the version we got the following result on kaggle.
 
![image](https://github.com/mdche001/EC542_Finalproject/blob/master/Image/xgboost.png)



## Authors

* **Mingdao Che** - *Initial work* - [PurpleBooth](https://github.com/mdche001/EC542_Finalproject)
* **Wanxuan Chen** - *Initial work*
* **Zhuyun Cheng** - *Initial work*
