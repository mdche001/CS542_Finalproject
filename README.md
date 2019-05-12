# EC542_Finalproject

## Project Introduction

This is a Kaggle Competetion project
Link: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/overview

## Model diagram
![image](https://github.com/mdche001/EC542_Finalproject/blob/master/Image/Blank%20Diagram.png)


## Getting Started


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



### Break down into end to end tests

## Data Process

### Data visualization

The Data visualization work in the DataVisualization.ipynb. This notebook include the analysis graph.

### Data processing

In the FeatureCreation.py file, the new features construction is ccompleted. Provided more features for the model training.


## Running the model

 To get the XGBoost model, first run the XGBoost.py file, then u will get a model named xgboost.model, which is only 2MB. With the model, you can run the testxgboost.py, then we can get the final result xgb_subnew file, which is the version we got the following result on kaggle.
 
![image](https://github.com/mdche001/EC542_Finalproject/blob/master/Image/xgboost.png)

## Authors

* **Mingdao Che** - *Initial work* - [PurpleBooth](https://github.com/mdche001/EC542_Finalproject)
* **Wanxuan Chen** - *Initial work*
* **Zhuyun Cheng** - *Initial work*
