# Task2 - Property recommendation
This folder contains data and code for Singapore property price recommendation.

## Data preprocessing
We further processed cleaned data of task1 for task 2, and generate pseudo user profile with random search criterion. 
The corresponding codes is **gen_user_profile.ipynb** . **utils_task2.py** contains functions for data preprocessing

### Data file
The data used for recommendation is **df_task2_onehot.csv**. For preprocessing, we utilize **train_final_complete_nodrop.csv**, **train.csv**  and **sg-mrt-stations.csv** in task 1 data folder.

## Models
**task2.ipynb** contains code for recommendation. **property_recommendation.py** contains corresponding functions.
