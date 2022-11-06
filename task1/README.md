# Task1 - Property price recommendation
This folder contains data and code for Singapore property price recommendation.

## EDA
The graphs of our EDA is shown in **EDA.ipynb**

## Data preprocessing
### Code
We processe the train data and test data separately. The corresponding codes are **preprocess_train.ipynb** and **preprocess_test.ipynb** . 
**utils.py** is required for data preprocessing which contains all the necessary functions.
### Data file
The data files are saved in the [data](https://github.com/largecats/ghostbusters/tree/main/task1/data) folder. **train_final_complete.csv** is the cleaned file for training.
**test_final_complete_cleaned.csv** contains data for final prediction. **train.csv** and **test.csv** are raw data files with manually filled geographical information.

## Models
We experiment with models for price prediction<br>
- **task1_knn&linear_model.ipynb**</br>
This notebook contains code for KNN regressor and linear regressor
- **task1_svr.ipynb**</br>
This notebook contains code for Support vectore regressor
- **task1 - tree ensembles.ipynb**</br>
This notebook contains code for tree regressors: gradient boosting, random forest and adaboost
- **task1-mlp.py**</br>
This contains code for multilayer perceptron. To run this, **mlp.py** is required, which contains model structure and dataset function.<br>
To train the model:
```bash
python task1-mlp.py --train_dataset_path <TRAIN_SET_PATH> --save_model_dir <SAVE_MODEL_FOLDER>
```
To make prediction:
```bash
python task1-mlp.py --is_test true --test_dataset_path <TEST_SET_PATH> --best_model_path <BEST_MODEL_PATH>
```
## Models
- **task1 - evaluation.ipynb**</br>
This notebook contains evaluation and comparison of different model on per_price segments.
