import pickle
from sklearn.metrics import (confusion_matrix,
     classification_report)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os


root = "./data/"

def create_data_feature_steps(data, steps=3):
    new_df = []
    for i in range(steps, data.shape[0]):
        new_df = np.append(new_df,
                           data[i-steps: steps+i])
    new_df_cols = ['col'+str(i) for i in range(steps)]
    new_df = pd.DataFrame(new_df.reshape(-1,steps),
                 columns=new_df_cols)
    return new_df

def preprocess_data(data, steps=3):
    data = data.drop('time', axis=1)
    data = StandardScaler().fit_transform(data)
    data = create_data_feature_steps(data, steps=steps)
    return data

def read_multiple_files(path=root, attr='filtered', shuffle=True):
    print('init reading files')
    cols = ['time','value']
    
    all_files = os.listdir(path)
    
    data = pd.DataFrame()
    
    for f in tqdm(all_files):
        if f.split('.')[1]==attr:
            
            # reading csv file
            df = pd.read_csv(os.path.join(path, f),
                             names=cols, header=0)
            
            # preprocessing data with steps
            df = preprocess_data(df, 3)
            
            # adding labels
            labels = np.array([list(
                                f.split('.')[0]
                                )[1]
                           ]*df.shape[0]).astype(int)
            
            df['label'] = labels
            
            # appending df to main data
            data = data.append(df)
    
    # shuffling
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
            
        
    return data

data = read_multiple_files()

def predict(data_path, pkl_model_path='nb_02.pkl'):
    
    # loading pickle file
    model_pkl = open(pkl_model_path, 'rb')
    saved_model = pickle.load(model_pkl)
    print("Loading model :: ", saved_model)

    # converting into model format
    test = read_multiple_files()
    X_test = test.drop('label', axis=1)
    y_test = test['label']

    # evaluation
    pred = saved_model.predict(X_test)
    print(classification_report(y_test, pred))

predict(root)
