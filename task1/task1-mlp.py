
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import time
import pickle
import argparse
import numpy as np
import random
from tqdm import tqdm

import pandas as pd
from sklearn.preprocessing import StandardScaler

from mlp import *

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def per_price_to_whole(pred,size_sqft):
    print(pred[:,0,0].shape)
    print(size_sqft.shape)
    price = pred[:,0,0] * size_sqft.to_numpy()
    return price

def price_to_submission(pred,save_path):
    df = pd.DataFrame()
    df['Predicted'] = pred
    df.to_csv(save_path,index_label ='Id')

def prepare_dataset(args,house_data,train_ratio):
    num_train = int(house_data.shape[0]*train_ratio)
    idxs = np.arange(len(house_data))
    np.random.shuffle(idxs)
    train_data = house_data[idxs[:num_train]]
    valid_data = house_data[idxs[num_train:]]

    train_dataset=houseDataset(train_data)
    train_loader=DataLoader(train_dataset,batch_size=int(args.batch_size),shuffle=False)

    valid_dataset=houseDataset(valid_data)
    valid_loader=DataLoader(valid_dataset,batch_size=int(args.batch_size),shuffle=False)

    return train_loader, valid_loader

def predict(args,model,test_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.best_model_path))
    model.to(device)
    model.eval()
    prices = []
    test_dataset = houseTestDataset(test_data)
    test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)
    with torch.no_grad():
        for step, data in enumerate(tqdm(test_loader)):
            input_tensor = data.to(device)
            pred = model(input_tensor).detach().cpu().numpy()
            prices.append(pred)
    res = np.asarray(prices)
    return res
        

def train(model, args,trainset_loader,validset_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = 1e-4)
    criterion = nn.MSELoss()

    start_time = time.time()
    best_model_id = -1
    min_valid_loss =  float("inf")
    epoch_num = int(args.epoch_num)
    valid_every_k_epoch = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(args.best_model_path) and args.load_model:
        print('Load model at:', args.best_model_path)
        model.load_state_dict(torch.load(args.best_model_path))
        
    model.to(device)
    train_losses = []
    valid_losses = []
    for epoch in range(epoch_num):
        model.train()
        epoch_train_loss = 0
        for step, (data,labels) in enumerate(tqdm(trainset_loader)):
            input_tensor = data.to(device)
            label = labels.float().to(device)

            # Zero the gradients
            optimizer.zero_grad()

            logits = model(input_tensor)

            loss = criterion(logits,label)
            loss.backward()
            optimizer.step()

            epoch_train_loss+=loss.detach().item()

        avg_train_loss = epoch_train_loss / len(trainset_loader)
        print('\n', 'Epoch '+str(epoch)+'train loss : ' , str(avg_train_loss))
        train_losses.append(avg_train_loss)

        if(epoch+1)% valid_every_k_epoch == 0:
            epoch_valid_loss = 0
            model.eval()
            with torch.no_grad():
                for vstep, (vdata,vlabels) in enumerate(tqdm(validset_loader)):
                    vinput = vdata.to(device)
                    vlabel = vlabels.to(device)

                    vlogits = model(vinput)

                    vloss = criterion(vlogits,vlabel)
                    epoch_valid_loss +=vloss.item()

            avg_val_loss = epoch_valid_loss/len(validset_loader)
            print('\n', 'Epoch ',  epoch , ' Val loss : ' , avg_val_loss)
            valid_losses.append(avg_val_loss)

            if avg_val_loss < min_valid_loss:
                min_valid_loss = avg_val_loss
                best_model_id = epoch
                torch.save(model.state_dict(), args.save_model_dir + '/model'+'.pth')
                print('\n', 'Best Epoch ', str(epoch))  
            
    fig = plt.figure()
    x1 = np.arange(epoch_num)
    x2 = np.arange(epoch_num/valid_every_k_epoch)*valid_every_k_epoch+valid_every_k_epoch
    plt.plot(x1,train_losses,label="train loss")
    plt.plot(x2,valid_losses,label="valid loss")
    plt.legend()
    fig.savefig('loss_graph')

    print('\n', 'Best Epoch ', str(best_model_id),'\n','Min Loss: ', str(min_valid_loss))  
    


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_path', default='/home/k/kzheng3/5228/dataprocess/train_final_complete.csv', help='path to the train set')
    parser.add_argument('--test_dataset_path', default='/home/k/kzheng3/5228/dataprocess/test_final_complete_cleaned.csv', help='path to the train set')
    parser.add_argument('--save_model_dir', default='./results', help='path to save the trained models')
    parser.add_argument('--batch_size', default=64, help='batch size for train and validation')
    parser.add_argument('--epoch_num', default=1500, help='number of train epochs')
    parser.add_argument('--device', default='cuda', help='cuda device')
    parser.add_argument('--learning_rate', default=1e-3, help='learning rate')
    parser.add_argument('--best_model_path', default='./results/model.pth')
    parser.add_argument('--load_model', default=False)
    parser.add_argument('--is_test', default=False)

    args = parser.parse_args()
    set_seeds(20) # reproduce

    scaler = StandardScaler()
    feature_idx = [
        'built_year', 'num_beds', 'num_baths', 'lat', 'lng', 'size_sqft',
        'tenure_group', 'subzone_per_price_encoded',
        'property_type_ordinal',
        #mrt
        'dist_to_nearest_important_mrt_rounded',
        #schools
        'number_of_nearby_primary_schools', 
        'number_of_nearby_secondary_schools', 
        #shopping mall
        'number_of_nearby_shopping_malls',
        #CR
        'name_of_nearest_BN_ordinal',
        'name_of_nearest_CR_ordinal'
    ]
    house_data = pd.read_csv(args.train_dataset_path)
    feature_data = house_data[feature_idx]
    scaler.fit(feature_data)

    if args.is_test:
        house_data = pd.read_csv(args.test_dataset_path)
        feature_data = house_data[feature_idx]

        normalized_feature = scaler.transform(feature_data)

        model = BaseNN(len(feature_idx))
        pred = predict(args,model,normalized_feature)
        price = per_price_to_whole(pred,house_data.size_sqft)
        print(price)
        price_to_submission(price,'./results/test_prediction.csv')
        
        
    else:
        target_idx = ['per_price']

        price_data = house_data[target_idx].to_numpy()

        #Normalize Feature
        normalized_feature = scaler.transform(feature_data)
        input_data = np.concatenate((normalized_feature, price_data), axis=1)

        #Dataset loader
        train_loader,valid_loader = prepare_dataset(args,input_data,0.8)

        if not os.path.exists(args.save_model_dir):
            os.mkdir(args.save_model_dir)
        
        model = BaseNN(len(feature_idx))
        train(model,args,train_loader,valid_loader)





    