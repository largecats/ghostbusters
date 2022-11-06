from dataclasses import replace
from select import select
from tabnanny import check
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopy.distance # to compute distance in auxiliary data
import pylab as pl
import math
import random

# compute distance between to points given their latitude and longitude
def get_distance(mrt_lat, mrt_lng, lat, lng):
    return geopy.distance.distance((lat, lng), (mrt_lat, mrt_lng))

# compute distance to nearest facility (mrt)
def get_distance_to_nearest_facility(df_facilities, row):
    lat, lng = row.lat, row.lng
    distances = df_facilities.apply(lambda r: get_distance(r.lat, r.lng, lat, lng).km, axis=1) # distance in km
    min_dist = distances.min()
    return min_dist
# Get nearby mrt name and dist
def get_name_and_dist_of_nearest_mrt(row,df_cct):
    lat, lng = row.lat, row.lng
    df_cct['dists'] = df_cct.apply(lambda r: get_distance(r.lat, r.lng, lat, lng).km, axis=1)
    if len(df_cct[df_cct['dists']==df_cct['dists'].min()]['name']) > 1:
        return df_cct[df_cct['dists']==df_cct['dists'].min()]['name'].iloc[0], df_cct['dists'].min()
    return df_cct[df_cct['dists']==df_cct['dists'].min()]['name'].item(), df_cct['dists'].min()

# mrt information to columns
def add_info_of_nearest_mrt(df,df_cc):
    res = df.apply(lambda r: get_name_and_dist_of_nearest_mrt(r,df_cc),axis = 1)
    res = np.array([*res])
    df['name_of_nearest_mrt'] = res[:,0]
    df['dist_to_nearest_mrt'] = res[:,1]
    return df

# Randomly choose attributes as search criterion
def get_search_features(feats,mandatory = False,single = False):
    num = random.randint(1,len(feats))
    if mandatory:
        selected = random.sample(list(feats), k = num)
    elif single:
        selected = np.random.choice(feats, num, replace=False, p=[0.4, 0.4, 0.2])
    else:
        selected = np.random.choice(feats, num, replace=False, p=[0.04,0.16,0.16,0.16,0.16,0.16,0.16])
    if np.any(selected=='None'):
        selected = []
    return selected

# Randomly choose minimum threshold and maximum threshold for range-based attributed
def define_range(feat):
    choices = np.random.choice([True, True, False],2,replace = False)
    has_min = choices[0]
    has_max = choices[1]
    if not has_min:
        has_max = True
    min_thres, max_thres = 0,0
    if has_min:
        min_thres = min(random.randint(int(np.nanpercentile(feat,1)),int(np.nanpercentile(feat,90))),random.randint(int(np.nanpercentile(feat,10)),int(np.nanpercentile(feat,90))),random.randint(int(np.nanpercentile(feat,10)),int(np.nanpercentile(feat,90))))
        if has_max:
            max_thres = min(random.randint(min_thres,int(np.amax(feat))),random.randint(min_thres,int(np.amax(feat))),random.randint(min_thres,int(np.amax(feat))))
    elif has_max:
        max_thres = min(random.randint(min_thres,int(np.amax(feat))),random.randint(min_thres,int(np.amax(feat))),random.randint(min_thres,int(np.amax(feat))))
    if min_thres > max_thres:
        max_thres = np.amax(feat) + 1
    return min_thres,max_thres

# Randomly choose multiple close numbers
def define_close_num(feat):
    center = np.random.choice(feat)
    selected = [center,np.random.choice([center+1,0], 1)[0],np.random.choice([center-1,0], 1)[0],np.random.choice([center-2,0], 1)[0],np.random.choice([center+2,0], 1)[0]]
    selected = [i for i in selected if (i > 0) and (i<=np.amax(feat))]
    return selected

# Randomly choose multiple values without constraints
def define_select_multiple(feat):
    num = random.randint(1,3)
    selected = np.random.choice(feat,num,replace=False)
    list_set = set(selected)
    selected = (list(list_set))
    return selected


