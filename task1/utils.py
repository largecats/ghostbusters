from tabnanny import check
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopy.distance # to compute distance in auxiliary data
import pylab as pl
import math

"""
PROPERTY_TYPE
"""
##  Generalize property types into 3 class hdb condo landed
def generalize_property_type(v):
    v = v.lower()
    if ('hdb' in v):
        return 'hdb'
    if ('condo' in v) or ('apartment' in v) or ('walk' in v) or ('cluster' in v):
        return 'condo'
    return 'landed'

##  Transform all property type into lower case, generalize hdb and bungalows
def standardize_property_type(v):
    v = v.lower()
    if ('hdb' in v):
        return 'hdb'
    elif ('bungalow' in v):
        return 'bungalow'
    else:
        return v

"""SIZE_SQFT"""
##  Transform small size_sqft from size square meter to size square foot
def standardize_size(v,df,minthres,maxthres,ppt_type):
    if v.general_property_type != ppt_type:
        return v.size_sqft
    elif v.size_sqft<minthres:
        return check_sqm_to_sqft(v,df)
    elif v.size_sqft >maxthres:
        return np.nan
    else:
        return v.size_sqft

#   Some data are shown in square meter but not square feet
#   Check if there is house in the same property with same sqft after transformation
def check_sqm_to_sqft(v,df):
    size_unq = df[df.property_name==v.property_name].size_sqft.unique()
    sizesqft = int(v.size_sqft*10.7639)
    if np.sum(np.isin(size_unq,sizesqft)):
        return sizesqft
    else:
        return np.nan


"""TENURE"""
##  Group tenure based on years
def standardize_tenure_text(v):
    if pd.isna(v):
        return 'NaN'
    elif v=="freehold":
        return 'freehold'
    elif v =='100-year leasehold' or v =='99-year leasehold' or v =='102-year leasehold' or v =='103-year leasehold' or v == '110-year leasehold':
        return 'around 99-year'
    elif  ('999-year leasehold' in v) or  ('929-year leasehold' in v) or  ('956-year leasehold' in v) or  ('946-year leasehold' in v) or  ('947-year leasehold' in v):
        return 'around 999-year'

##  Group tenure based on years
def standardize_tenure(v):
    if pd.isna(v):
        return 0
    elif v=="freehold":
        return 3
    elif v =='100-year leasehold' or v =='99-year leasehold' or v =='102-year leasehold' or v =='103-year leasehold' or v == '110-year leasehold':
        return 1
    elif  ('999-year leasehold' in v) or  ('929-year leasehold' in v) or  ('956-year leasehold' in v) or  ('946-year leasehold' in v) or  ('947-year leasehold' in v):
        return 2



"""Num beds and Num baths"""
##  Try filling noisy data with information of the same property
def standardize_bednbath(v,df,minthres,maxthres):
    sameppt = df[df.property_name==v.property_name]
    sameppt = sameppt[(sameppt.bed2bath<maxthres) & (sameppt.bed2bath>minthres)]
    if v.bed2bath<minthres or v.bed2bath>maxthres:
        nbed,nbath = get_same_entries_information(v,sameppt)
        return nbed,nbath
    else:
        return v.num_beds,v.num_baths

##  Edit noisy floor plan with  data with the same size_sqft of the same property
def get_same_entries_information(v,df):
    target = df[df.size_sqft == v.size_sqft]
    if target.shape[0] == 0:
        return np.nan, np.nan
    return pd.Series.mode(target.num_beds)[0], pd.Series.mode(target.num_baths)[0]



"""AUXILIARY DATAS"""
# compute distance between to points given their latitude and longitude
def get_distance(mrt_lat, mrt_lng, lat, lng):
    return geopy.distance.distance((lat, lng), (mrt_lat, mrt_lng))

# compute distance to nearest facility (mrt)
def get_distance_to_nearest_facility(df_facilities, row):
    lat, lng = row.lat, row.lng
    distances = df_facilities.apply(lambda r: get_distance(r.lat, r.lng, lat, lng).km, axis=1) # distance in km
    min_dist = distances.min()
    return min_dist

# compute number of facilities (schools, shopping malls) in given radius
def get_number_of_nearby_facilities(df_facilities, radius, row):
    lat, lng = row.lat, row.lng
    distances = df_facilities.apply(lambda r: get_distance(r.lat, r.lng, lat, lng).km, axis=1) # distance in km
    return len(distances[distances <= radius])

"""
Nearest MRT
"""
def add_distance_to_nearest_mrt(df_mrt, df, col_name='dist_to_nearest_mrt'):
    df[col_name] = df.apply(lambda row: get_distance_to_nearest_facility(df_mrt, row), axis=1)
    return df

"""
Near Shopping Mall
"""
def add_number_of_nearby_shopping_malls(df_shopping_malls, df):
    df['number_of_nearby_shopping_malls'] = df.apply(lambda row: get_number_of_nearby_facilities(df_shopping_malls, 0.3, row), axis=1) # in 0.3km
    return df


"""
Near School
"""
def add_number_of_nearby_primary_schools(df_primary_schools, df):
    df['number_of_nearby_primary_schools'] = df.apply(lambda row: get_number_of_nearby_facilities(df_primary_schools, 1., row), axis=1) # in 1km
    return df

def add_number_of_nearby_secondary_schools(df_secondary_schools, df):
    df['number_of_nearby_secondary_schools'] = df.apply(lambda row: get_number_of_nearby_facilities(df_secondary_schools, 1., row), axis=1) # in 1km
    return df

"""
Commercial Centre
"""
def get_name_of_nearest_commercial_centre(row,df_cct,thres):
    lat, lng = row.lat, row.lng
    df_cct['dists'] = df_cct.apply(lambda r: get_distance(r.lat, r.lng, lat, lng).km, axis=1)
    if np.amin(df_cct['dists']) > thres:
        return 'None'
    else:
        return df_cct[df_cct['dists']==df_cct['dists'].min()]['name'].item()

def add_name_of_nearest_commercial_centre_by_type(df,df_cc,cc_type,thres):
    df_cct = df_cc[df_cc['type']==cc_type]
    df['name_of_nearest_'+cc_type] = df.apply(lambda r: get_name_of_nearest_commercial_centre(r,df_cct,thres),axis = 1)
    return df

"""
Methods for finding outliers
"""
# Find outlier with n sigma
def outlier_nsigma(df,colName):#input data column
    dc = df[colName]
    nsigma = 3
    upthres = dc.mean()+nsigma*dc.std()
    lowthres = dc.mean()-nsigma*dc.std()
    cri = (dc>upthres)|(dc<lowthres)
    idx = df[cri].index
    return idx

# Find outlier with percentage
def outlier_percentage(df,colName, pthres):
    # Upper and lower limits by percentiles
    dc = df[colName]
    upthres = dc.quantile(1-pthres)
    lowthres = dc.quantile(pthres)
    cri1 = (dc>upthres)|(dc<lowthres)
    idx = df[cri1].index
    return idx
  
# Find outlier with iqr
def outlier_iqr(df,colName):
    dc = df[colName]
    q3 = dc.quantile(0.75)
    q1 = dc.quantile(0.25)
    iqr_range = q3 - q1
    # Obtaining the lower and upper bound
    upthres = q3 + 1.5 * iqr_range
    lowthres = q1 - 1.5 * iqr_range
    cri1 = (dc>upthres)|(dc<lowthres)
    idx = df[cri1].index
    return idx

### Find outlier combining different mthods ###
def get_outlier(data,atype):
    noise_std = outlier_nsigma(data,atype)
    noise_pct = outlier_percentage(data,atype,0.01)
    noise_iqr = outlier_iqr(data,atype)
    mask = np.isin(noise_iqr,noise_std)
    final_noise = noise_iqr[mask]
    mask2 = np.isin(final_noise,noise_pct)
    final_noise = final_noise[mask2]
    return final_noise

"""
TEST CLEAINING
"""
def set_encoded_with_csv(v,df,ecol,tcol):
    d = df[df[tcol] == v[tcol]]
    if d.shape[0] == 0:
        d = df[(df['property_type'] == v['property_type'])&(df['subzone'] == v['subzone'])]
    return d[ecol].median()