#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
import h3
import pickle


def remove_site_hexs(model_data,site_location_data,hex_col='hexid',res=9,lat_col='latitude',lng_col='longitude',hex_ring=False):
    '''
    Remove the hex where a site is located.
    
    An entire hex ring can also be removed.
    
    Parameters
    ----------
    model_data : dataframe
        Input dataframe from which hexes need to be removed
    site_location : dataframe
        Dataframe containing site locations
    hex_col : str, default 'hexid'
        Identifier column for the model_data
    res : int, default 9
        Resolution on which hexid to be created
    lat_col : str, default 'latitude'
        Identifier column for latitude
    lng_col : str, default 'longitude'
        Identifier column for longitude
    hex_ring : bool, default False
        If true, removes the hex ring
        
    Returns
    -------
    Dataframe
        Returns the dataframe with site hexes removed.
    '''
    
    md = model_data.copy(deep=True)
    sl = site_location_data.copy(deep=True)
    
    sl['hex9'] = sl.apply(lambda x: h3.geo_to_h3(x[lat_col],x[lng_col],resolution=res),axis=1)
    sl['hex_rings'] = sl['hex9'].apply(lambda x: h3.k_ring(x))
    
    hex_rings = sl.explode(column='hex_rings').reset_index(drop=True)
    
    if hex_ring == False:
        md = md[~md[hex_col].isin(hex_rings['hex9'])]
    else:
        md = md[~md[hex_col].isin(hex_rings['hex_rings'])]
    
    return md


def within_buffer(model_data,site_location_data,hex_col='hexid',lat_col='latitude',lng_col='longitude',buffer=2000):
    '''
    Keep the hexes within an aerial buffer of a site location.
    
    Parameters
    ----------
    model_data : dataframe
        Input dataframe from which hexes need to be removed
    site_location : dataframe
        Dataframe containing site locations
    hex_col : str, default 'hexid'
        Identifier column for the model_data
    lat_col : str, default 'latitude'
        Identifier column for latitude
    lng_col : str, default 'longitude'
        Identifier column for longitude
    buffer : int, default 2000
        Aerial buffer distance
        
    Returns
    -------
    Dataframe
        Returns the dataframe with site hexes removed.
    '''
    
    md = model_data.copy(deep=True)
    sl = site_location_data.copy(deep=True)
    
    sl = gpd.GeoDataFrame(site_location_data, geometry=gpd.points_from_xy(site_location_data[lng_col],
                                                                     site_location_data[lat_col]),crs={'init':'epsg:4326'})

    sl = sl.to_crs({'init':'epsg:3857'})
    sl['geometry'] = sl['geometry'].buffer(buffer)
    sl = sl.to_crs({'init':'epsg:4326'})
    
    
    md['lat'] = md[hex_col].apply(lambda x: h3.h3_to_geo(x)[0])
    md['lng'] = md[hex_col].apply(lambda x: h3.h3_to_geo(x)[1])

    md = gpd.GeoDataFrame(md, geometry=gpd.points_from_xy(md.lng,md.lat), crs={'init':'epsg:4326'})
    
    new_md = gpd.sjoin(md,sl)
    new_md = new_md.drop_duplicates(subset=hex_col,keep='first').reset_index(drop=True)
    
    return new_md.drop(['geometry',lat_col,lng_col,'lat','lng','index_right'],axis=1)

    

def tam_regression_model(model_data,dv_col,hex_col='hexid',feats=[],train_test=0.2,cv=2,alphas=[i/100 for i in range(1,11,1)],outlier_limit=0.95):
    '''
    Create the Regression model.
    
    Parameters
    ----------
    model_data : dataframe
        Input dataframe containing the GeoIQ features and hexid
    dv_col : str
        Identifier for the dependent variable column
    hex_col : str, default 'hexid'
        Identifier for the hex column
    feats : list, default []
        List of features to be used in the model
    train_test = float, default 0.2
        Train-test split ratio
    cv : int, default 2
        Cross validation to be used in the model
    alphas : list, default [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        List of alpha values for LassoCV model
    outlier_limit : int,  default 0.95
        Upper range for treating outliers in the dv column
        
    Returns
    -------
    Float
        Selected alpha value by the model
    Float
        Training R2
    Float
        Test R2
    Integer
        Number of features selected in the model
    DataFrame
        DataFrame with the selcted features and their coefficients
    Object
        Model object
    Object
        Scaler object
    Array
        List of features selected in the model
    '''
    
    md = model_data.copy(deep=True)
    
    to_remove = []
    for i in md.drop([hex_col],axis=1).columns.tolist():
        if (md[i].dtype != 'float64') & (md[i].dtype != 'int64'):
            to_remove.append(i)

    if len(feats)==0:
        feats = [x for x in md.drop([hex_col,dv_col],axis=1).columns.tolist() if x not in to_remove]
        
    outlier = md[dv_col].quantile(outlier_limit)
    md[dv_col] = np.where(md[dv_col]>outlier,outlier,md[dv_col])
    
    md['log_dv'] = np.log(md[dv_col]+1)
    
    dist_cols = [x for x in md.columns if 'dist' in x]

    for i in dist_cols:
        if i in md.columns:
            md[i] = md[i].replace(-1,6999)

    X = md[feats]
    y = md['log_dv']
    
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=train_test, random_state = 17)
    
    X_train_copy = X_train.copy()

    scaling_features = StandardScaler()
    scaling_fit = scaling_features.fit(X_train[feats])
    X_train_scaled = scaling_fit.transform(X_train[feats])
    X_train_scaled = pd.DataFrame(X_train_scaled)
    X_train_scaled.columns = X_train[feats].columns
    X_train = X_train_scaled

    X_holdout_scaled = scaling_fit.transform(X_holdout[feats])
    X_holdout_scaled = pd.DataFrame(X_holdout_scaled)
    X_holdout_scaled.columns = X_holdout[feats].columns
    X_holdout = X_holdout_scaled
    

    model = LassoCV(alphas=alphas, cv=cv, random_state=42).fit(X_train, y_train)
    imp = pd.DataFrame(data={'features':X_train.columns,'coef': model.coef_
                  }).sort_values(by='coef',ascending=False)

    return model.alpha_, r2_score(y_train, model.predict(X_train)), r2_score(y_holdout, model.predict(X_holdout)), len(np.round(imp[imp['coef']!=0],5)), np.round(imp[imp['coef']!=0],5), model, scaling_fit, model.feature_names_in_


def predict(prediction_data,model,scaler,features,hex_col='hexid',outlier_limit=0.95):
    '''
    Generate predictions on unseen hexes.
    
    Parameters
    ----------
    prediction_data : dataframe
        Input dataframe for which predictions need to be generated
    model : object
        Trained model object
    scaler : object
        Scaler object
    features : list
        List of features used in model training
    hex_col : str, default 'hexid'
        Identifier for the hex column
    outlier_limit : float, default 0.95
        Upper range for treating outliers in the predicted column
        
    Returns
    -------
    DataFrame
        Returns the dataframe with hexid and predictions
    '''
    
    to_predict = prediction_data.copy(deep=True)
    to_predict = to_predict.dropna().reset_index(drop=True)

    to_predict_scaled = scaler.transform(to_predict[features])
    to_predict_scaled = pd.DataFrame(to_predict_scaled)
    to_predict_scaled.columns = prediction_data[features].columns

    to_predict_scaled['log_preds'] = model.predict(to_predict_scaled)
    to_predict_scaled['preds'] = np.exp(to_predict_scaled['log_preds'])-1
    outlier = to_predict_scaled['preds'].quantile(0.95)

    to_predict_scaled['preds'] = np.where(to_predict_scaled['preds']>outlier,outlier,to_predict_scaled['preds'])

    to_predict_scaled[hex_col] = to_predict[hex_col]
    
    return to_predict_scaled[[hex_col,'preds']]


def calculate_tam(predicted,tam_number,hex_col='hexid'):
    '''
    Calculate TAM at hex level.
    
    Parameters
    ----------
    predicted : dataframe
        Input dataframe containg the predictions at hex level
    tam_number : float
        TAM number to be redistributed based on the model predictions
    hex_col : str, default 'hexid'
        Identifier for hex column
        
    Returns
    -------
    DataFrame
        Returns datarame cotaining the TAM and predictions at a hex level
    '''
    
    to_predict_scaled = predicted.copy(deep=True)
    
    to_predict_scaled['preds'] = np.where(to_predict_scaled['preds']<0,0,to_predict_scaled['preds'])
    to_predict_scaled['preds_perc'] = to_predict_scaled['preds']/to_predict_scaled['preds'].sum()
    to_predict_scaled['tam'] = to_predict_scaled['preds_perc'] * tam_number
    
    return to_predict_scaled[[hex_col,'preds','tam']]


def save_model(model,scaler,variable_ordering,path):
    '''
    Save the model object, scaler object and variable ordering as pickle files.
    
    Parameters
    ----------
    model : object
        Trained model object
    scaler : object
        Scaler object used in model training
    variable_ordering : list
        List of features used in model training
    path : str
        Path to save the model, scaler and variable ordering pickle files
    '''
    
    with open(path+'/model.pkl','wb') as f:
        pickle.dump(model,f)
        
    with open(path+'/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)
        
    with open(path+'/variable_ordering.pkl','wb') as f:
        pickle.dump(variable_ordering,f)
    

