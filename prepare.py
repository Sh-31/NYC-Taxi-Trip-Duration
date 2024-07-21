import os
import joblib
import datetime
import pandas as pd
import numpy as np
from geopy import distance
from geopy.point import Point
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import math
import json


def remove_outliers(df, target_col, feature_col, method='zscore', factor=3):
    """
    Removes outliers from a DataFrame based on the specified method.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to remove outliers.
    target_col (str): The target column.
    feature_col (str): The feature column.
    method (str): The method to use for outlier detection ('zscore' or 'iqr'). Default is 'zscore'.
    factor (float): The factor to use for IQR outlier detection. Default is 3.

    Returns:
    pd.DataFrame: A DataFrame with outliers removed.
    """
    if target_col not in df.columns or feature_col not in df.columns:
        raise ValueError(f"Columns '{target_col}' or '{feature_col}' not found in DataFrame")

    if method == 'zscore':
        z_scores = np.abs(stats.zscore(df[[target_col, feature_col]]))
        filtered_entries = (z_scores < factor).all(axis=1)
    
    elif method == 'iqr':
        Q1 = df[feature_col].quantile(0.25)
        Q3 = df[feature_col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        filtered_entries = (df[feature_col] >= lower_bound) & (df[feature_col] <= upper_bound)
    
    else:
        raise ValueError("Method must be either 'zscore' or 'iqr'")

    df_cleaned = df[filtered_entries].copy()
    return df_cleaned

def delete_infrequent_categories(df, categorical_features, threshold=5):
    """
    Deletes rows containing infrequent categories in the specified categorical features.
    """
    for feature in categorical_features:

        category_counts = df[feature].value_counts()
        
        infrequent_categories = category_counts[category_counts < threshold].index
        
        df = df[~df[feature].isin(infrequent_categories)]

    return df

def haversine_distance(row):
    pick = Point(row['pickup_latitude'], row['pickup_longitude'])
    drop = Point(row['dropoff_latitude'], row['dropoff_longitude'])
    dist = distance.geodesic(pick, drop)
    return dist.km

def calculate_direction(row):
    pickup_coordinates =  Point(row['pickup_latitude'], row['pickup_longitude'])
    dropoff_coordinates = Point(row['dropoff_latitude'], row['dropoff_longitude'])
    
    # Calculate the difference in longitudes
    delta_longitude = dropoff_coordinates[1] - pickup_coordinates[1]
    
    # Calculate the bearing (direction) using trigonometry
    y = math.sin(math.radians(delta_longitude)) * math.cos(math.radians(dropoff_coordinates[0]))
    x = math.cos(math.radians(pickup_coordinates[0])) * math.sin(math.radians(dropoff_coordinates[0])) - \
        math.sin(math.radians(pickup_coordinates[0])) * math.cos(math.radians(dropoff_coordinates[0])) * \
        math.cos(math.radians(delta_longitude))
    
    # Calculate the bearing in degrees
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    
    # Adjust the bearing to be in the range [0, 360)
    bearing = (bearing + 360) % 360
    
    return bearing


# 3 Manhattan Distance
def manhattan_distance(row):
    # Convert the latitude and longitude differences to distances
    lat_distance = abs(row['pickup_latitude'] - row['dropoff_latitude']) * 111  # approx 111 km per degree latitude
    lon_distance = abs(row['pickup_longitude'] - row['dropoff_longitude']) * 111 * math.cos(math.radians(row['pickup_latitude']))  # adjust for latitude
    
    # Manhattan distance is the sum of the latitudinal and longitudinal distances
    return lat_distance + lon_distance


def marge_spilite(df_train , df_test):
    
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    df_combined = df_combined.sample(frac=1, random_state=7).reset_index(drop=True)

    train_size = 0.90
    val_size = 0.05
    test_size = 0.05

    df_train_val, df_test = train_test_split(df_combined, test_size=test_size, random_state=42) # split train_val 0.95 and test 0.05

    relative_val_size = val_size / (train_size + val_size)
    df_train, df_val = train_test_split(df_train_val, test_size=relative_val_size, random_state=42) # split train 0.9 and val 0.05

    # Verify the sizes of the splits
    print(f'Train set size: {len(df_train)}')
    print(f'Validation set size: {len(df_val)}')
    print(f'Test set size: {len(df_test)}')
    return df_train, df_val, df_test


if __name__ == "__main__":
    df_train = pd.read_csv('split/train.csv')
    df_val = pd.read_csv('split/val.csv')

    df_train['trip_duration'] = np.log1p(df_train['trip_duration'])
    df_val['trip_duration']  = np.log1p(df_val['trip_duration'])
    
    df_train["pickup_datetime"] = pd.to_datetime(df_train["pickup_datetime"]) 
    df_val["pickup_datetime"] = pd.to_datetime(df_val["pickup_datetime"])
   
    # From our EDA, can use these distance features.
    df_train['distance_haversine'] = df_train.apply(haversine_distance, axis=1)
    df_val['distance_haversine'] = df_val.apply(haversine_distance, axis=1)
   
    df_train['direction'] =   df_train.apply(calculate_direction, axis=1)
    df_val['direction']   =   df_val.apply(calculate_direction,   axis=1)

    df_train['distance_manhattan'] = df_train.apply(manhattan_distance, axis=1)
    df_val['distance_manhattan'] = df_val.apply(manhattan_distance, axis=1)
  
    # From our EDA, we can use these features.
    bins = [0, 2, 5, 8, 11, 12]  # 0, 2, 5, 8, 11, 12 represent the starting and ending months of each season
    labels = ['0', '1', '2', '3', '4'] # Labels for each season ['Winter', 'Spring', 'Summer', 'Autumn', 'Winter'] 

    df_train["pickup_hour"] = df_train["pickup_datetime"].dt.hour
    df_train["pickup_day"]  = df_train["pickup_datetime"].dt.day
    df_train["pickup_dayofweek"] = df_train["pickup_datetime"].dt.dayofweek
    df_train["pickup_month"]  = df_train["pickup_datetime"].dt.month
    df_train['pickup_Season'] = pd.cut(df_train["pickup_month"] , bins=bins, labels=labels, right=False,ordered=False) 

    df_val["pickup_hour"] = df_val["pickup_datetime"].dt.hour
    df_val["pickup_day"]  = df_val["pickup_datetime"].dt.day
    df_val["pickup_dayofweek"] = df_val["pickup_datetime"].dt.dayofweek
    df_val["pickup_month"]  = df_val["pickup_datetime"].dt.month
    df_val['pickup_Season'] = pd.cut(df_val["pickup_month"] , bins=bins, labels=labels, right=False,ordered=False)

    df_train.drop(columns=['id', 'pickup_datetime'], inplace=True)
    df_val.drop(columns=['id', 'pickup_datetime'], inplace=True)

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)

    # From our EDA, we need to use remove outliers from passenger_count and seems distance_km is necessary also.
    # categorical_features = ['vendor_id', 'passenger_count',  "pickup_hour", "pickup_day", "pickup_dayofweek","pickup_month","pickup_Season",'store_and_fwd_flag']

    # threshold = 5
    # df_train = delete_infrequent_categories(df_train, categorical_features, threshold=threshold)


    # features_with_outliers = [
    #                             {'categorical_features':categorical_features, "method": 'clip' , 'threshold':threshold},                                     
    #                          ]

    # for dict_feature in features_with_outliers:
    #         if 'categorical_features' in dict_feature.keys(): continue
    #         df_train = remove_outliers(df_train, 'trip_duration', dict_feature['feature'], method=dict_feature['method'] , factor=dict_feature['factor'])


    id = 0 # id for each version dataset splitm
    directory = f'processed_data/{id}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    df_train.to_csv(f"{directory}/train.csv", index=False)
    df_val.to_csv(f"{directory}/val.csv", index=False)

    # Save metadata
    metadata = {
        'version': id,
        'version_description': "This version have all rows without any outlier removal.",
        'feature_names': df_train.columns.tolist(),
        'num_rows_train': len(df_train),
        'num_rows_val': len(df_val),
        'outlier_pipeline': 
         None
        ,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(f"{directory}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)

    print("Data and metadata have been saved successfully.")