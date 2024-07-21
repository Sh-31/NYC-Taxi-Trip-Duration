import numpy as np
import pandas as pd
from prepare import haversine_distance , calculate_direction ,  manhattan_distance
from train import log_transform, with_suffix, FunctionTransformer, predict_eval, target
from helper import load_model 

def prepare_data(df_test:pd.DataFrame):
    
    df_test['trip_duration'] = np.log1p(df_test['trip_duration']) 
    df_test["pickup_datetime"] = pd.to_datetime(df_test["pickup_datetime"]) 
      
    # From our EDA, can use these distance features.
    df_test['distance_haversine'] = df_test.apply(haversine_distance, axis=1)
    df_test['direction'] =   df_test.apply(calculate_direction, axis=1)
    df_test['distance_manhattan'] = df_test.apply(manhattan_distance, axis=1)

    bins = [0, 2, 5, 8, 11, 12]  # 0, 2, 5, 8, 11, 12 represent the starting and ending months of each season
    labels = ['0', '1', '2', '3', '4'] # Labels for each season ['Winter', 'Spring', 'Summer', 'Autumn', 'Winter'] 

    df_test["pickup_hour"] = df_test["pickup_datetime"].dt.hour
    df_test["pickup_day"]  = df_test["pickup_datetime"].dt.day
    df_test["pickup_dayofweek"] = df_test["pickup_datetime"].dt.dayofweek
    df_test["pickup_month"]  = df_test["pickup_datetime"].dt.month
    df_test['pickup_Season'] = pd.cut(df_test["pickup_month"] , bins=bins, labels=labels, right=False,ordered=False) 

    df_test.drop(columns=['id', 'pickup_datetime'], inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    return df_test 


if __name__ == "__main__":

    train_path = "split/train.zip" # "split/train.csv"
    val_path   = "split/val.zip"   # "split/val.csv"
    test_path  = "split/test.zip"  # "split/test.csv"
    model_path = "model_2024_07_20_14_34_Train_RMSE_0.44_R2_0.70_Test_RMSE_0.44_R2_0.69.pkl"

    LogFeatures = FunctionTransformer(log_transform, feature_names_out=with_suffix)
    modeling_pipeline = load_model(model_path)

    data_preprocessor = modeling_pipeline['data_preprocessor']
    training_features = modeling_pipeline['selected_feature_names']
    model = modeling_pipeline['model']

    # df_train = pd.read_csv(train_path)
    # df_train = prepare_data(df_train)
    # df_train_processed = data_preprocessor.transform(df_train[training_features])
    # rmse, r2, _  = predict_eval(model, df_train_processed, df_train[target] ,'train')

    df_val = pd.read_csv(val_path)
    df_val = prepare_data(df_val)
    df_vail_processed = data_preprocessor.transform(df_val[training_features])
    rmse, r2, _  = predict_eval(model, df_vail_processed, df_val[target] ,'vail')

    # df_test = pd.read_csv(test_path)
    # df_test = prepare_data(df_test)
    # df_test_processed = data_preprocessor.transform(df_test[training_features])
    # rmse, r2, _  = predict_eval(model, df_test_processed, df_test[target] ,'test')