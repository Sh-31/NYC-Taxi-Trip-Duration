import joblib
import datetime
import numpy as np
import pandas as pd
from helper import update_baseline_metadata
from sklearn.preprocessing import OneHotEncoder, StandardScaler,RobustScaler, MinMaxScaler, PolynomialFeatures, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LassoCV
from sklearn.metrics import r2_score, root_mean_squared_error


# Global variables
seed = 31  # random seed
degree = 6  # degree of polynomial features 
do_feature_selection = False # do feature selection using Lasso
do_add_speed = False # add speed feature
speed_model, data_preprocessor_speed = None , None # speed model  
np.random.seed(seed)

target = 'trip_duration'
# Feature of target feature (trip_duration) 

numeric_features = [
      "dropoff_longitude", "dropoff_latitude", "distance_haversine", "distance_manhattan", "direction",
]

categorical_features = [
     "passenger_count","pickup_day","pickup_month","pickup_Season","pickup_dayofweek","pickup_hour","store_and_fwd_flag","vendor_id"
]

# Feature of speed model
speed_numeric_features = [
     "pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","distance_haversine","distance_manhattan","direction",
]

speed_categorical_features = [
     "passenger_count","pickup_day","pickup_month","pickup_Season","pickup_dayofweek","pickup_hour","store_and_fwd_flag","vendor_id"
]

def predict_eval(model, data_preprocessed, target, name) -> str:
    y_train_pred = model.predict(data_preprocessed)
    rmse = root_mean_squared_error(target, y_train_pred)
    r2 = r2_score(target, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")
    return rmse, r2, f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}"

def data_preprocessing_pipeline(categorical_features=[], numeric_features=[]):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown="infrequent_if_exist"))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    data_preprocessor = Pipeline(steps=[
        ('preprocessor', column_transformer)
    ])

    return data_preprocessor

def log_transform(x): 
    return np.log1p(np.maximum(x, 0))

def with_suffix(_, names: list[str]):  # https://github.com/scikit-learn/scikit-learn/issues/27695
    return [name + '__log' for name in names]

def pipeline(train, test, do_feature_selection=True):

    LogFeatures = FunctionTransformer(log_transform, feature_names_out=with_suffix)
    train_features = numeric_features + categorical_features

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()), 
        ('poly', PolynomialFeatures(degree=degree)),
        ('log', LogFeatures),
    ])

    categorical_transformer = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown="infrequent_if_exist"))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    data_preprocessor = Pipeline(steps=[
        ('preprocessor', column_transformer)
    ])

   
    train_preprocessed = data_preprocessor.fit_transform(train[train_features])
    test_preprocessed = data_preprocessor.transform(test[train_features])

    if do_feature_selection:
        lasso_cv = LassoCV(cv=5, max_iter=5000, random_state=seed)
        lasso_cv.fit(train_preprocessed, train[target])
        selected_feature_indices = [i for i, coef in enumerate(lasso_cv.coef_) if coef != 0]
        
        all_feature_names = data_preprocessor.named_steps['preprocessor'].get_feature_names_out()
        selected_feature_names = all_feature_names[selected_feature_indices]

        print("LassoCV selected features: ", selected_feature_names)

        train_preprocessed_lasso = train_preprocessed[:, selected_feature_indices]
        test_preprocessed_lasso = test_preprocessed[:, selected_feature_indices]
    else:
        train_preprocessed_lasso = train_preprocessed
        test_preprocessed_lasso = test_preprocessed
        selected_feature_names = train_features

    ridge = Ridge(alpha=1, random_state=seed)
    ridge.fit(train_preprocessed_lasso, train[target])

    train_rmse, train_r2, _ = predict_eval(ridge, train_preprocessed_lasso, train[target], "train")
    test_rmse, test_r2, _ = predict_eval(ridge, test_preprocessed_lasso, test[target], "val")

    return ridge, selected_feature_names, data_preprocessor, train_rmse, train_r2, test_rmse, test_r2

def add_speed(df, speed_data_preprocessor, model=None, train_or_predict="train", validate=False):
    speed_train_features = speed_categorical_features + speed_numeric_features

    if train_or_predict == "train":
        df_preprocessed = speed_data_preprocessor.fit_transform(df[speed_train_features])
    else:
        df_preprocessed = speed_data_preprocessor.transform(df[speed_train_features])

    if train_or_predict == "train":
        df_speed = df["distance_haversine"] / (df['trip_duration'])
        speed_model = Ridge(alpha=1, random_state=seed)
        speed_model.fit(df_preprocessed, df_speed)
        df["speed"] = speed_model.predict(df_preprocessed)

        rmse = root_mean_squared_error(df_speed, df["speed"])
        r2 = r2_score(df_speed, df["speed"])
        print(f"Speed Model Train RMSE: {rmse:.4f}, Train R2: {r2:.4f}")
        numeric_features.append("speed")
        return speed_model, df
    else:
        df["speed"] = model.predict(df_preprocessed)

        if validate:
            df_speed = df["distance_haversine"] / (df['trip_duration'])
            rmse = root_mean_squared_error(df_speed, df["speed"])
            r2 = r2_score(df_speed, df["speed"])
            print(f"Test Speed Model RMSE: {rmse:.4f}, Test R2: {r2:.4f}")

        return df["speed"]

if __name__ == "__main__":
    data_version = 0

    data_path =  f"processed_data/{data_version}"
    train_path = f"{data_path}/train.csv"
    val_path =   f"{data_path}/val.csv"

    df_train = pd.read_csv(train_path)
    df_val =   pd.read_csv(val_path)
   
    if do_add_speed:

        data_preprocessor_speed = data_preprocessing_pipeline(
           categorical_features=speed_categorical_features, numeric_features=speed_numeric_features
        )

        speed_model, df_train = add_speed(df_train, data_preprocessor_speed, train_or_predict="train")
        df_val["speed"] = add_speed(df_val, data_preprocessor_speed, model=speed_model, train_or_predict="predict", validate=True)
        train_features = categorical_features + numeric_features

    model, selected_feature_names, data_preprocessor, train_rmse, train_r2, test_rmse, test_r2 = pipeline(df_train, df_val, do_feature_selection)

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    filename = f'model_{now}_Train_RMSE_{train_rmse:.2f}_R2_{train_r2:.2f}_Test_RMSE_{test_rmse:.2f}_R2_{test_r2:.2f}.pkl'

    if do_feature_selection:
        selected_feature_names = selected_feature_names.tolist()

    model_data = {
        'model': model,
        'speed_model':speed_model,
        'data_preprocessor_speed':data_preprocessor_speed,
        'data_path': data_path,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'selected_feature_names': selected_feature_names,
        'data_preprocessor': data_preprocessor,
        'data_version': data_version,
        'random_seed': seed
    }

    joblib.dump(model_data, filename)
    print(f"Model saved as {filename}")
    update_baseline_metadata(model_data)