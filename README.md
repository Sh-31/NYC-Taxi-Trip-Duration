# [NYC-Taxi-Trip-Duration](https://www.kaggle.com/code/sherif31/new-york-city-taxi-trip-duration) 
## Project Overview
This project aims to predict the total ride duration of taxi trips in New York City as part of the NYC Taxi Duration Prediction competition on Kaggle. The model uses data provided by the NYC Taxi and Limousine Commission, including information such as pickup time, geo-coordinates, number of passengers, and other variables.

## Dependencies
```shell
pip install -r requirements.txt  (for Windows)
pip3 install -r requirements.txt (for Linax)
```
## Usage
```shell
cd  NYC-Taxi-Trip-Duration
python load_test.py (for Windows)
python3 load_test.py (for Linax)
```
## Repo structure and File descriptions
```
NYC-Taxi-Trip-Duration/
├── README.md
├── requirements.txt
├── (EDA) New York City Taxi-Trip Duration.ipynb
├── Trip Duration Prediction Project Report.pdf
├── load_test.py
├── train.py
├── helper.py
├── prepare.py
├── history.json
├── Baseline_model_metadata.json
├── processed_data/   
│   └── 0
│       ├── train.csv.zip
│       ├── val.csv.zip
│       └── metadata.json
│   └── 1
│       ├── train.csv.zip
│       ├── val.csv.zip
│       └── metadata.json
└── split
    ├── train.csv.zip
    ├── test.csv.zip
    └── submission.csv.zip
```
- `README.md`: Contains information about the project, how to set it up, and any other relevant details.
- `requirements.txt`: Lists all the dependencies required for the project to run.
- `(EDA) New York City Taxi-Trip Duration.ipynb`: Jupyter notebook containing exploratory data analysis on New York City taxi trip duration.
- `load_test.py`: Python script for loading test data.
- `train.py`: Python script for training the model (main script).
- `helper.py`: Python script containing helper functions used in other scripts.
- `prepare.py`: Python script for data preparation (Data Versioning).
- `history.json`: JSON file containing the training history of trained models.
- `Baseline_model_metadata.json`: JSON file containing metadata about the baseline model.
- `Trip Duration Prediction Project Report.pdf`: Report summarizes the project.
- `processed_data/0`: Directory containing processed data for a specific iteration (id) of data processing (data verison).
  - `train.csv.zip`: Training data in zip format.
  - `val.csv.zip`: Validation data in zip format.
  - `metadata.json`: Metadata related to this processed data.
- `processed_data/1`: Another directory containing processed data for a different verison.
- `split`: Directory containing data split for training/testing.
  - `train.csv.zip`: Training data in zip format.
  - `test.csv.zip`: Test data in zip format.
  - `submission.csv.zip`: Submission file in zip format.
- - `model_2024_07_20_14_34_Train_RMSE_0.44_R2_0.70_Test_RMSE_0.44_R2_0.69.pkl`: The baseline model stored as pkl format.
## Notes

All the data files are compressed due to GitHub's limitations on pushing larger data files. Please be careful and use appropriate software to decompress and unzip the data files as needed.

## Data exploration

### Target Variable: Trip Duration
- Distribution resembles a Gaussian distribution with a long right tail (right-skewed)
- Most trips are between 150 seconds and 1000 seconds (about 2.5 to 16.7 minutes)
- Log transformation applied to visualize better and help with modeling large values

### Feature Analysis
1. Discrete Numerical Features:
   - Vendor ID and passenger count analyzed
   - No significant difference in trip duration among vendors
   - Trips with 7-8 passengers tend to have shorter durations, possibly due to trip purpose

2. Geographical Features:
   - Haversine distance calculated using pickup and dropoff coordinates
   - Most trips range from less than 1 km to 25 km
   - Speed of trips calculated using distance and duration

3. Temporal Analysis:
   - Longer trip durations observed during summer months
   - Weekend trips generally longer than weekdays
   - Shorter durations during morning and evening rush hours

### Correlation Analysis
- Strong positive correlation between trip duration and distance
- Negative correlation between trip duration and speed

## Modeling

### Data Pipeline
1. Feature splitting into categorical and numerical
2. One-hot encoding for categorical features
3. Standard scaling for numerical features
4. Polynomial Features (degree=6)
5. Log transformation applied

### Results
- RMSE: 0.4427 (Validation)
- R²: 0.6938 (Validation)

### Lessons and Future Work
- Feature selection improves model performance
- Outlier removal for intra-trip duration doesn't improve performance
- Estimating speed as a separate feature didn't significantly improve results
- Consider exploring more complex algorithms like XGBoost and ensemble methods for better performance


