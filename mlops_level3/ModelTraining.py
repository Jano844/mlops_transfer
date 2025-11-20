import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import joblib
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
# mlflow Vars
RUN_ID = "test_run_forklifts_01" 
EXPERIMENT_NAME = 'forklifts_experiment'

DATA_PATH = '../cleaned_data/forklifts_combined.csv'
data = pd.read_csv(DATA_PATH, sep=';')
print(len(data))


CATEGORICAL_FEATURES = ['Identifier']
NUMERIC_FEATURES = ['Speed', 'Height', 'weekday']
INTERVAL_FEATURES = ['Latitude', 'Longtitude', 'Hours', 'Minutes']
WEIGHT = (data['Loaded'] == 0).sum() / (data['Loaded'] == 1).sum()
FILENAME = '../models/xgboost_forklift_model.pkl'



def get_xgb_params():
    xgb_params = {
        'objective': 'binary:logistic',
        'learning_rate': 0.1,
        'max_depth': 15,
        'n_estimators': 1000,
        'eval_metric': 'auc',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': WEIGHT,
        'min_child_weight': 3,
        'gamma': 0.1,  
    }
    return xgb_params


# def preprocess_data():
#     global data
#     MIN_CONS = 3
#     # print((data['Loaded'] == 1).sum() / (data['Loaded'] == 0).sum())


#     data.sort_values(['Identifier', 'TimeStamp'], inplace=True)
#     data['status_change'] = (
#             (data['Loaded'] != data['Loaded'].shift()) | 
#             (data['Identifier'] != data['Identifier'].shift())
#         ).cumsum()
#     consecutive_counts = data.groupby(['Identifier', 'status_change']).agg({
#             'Loaded': 'first',
#             'status_change': 'count'
#         }).rename(columns={'status_change': 'count'}).reset_index()
#     valid_sequences = consecutive_counts[consecutive_counts['count'] >= MIN_CONS]
#     valid_sequences['merge_key'] = valid_sequences['Identifier'].astype(str) + '_' + valid_sequences['status_change'].astype(str)
#     data['merge_key'] = data['Identifier'].astype(str) + '_' + data['status_change'].astype(str)
#     data = data[data['merge_key'].isin(valid_sequences['merge_key'])].copy()
#     data = data.drop(['status_change', 'merge_key'], axis=1)

def add_direction():
    global data
    add_direction_Longtitude()
    add_direction_Latitude()

    data['Direction'] = data['LoDirection'] + data['LaDirection']
    data.drop(columns=['LoDirection', 'LaDirection'], inplace=True)
    
    data['angle'] = np.arctan2(data['LaDiff'], data['LoDiff']) * (180 / np.pi)
    data.drop(columns=['LoDiff', 'LaDiff'], inplace=True)

    CATEGORICAL_FEATURES.append('Direction')
    NUMERIC_FEATURES.append('angle')



def add_direction_Longtitude():
    global data

    diffs = data['Longtitude'].diff().fillna(0)
    directions = []
    diff = []

    for d in diffs:
        if d > 0:
            directions.append('E')
        elif d < 0:
            directions.append('W')
        else:
            directions.append('')
        diff.append(d)
        

    data['LoDirection'] = directions
    data['LoDiff'] = diff
    identifier_changed = data['Identifier'] != data['Identifier'].shift(1)
    data.loc[identifier_changed, 'LoDirection'] = ''
    data.loc[identifier_changed, 'LoDiff'] = 0

def add_direction_Latitude():
    global data

    diffs = data['Latitude'].diff().fillna(0)
    directions = []
    diff = []

    for d in diffs:
        if d > 0:
            directions.append('N')
        elif d < 0:
            directions.append('S')
        else:
            directions.append('')
        diff.append(d)
        

    data['LaDirection'] = directions
    data['LaDiff'] = diff
    identifier_changed = data['Identifier'] != data['Identifier'].shift(1)
    data.loc[identifier_changed, 'LaDirection'] = ''
    data.loc[identifier_changed, 'LaDiff'] = 0


def add_feat_height_change():
    global data

    diffs = data['Height'].diff().fillna(0)
    diff = []

    for d in diffs:
        diff.append(d)
        

    data['HeightDiff'] = diff
    identifier_changed = data['Identifier'] != data['Identifier'].shift(1)
    data.loc[identifier_changed, 'HeightDiff'] = np.nan
    NUMERIC_FEATURES.append('HeightDiff')


def prepreprocess_data():
    global data
    # mask = (data['Loaded'] == 1) & (data['Loaded'].shift(1) == 0) & (data['Loaded'].shift(-1) == 0)
    # num_changed = mask.sum()
    # data.loc[mask, 'Loaded'] = 0
    # print(f"\nNumber of rows changed: {num_changed}")

    mask = (data['Loaded'] == 0) & (data['Loaded'].shift(1) == 1) & (data['Loaded'].shift(-1) == 1)
    num_changed = mask.sum()
    data.loc[mask, 'Loaded'] = 1
    # print(f"\nNumber of rows changed: {num_changed}")

def preprocess_data():
    global data
    MIN_CONS = 3

    prepreprocess_data()

    data = data.sort_values(['Identifier', 'TimeStamp']).copy()
    data['status_change'] = (
        (data['Loaded'] != data['Loaded'].shift()) |
        (data['Identifier'] != data['Identifier'].shift())
    ).cumsum()
    consecutive_counts = (
        data.groupby(['Identifier', 'status_change'])
            .agg(
                Loaded=('Loaded', 'first'),
                count=('Loaded', 'size')
            )
            .reset_index()
    )
    valid_sequences = consecutive_counts[consecutive_counts['count'] >= MIN_CONS].copy()
    valid_sequences['merge_key'] = (
        valid_sequences['Identifier'].astype(str) + '_' +
        valid_sequences['status_change'].astype(str)
    )
    data['merge_key'] = (
        data['Identifier'].astype(str) + '_' +
        data['status_change'].astype(str)
    )
    data = data[data['merge_key'].isin(valid_sequences['merge_key'])].copy()
    data.drop(columns=['status_change', 'merge_key'], inplace=True)

    add_direction()
    add_feat_height_change()
    data['Speed_rolling_5'] = data['Speed'].rolling(3, min_periods=1).mean()
    data['Height_rolling_5'] = data['Height'].rolling(3, min_periods=1).mean()
    NUMERIC_FEATURES.append('Speed_rolling_5')
    NUMERIC_FEATURES.append('Height_rolling_5')



def cutoff_outliers():
    global data
    MAX_HEIGHT = 6.7  # maximum height in meters
    MAX_SPEED = 20.0  # maximum speed in m/s

    data['Height'] = data['Height'].clip(upper=MAX_HEIGHT)
    data['Speed'] = data['Speed'].clip(upper=MAX_SPEED)


def extract_time_features():
    global data
    data['TimeStamp'] = pd.to_datetime(data['TimeStamp'], unit='ms')
    data['Hours'] = data['TimeStamp'].dt.hour
    data['Minutes'] = data['TimeStamp'].dt.minute
    data['Seconds'] = data['TimeStamp'].dt.second
    data['weekday'] = data['TimeStamp'].dt.weekday


def split_data():
    global data
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Loaded'])
    X = train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES + INTERVAL_FEATURES]
    y = train_df['Loaded']
    X_test = test_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES + INTERVAL_FEATURES]
    y_test = test_df['Loaded']
    return X, X_test, y, y_test

def train_and_save(X, y):
    global WEIGHT
    WEIGHT = (data['Loaded'] == 0).sum() / (data['Loaded'] == 1).sum()

    cats = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('nom', OneHotEncoder(handle_unknown='ignore'))
    ])
    ints = Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', RobustScaler(with_scaling=True))
    ])
    rats = Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', StandardScaler(with_mean=False))
    ])


    params = get_xgb_params()
    mlflow.log_params(params)

    prep = ColumnTransformer([
        ('n', cats, CATEGORICAL_FEATURES),
        ('i', ints, INTERVAL_FEATURES),
        ('r', rats, NUMERIC_FEATURES)
    ])
    model = Pipeline([
        ('pre', prep),
        ('xgb', XGBClassifier(**params))
    ])

    model.fit(X, y)
    joblib.dump(model, FILENAME)


def evaluate_model(loaded_model, X_test, y_test):

    predictions = loaded_model.predict(X_test)
    predictions = np.rint(predictions)

    # df_results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    # df_results.head(10)

    report = classification_report(y_test, predictions, output_dict=True)


    metrics = {
        # "accuracy": accuracy_score(y_test, predictions),
        # "precision": precision_score(y_test, predictions),
        # "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        # "AUC": roc_auc_score(y_test, predictions),
        "precision_1": report['1']['precision'],
        "recall_1": report['1']['recall'],
        "f1_1": report['1']['f1-score'],
        # "confusion_matrix": confusion_matrix(y_test, predictions),
    }
   

    # print(f"Accuracy:  {metrics['accuracy']:.3f}")
    # print(f"Precision: {metrics['precision']:.3f}")
    # print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-score:  {metrics['f1']:.3f}")
    # print(f"ROC-AUC:   {metrics['AUC']:.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    return metrics



# def main():
#     preprocess_data()
#     cutoff_outliers()
#     extract_time_features()
#     X, X_test, y, y_test = split_data()

#     train_and_save(X, y)

#     loaded_model = joblib.load(FILENAME)
#     evaluate_model(loaded_model, X_test, y_test)



def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # wenn du mlflow ui lokal hast
    mlflow.set_experiment(EXPERIMENT_NAME)            # optional

    with mlflow.start_run(run_name=RUN_ID):

        preprocess_data()
        cutoff_outliers()
        extract_time_features()


        X, X_test, y, y_test = split_data()

        # Beispiel: wichtige Parameter loggen
        mlflow.log_param("train_shape", X.shape)
        mlflow.log_param("test_shape", X_test.shape)

        train_and_save(X, y)

        loaded_model = joblib.load(FILENAME)

        # Modell als Artifact hochladen
        mlflow.sklearn.log_model(loaded_model, "model")

        # Evaluation als Metric loggen
        metrics = evaluate_model(loaded_model, X_test, y_test)
        for key, val in metrics.items():
            mlflow.log_metric(key, val)
        
        mlflow.log_param("CATEGORICAL_FEATURES", ",".join(CATEGORICAL_FEATURES))
        mlflow.log_param("NUMERIC_FEATURES", ",".join(NUMERIC_FEATURES))
        mlflow.log_param("INTERVAL_FEATURES", ",".join(INTERVAL_FEATURES))
    print(len(data))

        
    

if __name__ == "__main__":
    main()
