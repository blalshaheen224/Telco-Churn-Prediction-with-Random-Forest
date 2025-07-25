import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib


path = '/content/WA_Fn-UseC_-Telco-Customer-Churn.csv'
def load_data(path):
  df = pd.read_csv(path)
  return df

def explore_data(df):
    print('head')
    print(df.head())
    print('info')
    print(df.info())
    print('describe')
    print(df.describe())
    print('isnull')
    print(df.isnull().sum())


def clean_data(df):
    df = df.dropna()  
    df = df.drop_duplicates()
    return df


def preprocessing_data(df):
  encoders ={}
  for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
  df_yes = df[df['Churn']==1]
  df_no = df[df['Churn']==0]
  df_yes = resample(df_yes , replace=True ,n_samples=len(df_no) ,random_state=42)
  df = pd.concat([df_no ,df_yes])
  return df ,encoders


def get_feature_importance():
   X = df.drop(['Churn'], axis=1)
   y = df['Churn']
   model = RandomForestClassifier()
   model.fit(X, y)
   importances = model.feature_importances_
   feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
   feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
   print(feature_importance_df)

def feature_engineering(df):
  #df = df.drop(['customerID'], axis=1)
  cols_to_drop = ['StreamingTV', 'StreamingMovies', 'PhoneService','customerID']
  df = df.drop(columns=cols_to_drop)
  return df

def model_training(df):
  X = df.drop(['Churn'], axis=1)
  y = df['Churn']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = RandomForestClassifier()
  model.fit(X_train, y_train)
  return model , X_test, y_test
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Heatmap")
    plt.tight_layout()
    plt.show()
    print(classification_report(y_test, y_pred))
   
def save_model_bundle(model, encoders, path='model_bundle.pkl'):
  joblib.dump({'model': model, 'encoders': encoders}, path)

def load_model_bundle(path='model_bundle.pkl'):
    bundle = joblib.load(path)
    model = bundle['model']
    encoders = bundle['encoders']
    return model, encoders

def predict_new_data(new_df, model, encoders):
    
    new_df = new_df.dropna()
    new_df = new_df.drop_duplicates()

    for col, encoder in encoders.items():
        if col in new_df.columns:
            new_df[col] = encoder.transform(new_df[col])
    cols_to_drop = ['StreamingTV', 'StreamingMovies', 'PhoneService', 'customerID']
    new_df = new_df.drop(columns=[col for col in cols_to_drop if col in new_df.columns])
    
    predictions = model.predict(new_df)
    return predictions

def main():
  df = load_data(path)
  #explore_data(df)
  df = clean_data(df)
  df , encoders = preprocessing_data(df)
  df = feature_engineering(df)
  model , X_test, y_test =  model_training(df)
  evaluate_model(model, X_test, y_test)
  save_model_bundle(model, encoders)
  #get_feature_importance(df)


main()


