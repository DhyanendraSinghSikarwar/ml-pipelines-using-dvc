import numpy as np
import pandas as pd
import pickle
import json

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

def load_model():
    # Load the model
    return pickle.load(open('model.pkl', 'rb'))


def load_data():
    # fetch the data from data/features
    test_data = pd.read_csv('./data/features/test_bow.csv')
    return test_data


def prepare_data(test_data: pd.DataFrame) -> tuple:
    # Prepare the data for testing
    X_test = test_data.iloc[:, :-1].values  # All columns except the last one
    y_test = test_data.iloc[:, -1].values   # Last column is the label
    return X_test, y_test


def make_predictions(clf: GradientBoostingClassifier, X_test: np.ndarray) -> tuple:
    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
    return y_pred, y_pred_proba


def evaluate_model(y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> tuple:
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return accuracy, precision, recall, auc


def save_metrics(accuracy: float, precision: float, recall: float, auc: float) -> None:
    # Save the metrics to a JSON file
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }
    
    with open('metrics.json', 'w') as file:
        json.dump(metrics_dict, file, indent=4)


def main():
    # Load the model
    clf = load_model()
    
    # Load the data
    test_data = load_data()
    
    # Prepare the data
    X_test, y_test = prepare_data(test_data)
    
    # Make predictions
    y_pred, y_pred_proba = make_predictions(clf, X_test)
    
    # Evaluate the model
    accuracy, precision, recall, auc = evaluate_model(y_test, y_pred, y_pred_proba)
    
    # Save the metrics
    save_metrics(accuracy, precision, recall, auc)

if __name__ == "__main__":
    main()
