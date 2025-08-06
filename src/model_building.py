import numpy as np
import pandas as pd
import pickle
import yaml

from sklearn.ensemble import GradientBoostingClassifier

def load_params():
    return yaml.safe_load(open('params.yaml', 'r'))


def load_data():
    # fetch the data from data/features
    train_data = pd.read_csv('./data/features/train_bow.csv')
    return train_data


def prepare_data(train_data: pd.DataFrame) -> tuple:
    # Prepare the data for training
    X_train = train_data.iloc[:, :-1].values  # All columns except the last one
    y_train = train_data.iloc[:, -1].values   # Last column is the label
    return X_train, y_train


def model_building(params: dict, X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
    # Define the model
    clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
    
    # Train the model
    clf.fit(X_train, y_train)
    
    return clf


# Save the model
def save_model(model: GradientBoostingClassifier, filename: str) -> None:
    # Save the model
    pickle.dump(model, open(filename, 'wb'))


def main():
    # Load parameters
    params = load_params()['model_building']
    
    # Load data
    train_data = load_data()

    # Prepare data
    X_train, y_train = prepare_data(train_data)

    # Build model
    clf = model_building(params, X_train, y_train)

    # Save model
    save_model(clf, 'model.pkl')

if __name__ == "__main__":
    main()
