import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import CountVectorizer

# fetch the data from data/processed
train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

# Apply BOW
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

#apply Bag of Words
vectorizer = CountVectorizer()

# Fit and transform the training data, transform the test data
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

train_df = pd.DataFrame(X_train_bow.toarray(), columns=vectorizer.get_feature_names_out())
train_df['label'] = y_train
test_df = pd.DataFrame(X_test_bow.toarray(), columns=vectorizer.get_feature_names_out())
test_df['label'] = y_test

train_data.fillna('', inplace=True)
test_data.fillna('', inplace=True)

# store the data inside data/features
data_path = os.path.join('data', 'features')
os.makedirs(data_path, exist_ok=True)

train_df.to_csv(os.path.join(data_path, 'train_bow.csv'), index=False)
test_df.to_csv(os.path.join(data_path, 'test_bow.csv'), index=False)
