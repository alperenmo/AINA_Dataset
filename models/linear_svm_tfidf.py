import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC


train_path = '/path/antimicrobial_nanoparticles_train_data.csv'
test_path = '/path/antimicrobial_nanoparticles_test_data.csv'

def load_and_preprocess(path):
  df = pd.read_csv(path)
  df['input_text'] = (df['Paper'] + ' ' + df['Abstract']).str.lower()
  df['label'] = df['Decision1'].map({'exclude': 0, 'include': 1})
  return df[['input_text', 'label']]

def tf_idf_transformer(train_data, text_data, max_features=1000):
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    X_train = tfidf.fit_transform(train_data)
    X_test = tfidf.transform(text_data)
    return X_train, X_test, tfidf

def evaluate_model(model, X_train, y_train, X_test, y_test,target_names = ['exclude','include'], use_random_state = True):
  if use_random_state and hasattr(model,'random_state'):
    model.set_params(random_state=42)
  model.fit(X_train,y_train)
  y_pred = model.predict(X_test)
  report = classification_report(
      y_test, y_pred,
      target_names = target_names
  )
  return report

if __name__ == '__main__':
  train_df = load_and_preprocess(train_path)
  test_df = load_and_preprocess(test_path)
  X_train, X_test, tfidf = tf_idf_transformer(train_df['input_text'], test_df['input_text'])
  y_train,y_test = train_df['label'], test_df['label']
  svm = LinearSVC()
  report = evaluate_model(svm, X_train, y_train, X_test, y_test)
  print(report)






