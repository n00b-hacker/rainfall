import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

df = pd.read_csv('weather.csv')
x = df.drop('Rainfall', axis=1)
y = df['Rainfall']

best_accuracy = 0
best_random_state_split = None
best_random_state_clf = None


def evaluate_model(random_state_split):
  global best_accuracy, best_random_state_split, best_random_state_clf
  x_train, x_test, y_train, y_test = train_test_split(
      x, y, test_size=0.2, random_state=random_state_split)
  for random_state_clf in range(1, 101):
    clf = RandomForestClassifier(random_state=random_state_clf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      best_random_state_split = random_state_split
      best_random_state_clf = random_state_clf
    print("{} | {} | {:.2f}% | Max= {:.2f}% | {} | {}".format(
        random_state_split, random_state_clf, accuracy, best_accuracy,
        best_random_state_split, best_random_state_clf))


Parallel(n_jobs=-1)(delayed(evaluate_model)(random_state_split)
                    for random_state_split in range(1, 102))
