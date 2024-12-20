import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

X = iris_df.drop(columns=['target'])
y = iris_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

rf_model = RandomForestClassifier(n_estimators=50, max_depth=1, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, output_dict=True)

metrics = {
    "accuracy": accuracy,
    "classification_report": classification_rep
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved to metrics.json:")
print(json.dumps(metrics, indent=4))
