import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import pandas as pd

# 1. Загрузка данных
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# 2. Разделение на обучающую и тестовую выборки
X = iris_df.drop(columns=['target'])
y = iris_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Обучение модели
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. Предсказание и метрики
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, output_dict=True)

# 5. Сохранение метрик в файл
metrics = {
    "accuracy": accuracy,
    "classification_report": classification_rep
}

# Сохраняем метрики в файл metrics.json
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Печатаем для проверки
print("Metrics saved to metrics.json:")
print(json.dumps(metrics, indent=4))
