from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df.to_csv('iris_data.csv', index=False)
print("Iris Dataset:")
print(iris_df.head())
