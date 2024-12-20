
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_metrics(metrics, filename_prefix):
    accuracy = metrics['accuracy']
    classification_report = metrics['classification_report']

    classes = list(classification_report.keys())[:-3]  
    precision = [classification_report[c]['precision'] for c in classes]
    recall = [classification_report[c]['recall'] for c in classes]
    f1_score = [classification_report[c]['f1-score'] for c in classes]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(x=classes, y=precision, ax=axes[0], palette='Blues_d')
    axes[0].set_title(f'Precision by Class ({filename_prefix})')
    axes[0].set_ylabel('Precision')
    
    sns.barplot(x=classes, y=recall, ax=axes[1], palette='Greens_d')
    axes[1].set_title(f'Recall by Class ({filename_prefix})')
    axes[1].set_ylabel('Recall')

    sns.barplot(x=classes, y=f1_score, ax=axes[2], palette='Oranges_d')
    axes[2].set_title(f'F1-Score by Class ({filename_prefix})')
    axes[2].set_ylabel('F1-Score')

    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_metrics_graph.png')

with open('metrics.json', 'r') as f:
    metrics = json.load(f)

plot_metrics(metrics, 'baseline')
with open('metrics_modified.json', 'r') as f:
    metrics_modified = json.load(f)

plot_metrics(metrics_modified, 'modified')

data = pd.read_csv('iris_data.csv') 

correlation_matrix = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png') 
