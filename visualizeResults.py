import matplotlib.pyplot as plt
import seaborn as sns

def parse_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    accuracies = []
    predictions = []
    labels = []

    for line in lines:
        if line.startswith("Accuratezza"):
            accuracy = float(line.split(":")[1].strip())
            accuracies.append(accuracy)
        elif line.startswith("Predizione"):
            parts = line.split(",")
            prediction = parts[0].split(":")[1].strip()
            label = parts[1].split(":")[1].strip()
            predictions.append(prediction)
            labels.append(label)

    return accuracies, predictions, labels

def plot_accuracies(accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.title('Model Accuracies per Augmentation Index')
    plt.xlabel('Augmentation Index')
    plt.ylabel('Accuracy')
    plt.grid(True)

def plot_confusion_matrix(predictions, labels):
    from sklearn.metrics import confusion_matrix
    import numpy as np

    unique_labels = sorted(set(labels))
    cm = confusion_matrix(labels, predictions, labels=unique_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", xticklabels=unique_labels, yticklabels=unique_labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

def plot_classification_report(predictions, labels):
    from sklearn.metrics import classification_report
    import pandas as pd

    unique_labels = sorted(set(labels))
    report = classification_report(labels, predictions, target_names=unique_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(10, 8))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues')
    plt.title('Classification Report')

if __name__ == "__main__":
    #file_path = './results/classification/cross-domain/myDatasetClassification_CDAug_fin.txt'
    file_path = './results/classification/cross-domain/myDatasetClassification_CD_fin.txt'
    accuracies, predictions, labels = parse_results(file_path)

    plot_confusion_matrix(predictions, labels)
    plot_classification_report(predictions, labels)
    plot_accuracies(accuracies)
    
    plt.show()