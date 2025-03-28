import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

def plot_confusion_matrix(predictions, labels, title):
    from sklearn.metrics import confusion_matrix
    import numpy as np

    unique_labels = sorted(set(labels))
    cm = confusion_matrix(labels, predictions, labels=unique_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", xticklabels=unique_labels, yticklabels=unique_labels, cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

def plot_classification_report(predictions, labels, title):
    from sklearn.metrics import classification_report
    import pandas as pd

    unique_labels = sorted(set(labels))
    report = classification_report(labels, predictions, target_names=unique_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(10, 8))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues')
    plt.title(f'Classification Report - {title}')

def get_result_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]

def calculate_metrics(predictions, labels):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    return accuracy, precision, recall, f1

def save_metrics_to_file(metrics, knn_values, output_file):
    # Crea un DataFrame con le metriche e i valori di KNN
    metrics_names = ['KNN Value', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    rounded_metrics = [[knn] + [round(value, 2) for value in metric] for knn, metric in zip(knn_values, metrics)]
    df = pd.DataFrame(rounded_metrics, columns=metrics_names)

    # Salva il DataFrame in un file .txt
    with open(output_file, 'w') as file:
        file.write(df.to_string(index=False))

def plot_metrics_table(metrics, knn_values, header_color='#40466e', row_colors=('#f1f1f2', 'white'), save_path=None):
    import pandas as pd
    import matplotlib.pyplot as plt

    # Crea il DataFrame
    metrics_names = ['KNN Value', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    rounded_metrics = [[knn] + [round(value, 2) for value in metric] for knn, metric in zip(knn_values, metrics)]
    df = pd.DataFrame(rounded_metrics, columns=metrics_names)

    # Crea la figura e l'asse
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Crea la tabella
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Personalizza le celle
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Intestazione
            cell.set_text_props(weight='bold', color='white', fontsize=14)
            cell.set_facecolor(header_color)
        else:
            # Colori alternati per le righe
            cell.set_facecolor(row_colors[row % 2])
        cell.set_edgecolor('black')
        cell.set_height(0.3)

    # Salva o mostra la tabella
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    results_directory = './results/classification/cross-domain/CD_577_SM_aug/'
    result_files = get_result_files(results_directory)
    last_folder_name = os.path.basename(os.path.normpath(results_directory))

    metrics = []
    knn_values = []

    for file_path in result_files:
        accuracies, predictions, labels = parse_results(file_path)
        knn_value = int(os.path.basename(file_path).split('knn')[1].split('.')[0])
        knn_values.append(knn_value)

        # Calcola le metriche
        accuracy, precision, recall, f1 = calculate_metrics(predictions, labels)
        metrics.append((accuracy, precision, recall, f1))

        # Traccia la matrice di confusione
        # title = f'{last_folder_name}_knn{knn_value}'
        # plot_confusion_matrix(predictions, labels, title)

    # Salva le metriche in un file .txt
    output_file = './metrics_results.txt'
    save_metrics_to_file(metrics, knn_values, output_file)

    # Traccia la tabella delle metriche
    plot_metrics_table(metrics, knn_values)

    print(f"Metrics saved to {output_file}")

    plt.show()