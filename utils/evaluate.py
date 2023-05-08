import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def evaluate_model(model, test_loader, device):
    model.eval()
    running_corrects = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds==labels.data)
            total += labels.size(0)

            y_true += labels.cpu().numpy().tolist()
            y_pred += preds.cpu().numpy().tolist()

    accuracy = running_corrects.double()/total
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')

    print(f'Test Accuracy = {100*accuracy:.4f}%\nF1 Score = {f1:.4f}\nRecall = {recall:.4f}\nPrecision = {precision:.4f}')

    return accuracy, y_true, y_pred

def conf_matrix(dataset, y_true, y_pred):
    le = LabelEncoder()
    le.fit(dataset.classes)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap='inferno')
    plt.xticks(np.arange(len(dataset.classes))[::2], le.transform(dataset.classes)[::2], rotation=90)
    plt.yticks(np.arange(len(dataset.classes))[::2], le.transform(dataset.classes)[::2])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.colorbar(shrink=1.0)