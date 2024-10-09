import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from Client.preprocessing import load_data, preprocess_data
from xgboost import DMatrix

def evaluate_model(model, X_test_dmatrix, y_test):
    y_pred = model.predict(X_test_dmatrix)

    if len(y_pred.shape) > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = (y_pred > 0.5).astype(int)

    y_test_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

    return accuracy, precision, recall, f1, conf_matrix


def plot_confusion_matrix(conf_matrix, classname, class_labels):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {classname}')
    plt.colorbar()

    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels)

    thresh = conf_matrix.max() / 2.0
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_test, y_scores, classname):
    y_test_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    if len(y_scores.shape) > 1:
        n_classes = y_scores.shape[1]
    else:
        n_classes = 2  

    y_test_bin = label_binarize(y_test_classes, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {i}')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {classname} Classes')
    plt.legend(loc="lower right")
    plt.show()

def compute_metrics_for_datasets(models, datasets):
    
    class_labels = ["class0", "class1", "class2", "class3"]

    for classname, model in zip(datasets, models):
        x_train, y_train, x_test, y_test = load_data(classname, "Client/Dataset")
        x_train_scaled, x_test_scaled = preprocess_data(x_train, x_test)
        
        min_test_size = min(x_test_scaled.shape[0], y_test.shape[0])
        x_test_scaled = x_test_scaled[:min_test_size]
        y_test = y_test[:min_test_size]
        
        X_test_dmatrix = DMatrix(x_test_scaled)
        accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test_dmatrix, y_test)
        
        print(f"Metrics for {classname}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        plot_confusion_matrix(conf_matrix, classname, class_labels)

        y_scores = model.predict(X_test_dmatrix)
        plot_roc_curve(y_test, y_scores, classname)
