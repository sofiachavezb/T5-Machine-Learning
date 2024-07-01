import numpy as np
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    num_classes = len(set(y_true))
    matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        matrix[y_true[i], y_pred[i]] += 1
    
    return matrix
    