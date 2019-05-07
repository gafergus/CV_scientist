import gin
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, classification_report

@gin.configurable
def loss_vs_Epochs(history, save_figs=False, model_name=None):
    history_dict = history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc = history_dict['acc']
    epochs = range(1, len(acc) + 1)
    history_dict.keys()
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss' + '_' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if save_figs:
        plt.savefig(str(model_name) + '.svg')
    plt.show()

@gin.configurable
def acc_vs_Epochs(history, save_figs=False, model_name=None):
    history_dict = history
    acc_values = history_dict['acc']
    epochs = range(1, len(acc_values) + 1)
    val_acc_values = history_dict['val_acc']
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy' + '_' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    if save_figs:
        plt.savefig(str(model_name) + '.svg')
    plt.show()

@gin.configurable
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', save_figs=False, model_name=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig_title = title + '_' + model_name

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(fig_title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_figs:
        plt.savefig(str(model_name) + '.svg')
    plt.show()

@gin.configurable
def binary_metrics(ground, pred, class_names=None, history=None, save_figs=False, model_name=None):
    predicted_class = np.argmax(pred, axis=1)
    class_rep = classification_report(ground, predicted_class)
    cnf_matrix = confusion_matrix(ground, predicted_class)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title='Confusion matrix',
                          save_figs=save_figs, model_name=model_name)
    acc_vs_Epochs(history, save_figs=save_figs, model_name=model_name)
    loss_vs_Epochs(history, save_figs=save_figs, model_name=model_name)
    sens = recall_score(ground, predicted_class, pos_label=1)
    spec = recall_score(ground, predicted_class, pos_label=0)
    roc_auc = roc_auc_score(ground, pred[:,1])
    small_value_to_avoid_div_by_zero = 0.0000001
    FP = cnf_matrix[0][1]
    FN = cnf_matrix[1][0]
    TP = cnf_matrix[0][0]
    TN = cnf_matrix[1][1]
    PPV = TP / (TP + FP + small_value_to_avoid_div_by_zero)
    NPV = TN / (TN + FN + small_value_to_avoid_div_by_zero)
    FPR = FP / (FP + TN + small_value_to_avoid_div_by_zero)
    FNR = FN / (TP + FN + small_value_to_avoid_div_by_zero)
    return sens, spec, roc_auc, class_rep, TP, TN, FP, FN, PPV, NPV, FPR, FNR

@gin.configurable
def multi_metrics(ground, pred, class_names=None, history=None, save_figs=False, model_name=None):
    predicted_class = np.argmax(pred, axis=1)
    class_rep = classification_report(ground, predicted_class)
    cnf_matrix = confusion_matrix(ground, predicted_class)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title='Confusion matrix',
                          save_figs=save_figs, model_name=model_name)
    acc_vs_Epochs(history, save_figs=save_figs, model_name=model_name)
    loss_vs_Epochs(history, save_figs=save_figs, model_name=model_name)
    small_value_to_avoid_div_by_zero = 0.0000001
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    PPV = TP / (TP + FP + small_value_to_avoid_div_by_zero)
    NPV = TN / (TN + FN + small_value_to_avoid_div_by_zero)
    FPR = FP / (FP + TN + small_value_to_avoid_div_by_zero)
    FNR = FN / (TP + FN + small_value_to_avoid_div_by_zero)

    return class_rep, TP, TN, FP, FN, PPV, NPV, FPR, FNR
