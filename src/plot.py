import pandas as pd

def scatter_real_vs_pred_subplot(plt, y, pred_y, x_label='Real', y_label='Pred', title=None):
    plt.scatter(y, pred_y, c='b', s=40, alpha=0.5)
    plt.title(title if title else x_label + ' vs ' + y_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def scatter_residual_error_subplot(plt, y, pred_y, x_label='Real', y_label='Pred', title=None):
    plt.ylim(-1.2, 1.2)
    plt.xlim(-1.2, 1.2)
    plt.scatter(y, y - pred_y, c='b', s=40, alpha=0.5)
    plt.hlines(y=0, xmin=-1, xmax=1)
    plt.title(title if title else x_label + ' vs ' + y_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def subplots(plt, y, pred_y, x_label='Real', y_label='Pred', title=None):
    plt.ylim(-1.2, 1.2)
    plt.xlim(-1.2, 1.2)
    plt.scatter(y, y - pred_y, c='b', s=40, alpha=0.5)
    plt.hlines(y=0, xmin=-1, xmax=1)
    plt.title(title if title else x_label + ' vs ' + y_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def confusion_matrix(plt, y, pred_y, **kwargs):
    pd.crosstab(y, pred_y, rownames=['Real'], colnames=['Predição']).head(2)
