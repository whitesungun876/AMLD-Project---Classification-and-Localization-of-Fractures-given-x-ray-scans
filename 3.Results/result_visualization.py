"""
Description: This scripts creates figures and tables for the results of the project.

Author: SEFA
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


####################################################################################
##### Initial configuration
####################################################################################

# define cwd and folder names
cwd = os.getcwd()
results_folder = os.path.join(cwd, '3.Results')
graphics_folder = os.path.join(results_folder, 'graphics')
pres_graphics_folder = os.path.join(cwd, '10. final presentation', 'plots')
tables_folder = os.path.join(results_folder, 'tables')
train_res_folder = os.path.join(results_folder, 'training_results')

# create folders only if they do not exist
os.makedirs(graphics_folder, exist_ok=True)
os.makedirs(tables_folder, exist_ok=True)

######################################################################################
##### Load data
######################################################################################

# YOLO results
df_yolo = pd.read_csv(os.path.join(train_res_folder, 'all_results_yolo.csv'))
df_rcnn_val = pd.read_csv(os.path.join(train_res_folder, 'train_results_rcnn.csv'))
df_rcnn_train = pd.read_csv(os.path.join(train_res_folder, 'val_results_rcnn.csv'))
df_rcnn_val_reg = pd.read_csv(os.path.join(train_res_folder, 'val_results_rcnn_reg.csv'))

##################################################################################
##### Define functions
##################################################################################

def plot_single(df, x, y, title, filename=None):
    fig, ax = plt.subplots(figsize=(6,4))
    #marker color is yelow
    ax.plot(df[x], df[y],
            marker='o', markerfacecolor = "orange", markeredgecolor = "orange",
            color = "blue", linestyle='-')
    #set title
    ax.set_title(title)
    ax.grid(True)
    if filename:
        # plt.savefig(os.path.join(graphics_folder, filename), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(pres_graphics_folder, filename), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def gen_yolo_dfs(df_yolo):
    """"
    Transforms df_yolo into three dataframes:
    df_yolo_train, df_yolo_val, df_yolo_metrics
    """
    yolo_col = df_yolo.columns.tolist()
    # save elements that start with "train/" in a list
    train_col = [col for col in yolo_col if col.startswith("train/")]
    val_col = [col for col in yolo_col if col.startswith("val/")]
    metrics_col = [col for col in yolo_col if col.startswith("metrics/")]

    df_yolo_train = df_yolo[["epoch"]+train_col]
    df_yolo_val = df_yolo[["epoch"]+val_col]
    df_yolo_metrics = df_yolo[["epoch"]+metrics_col]

    # remove (B) from the column names
    df_yolo_metrics.columns = df_yolo_metrics.columns.str.replace("metrics/", "")
    df_yolo_metrics.columns = df_yolo_metrics.columns.str.replace("(B)", "")
    df_yolo_train.columns = df_yolo_train.columns.str.replace("train/", "")
    df_yolo_val.columns = df_yolo_val.columns.str.replace("val/", "")

    #add total loss to df_yolo_train and df_yolo_val
    df_yolo_train["total_loss"] = df_yolo_train[["box_loss", "cls_loss", "dfl_loss"]].sum(axis=1)
    df_yolo_val["total_loss"] = df_yolo_val[["box_loss", "cls_loss", "dfl_loss"]].sum(axis=1)

    return df_yolo_train, df_yolo_val, df_yolo_metrics

def plot_multiple(df, x, y:list, title:list, filename=None):
    
    fig, axes = plt.subplots(nrows = 2, ncols=y/2, figsize=(12, 8))

    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.plot(df[x], df[y[i]], marker='o', linestyle='-', markerfacecolor = "orange",
                color = "blue")
        ax.set_title(title[i])

    plt.tight_layout()
    if filename:
        # plt.savefig(os.path.join(graphics_folder, filename), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(graphics_folder, filename), dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_val_train(
        df_train, df_val, x, y:list, title:list,
        n_rows = 1, n_cols = 4, filename = None, suptitle = None
):

    fig, axes = plt.subplots(nrows = n_rows, ncols=n_cols, figsize=(12, 4))

    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.plot(df_train[x], df_train[y[i]], marker='o', linestyle='-', markerfacecolor = "orange",
                color = "blue", label = "train")
        ax.plot(df_val[x], df_val[y[i]], marker='o', linestyle='-', markerfacecolor = "orange",
                color = "red", label = "val")
        ax.set_title(title[i])
        ax.legend()
    if suptitle:
        plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    if filename:
        # plt.savefig(os.path.join(graphics_folder, filename), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(pres_graphics_folder, filename), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def plot_val_only(
        df_val, x, y:list, title:list,
        n_rows = 1, n_cols = 4, filename = None,
        suptitle = None
):

    fig, axes = plt.subplots(nrows = n_rows, ncols=n_cols, figsize=(12, 4))

    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.plot(df_val[x], df_val[y[i]], marker='o', linestyle='-', markerfacecolor = "orange",
                color = "red", label = "val")
        ax.set_title(title[i])
        ax.legend()

    if suptitle:
        plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    if filename:
        # plt.savefig(os.path.join(graphics_folder, filename), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(pres_graphics_folder, filename), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def plot_train_metrics(
        df_train, df_val, x, y: list, title: list,
        df_metrics, metrics_y: list, metrics_title: list,
        filename=None, suptitle=None
):
    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 4))
    axes = axes.flatten()

    # First row: df_train and df_val
    for i in range(len(y)):
        ax = axes[i]
        ax.plot(df_train[x], df_train[y[i]], marker='o', linestyle='-', markerfacecolor="orange",
                color="blue", label="train")
        ax.plot(df_val[x], df_val[y[i]], marker='o', linestyle='-', markerfacecolor="orange",
                color="red", label="val")
        ax.set_title(title[i])
        ax.legend()

    # Second row: df_metrics
    for i in range(len(metrics_y)):
        ax = axes[i + n_cols]
        ax.plot(df_metrics[x], df_metrics[metrics_y[i]], marker='o', linestyle='-', markerfacecolor="green",
                color="purple")
        ax.set_title(metrics_title[i])

    # Hide unused axes if any
    for j in range(len(y) + len(metrics_y), len(axes)):
        axes[j].axis("off")

    if suptitle:
        plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    if filename:
        # plt.savefig(os.path.join(graphics_folder, filename), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(pres_graphics_folder, filename), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def main():

    ## load data
    # YOLO results
    df_yolo = pd.read_csv(os.path.join(train_res_folder, 'all_results_yolo.csv'))

    # RCNN results
    df_rcnn_train = pd.read_csv(os.path.join(train_res_folder, 'train_results_rcnn.csv'))
    df_rcnn_val = pd.read_csv(os.path.join(train_res_folder, 'val_results_rcnn.csv'))

    # RCNN regularized results (validation only)
    df_rcnn_val_reg = pd.read_csv(os.path.join(train_res_folder, 'val_results_rcnn_reg.csv'))
    df_rcnn_val_reg.drop(["mAP50-95"], axis=1, inplace=True)
    df_rcnn_val_reg.rename(columns={"mAP50": "mAP"}, inplace=True)

    # Convert YOLO dataframes
    df_yolo_train, df_yolo_val, df_yolo_metrics = gen_yolo_dfs(df_yolo)

    # plot the training and validation loss for YOLO
    plot_val_train(
        df_yolo_train, df_yolo_val,
        "epoch", ['box_loss', 'cls_loss', 'dfl_loss', 'total_loss'],
        ["Box loss", "Classification loss", "dfl loss", "Total loss"],
        filename="yolo_training_losses",
        suptitle="YOLOv11 learning curves"
    )
    plt.close()
    # plot losses and metrics for RCNN
    plot_val_train(
        df_rcnn_train, df_rcnn_val,
        "epoch",
        ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "loss_total", "mAP", "precision", "recall"],
        ["Classifier loss", "Box regression loss", "Objectness loss",
        "RPN box regression loss", "Total loss", "mAP", "Precision", "Recall"],
        n_rows = 2,
        filename="rcnn_training",
        suptitle="Faster RCNN learning curves"
    )
    
    plt.close()
    # Plot validation results for RCNN regularized
    plot_val_only(
        df_rcnn_val_reg,
        "epoch",
        ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "loss_total", "mAP", "precision", "recall"],
        ["Classifier loss", "Box regression loss", "Objectness loss",
        "RPN box regression loss", "Total loss", "mAP", "Precision", "Recall"],
        n_rows = 2,
        filename="rcnn_training_reg",
        suptitle="Faster RCNN learning curves (regularized)"
    )

    # plot losses and metrics for YOLO
    plot_train_metrics(
        df_yolo_train, df_yolo_val,
        "epoch",
        ['box_loss', 'cls_loss', 'dfl_loss', 'total_loss'],
        ["Box loss", "Classification loss", "dfl loss", "Total loss"],
        df_yolo_metrics,
        ['mAP50', 'mAP50-95', 'precision', 'recall'],
        ["mAP50", "mAP50-95", "Precision", "Recall"],
        filename="yolo_training",
        suptitle="YOLOv11 learning curves"
    )

    main_done = "Main function executed successfully"

    return print(main_done)


#####################################################################################
##### Main script
#####################################################################################

# run the main function
if __name__ == "__main__":
    main()

print("Script executed successfully")