import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, roc_auc_score
from itertools import cycle
import warnings
from sklearn import metrics
warnings.filterwarnings('ignore')


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'SCO-ATF-SCBAM', 'SAA-ATF-SCBAM', 'ECO-ATF-SCBAM', 'FOA-ATF-SCBAM', 'URP-FO-ATF-SCBAM']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
    for j in range(len(Algorithm) - 1):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    length = np.arange(Fitness.shape[1])
    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence Curve')
    plt.plot(length, Fitness[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label=Algorithm[1])
    plt.plot(length, Fitness[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
             markersize=12, label=Algorithm[2])
    plt.plot(length, Fitness[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label=Algorithm[3])
    plt.plot(length, Fitness[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label=Algorithm[4])
    plt.plot(length, Fitness[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label=Algorithm[5])
    plt.xlabel('No. of Iteration', fontname="Arial", fontsize=12, fontweight='bold', color='k')
    plt.ylabel('Cost Function', fontname="Arial", fontsize=12, fontweight='bold', color='k')
    plt.legend(loc=1, prop={'weight': 'bold'})
    plt.savefig("./Results/Conv.png")
    plt.show()


def Plot_ROC_Curve():
    lw = 3
    cls = ['CNN', 'ResNet50', 'm-GAN', 'SCBAM', 'URP-FO-ATF-SCBAM']
    Actual = np.load('Target.npy', allow_pickle=True)
    lenper = round(Actual.shape[0] * 0.75)
    Actual = Actual[lenper:, :]
    colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
    fig = plt.figure()
    fig.canvas.manager.set_window_title('ROC Curve')
    for i, color in zip(range(5), colors):  # For all classifiers
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc * 100
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            lw=2,
            label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Accuracy')
    plt.xlabel("False Positive Rate", fontname="Arial", fontsize=11, fontweight='bold', color='k')
    plt.ylabel("True Positive Rate", fontname="Arial", fontsize=11, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=11, fontweight='bold', color='#1d3557')
    plt.yticks(fontname="Arial", fontsize=11, fontweight='bold', color='#1d3557')
    plt.title("ROC Curve")
    plt.legend(loc="lower right", prop={'weight': 'bold'})
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def Table():
    eval = np.load('Evaluate.npy', allow_pickle=True)
    Algorithm = ['Kfold', 'SCO-ATF-SCBAM', 'SAA-ATF-SCBAM', 'ECO-ATF-SCBAM', 'FOA-ATF-SCBAM', 'URP-FO-ATF-SCBAM']
    Classifier = ['Kfold', 'CNN', 'ResNet50', 'm-GAN', 'SCBAM', 'URP-FO-ATF-SCBAM']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 2, 5, 8, 12, 15, 16]).astype(int)
    Table_Terms = [0, 2, 5, 8, 12, 15, 16]
    table_terms = [Terms[i] for i in Table_Terms]
    Kfold = ['Kfold 1', 'Kfold 2', 'Kfold 3', 'Kfold 4', 'Kfold 5']
    for k in range(len(Table_Terms)):
        value = eval[:, :, 4:]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Kfold)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
        print('-------------------------------- ', table_terms[k], '  Algorithm Comparison',
              '-------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Kfold)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
        print('-------------------------------', table_terms[k], '  Classifier Comparison',
              '-----------------------------------')
        print(Table)


def Plots_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    kfold = [1, 2, 3, 4]
    bar_width = 0.15
    Algorithm = ['SCO-ATF-SCBAM', 'SAA-ATF-SCBAM', 'ECO-ATF-SCBAM', 'FOA-ATF-SCBAM', 'URP-FO-ATF-SCBAM']
    Classifier = ['CNN', 'ResNet50', 'm-GAN', 'SCBAM', 'URP-FO-ATF-SCBAM']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Algorithm Comparison of Epochs')
            X = np.arange(len(kfold))
            bar1 = plt.bar(X + 0.00, Graph[:, 0], color='m', edgecolor='k', width=0.15, label=Algorithm[0])
            ax.bar_label(container=bar1, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 0]],
                         fontweight='bold', padding=5)
            bar2 = plt.bar(X + 0.15, Graph[:, 1], color='y', edgecolor='k', width=0.15, label=Algorithm[1])
            ax.bar_label(container=bar2, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 1]],
                         fontweight='bold', padding=5)
            bar3 = plt.bar(X + 0.30, Graph[:, 2], color='#9b5de5', edgecolor='k', width=0.15, label=Algorithm[2])
            ax.bar_label(container=bar3, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 2]],
                         fontweight='bold', padding=5)
            bar4 = plt.bar(X + 0.45, Graph[:, 3], color='#218380', edgecolor='k', width=0.15, label=Algorithm[3])
            ax.bar_label(container=bar4, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 3]],
                         fontweight='bold', padding=5)
            bar5 = plt.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='k', width=0.15, label=Algorithm[4])
            ax.bar_label(container=bar5, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 4]],
                         fontweight='bold', padding=5)
            plt.xticks(X + bar_width * 2, ['20', '40', '60', '80'], fontname="Arial",
                       fontsize=12,
                       fontweight='bold', color='k')

            plt.xlabel('No. of Epochs', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
                      ncol=5, fontsize=10, fancybox=True, shadow=False)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            path = "./Results/%s_Alg_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Method Comparison of Epochs')
            X = np.arange(len(kfold))
            bar1 = plt.bar(X + 0.00, Graph[:, 5], color='yellowgreen', edgecolor='k', width=0.15, label=Classifier[0])
            ax.bar_label(container=bar1, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 5]],
                         fontweight='bold', padding=5)
            bar2 = plt.bar(X + 0.15, Graph[:, 6], color='gold', edgecolor='k', width=0.15, label=Classifier[1])
            ax.bar_label(container=bar2, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 6]],
                         fontweight='bold', padding=5)
            bar3 = plt.bar(X + 0.30, Graph[:, 7], color='mediumpurple', edgecolor='k', width=0.15, label=Classifier[2])
            ax.bar_label(container=bar3, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 7]],
                         fontweight='bold', padding=5)
            bar4 = plt.bar(X + 0.45, Graph[:, 8], color='sandybrown', edgecolor='k', width=0.15, label=Classifier[3])
            ax.bar_label(container=bar4, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 8]],
                         fontweight='bold', padding=5)
            bar5 = plt.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='k', width=0.15, label=Classifier[4])
            ax.bar_label(container=bar5, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 4]],
                         fontweight='bold', padding=5)
            plt.xticks(X + bar_width * 2, ['20', '40', '60', '80'], fontname="Arial",
                       fontsize=12,
                       fontweight='bold', color='k')
            plt.xlabel('No. of Epochs', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
                      ncol=5, fontsize=10, fancybox=True, shadow=False)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            path = "./Results/%s_Mod_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def plot_seg_results():
    Eval_all = np.load('Evaluate_Seg_all.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = ['TERMS', 'SCO-AHDRAUNet++', 'SAA-AHDRAUNet++', 'ECO-AHDRAUNet++', 'FOA-AHDRAUNet++', 'URP-FO-AHDRAUNet++']
    Methods = ['TERMS', 'Unet', 'Unet3+', 'ResUnet', 'UNet++', 'URP-FO-AHDRAUNet++']
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'Precision', 'F1-Score', 'Sensitivity', 'Specificity', 'FPR', 'FNR',
             'NPV',
             'FDR', 'MCC']
    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Statistics[1::3])
            for k in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[k + 1], stats[i, k, 1::3])
            print('-------------------------------------------------- ', Terms[i - 4],
                  'Algorithm Comparison', '--------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Methods[0], Statistics[1::3])
            Table.add_column(Methods[1], stats[i, 5, 1::3])
            Table.add_column(Methods[2], stats[i, 6, 1::3])
            Table.add_column(Methods[3], stats[i, 7, 1::3])
            Table.add_column(Methods[4], stats[i, 8, 1::3])
            Table.add_column(Methods[5], stats[i, 4, 1::3])
            print('-------------------------------------------------- ', Terms[i - 4],
                  'Method Comparison', '--------------------------------------------------')
            print(Table)

            X = np.arange(len(Statistics) - 1)
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            fig.canvas.manager.set_window_title(str(Terms[i - 4]) + 'Algorithm Comparison')
            ax.bar(X + 0.00, stats[i, 0, :-1], color='#f77f00', edgecolor='w', width=0.15, label=Algorithm[1])
            ax.bar(X + 0.15, stats[i, 1, :-1], color='#52796f', edgecolor='w', width=0.15, label=Algorithm[2])
            ax.bar(X + 0.30, stats[i, 2, :-1], color='#4361ee', edgecolor='w', width=0.15, label=Algorithm[3])
            ax.bar(X + 0.45, stats[i, 3, :-1], color='#ff0054', edgecolor='w', width=0.15, label=Algorithm[4])
            ax.bar(X + 0.60, stats[i, 4, :-1], color='k', edgecolor='w', width=0.15, label=Algorithm[5])
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['#f77f00', '#52796f', '#4361ee', '#ff0054', 'k']]
            plt.legend(dot_markers,
                       ['SCO-AHDRAUNet++', 'SAA-AHDRAUNet++', 'ECO-AHDRAUNet++', 'FOA-AHDRAUNet++', 'URP-FO-AHDRAUNet++'],
                       loc='upper center',
                       bbox_to_anchor=(0.5, 1.20), fontsize=10,
                       frameon=False, ncol=3)

            # Remove axes outline
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            plt.xticks(X + 0.30, ('BEST', 'WORST', 'MEAN', 'MEDIAN'), fontname="Arial", fontsize=12, fontweight='bold',
                       color='#35530a')
            plt.xlabel('Statisticsal Analysis', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[i - 4], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            path = "./Results/%s_Seg_Alg_bar.png" % (Terms[i - 4])
            plt.savefig(path)
            plt.show()

            X = np.arange(len(Statistics) - 1)
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            fig.canvas.manager.set_window_title(str(Terms[i - 4]) + 'Method Comparison')
            ax.bar(X + 0.00, stats[i, 5, :-1], color='#ff1654', edgecolor='w', width=0.15, label=Methods[1])
            ax.bar(X + 0.15, stats[i, 6, :-1], color='#8ac926', edgecolor='w', width=0.15, label=Methods[2])
            ax.bar(X + 0.30, stats[i, 7, :-1], color='#9e2a2b', edgecolor='w', width=0.15, label=Methods[3])
            ax.bar(X + 0.45, stats[i, 8, :-1], color='#197278', edgecolor='w', width=0.15, label=Methods[4])
            ax.bar(X + 0.60, stats[i, 4, :-1], color='k', edgecolor='w', width=0.15, label=Methods[5])
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['#ff1654', '#8ac926', '#9e2a2b', '#197278', 'k']]
            plt.legend(dot_markers, ['Unet', 'Unet3+', 'ResUnet', 'UNet++', 'URP-FO-AHDRAUNet++'], loc='upper center',
                       bbox_to_anchor=(0.5, 1.20), fontsize=10,
                       frameon=False, ncol=3)

            # Remove axes outline
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            plt.xticks(X + 0.30, ('BEST', 'WORST', 'MEAN', 'MEDIAN'), fontname="Arial", fontsize=12, fontweight='bold',
                       color='#35530a')
            plt.xlabel('Statisticsal Analysis', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[i - 4], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            path = "./Results/%s_Seg_Mod_bar.png" % (Terms[i - 4])
            plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    plotConvResults()
    Plot_ROC_Curve()
    Plots_Results()
    Table()
    plot_seg_results()
