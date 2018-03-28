import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.

    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """

    # Create figure
    fig, ax = pl.subplots(2, 3, figsize=(30, 20))

    # Constants
    bar_width = 0.3
    colors = ['#A00000', '#00A0A0', '#00A000']

    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                # Creative plot code
                x1 = j // 3
                y1 = j % 3

                xs = i + k * bar_width
                # print(results)
                # print(learner)
                # print(i)
                # print(metric)
                # print('\n\n\n\n\n')
                ys = results[learner][str(i)][metric]
                curr_color = colors[k]

                ax[x1, y1].bar(xs, ys, width=bar_width, color=curr_color)
                ax[x1, y1].set_xticks([0.45, 1.45, 2.45])
                ax[x1, y1].set_xticklabels(["1%", "10%", "100%"])
                ax[x1, y1].set_xlabel("Training Set Size")
                ax[x1, y1].set_xlim((-0.1, 3.0))

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[0, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=learner))
    pl.legend(handles=patches, bbox_to_anchor=(-.80, 2.53), \
              loc='upper center', borderaxespad=0., ncol=3, fontsize='x-large')

    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize=16, y=1.10)
    pl.tight_layout()
    pl.show()


results = {"GaussianNB": {"0": {"train_time": 0.0027151107788085938, "pred_time": 0.03165102005004883, "acc_train": 0.4, "acc_test": 0.3517965726920951, "f_train": 0.3333333333333333, "f_test": 0.3101343466682625}, "1": {"train_time": 0.013508081436157227, "pred_time": 0.01782703399658203, "acc_train": 0.38, "acc_test": 0.36594803758982863, "f_train": 0.3239051094890511, "f_test": 0.3202198524360008}, "2": {"train_time": 0.10415506362915039, "pred_time": 0.017491817474365234, "acc_train": 0.5933333333333334, "acc_test": 0.5967938087341073, "f_train": 0.4125, "f_test": 0.42027854237705925}}, "SVC": {"0": {"train_time": 0.013877153396606445, "pred_time": 0.21414494514465332, "acc_train": 0.76, "acc_test": 0.7562189054726368, "f_train": 0.0, "f_test": 0.0}, "1": {"train_time": 0.8271317481994629, "pred_time": 1.7813868522644043, "acc_train": 0.83, "acc_test": 0.833167495854063, "f_train": 0.6590909090909092, "f_test": 0.6729876674181143}, "2": {"train_time": 82.24288702011108, "pred_time": 15.94149899482727, "acc_train": 0.85, "acc_test": 0.8362631288004422, "f_train": 0.7115384615384616, "f_test": 0.6722873097558305}}, "RandomForestClassifier": {"0": {"train_time": 0.019968032836914062, "pred_time": 0.018857240676879883, "acc_train": 0.9766666666666667, "acc_test": 0.802653399668325, "f_train": 0.9789156626506026, "f_test": 0.5870093782563389}, "1": {"train_time": 0.04376220703125, "pred_time": 0.01930093765258789, "acc_train": 0.98, "acc_test": 0.8322830292979547, "f_train": 0.9738372093023256, "f_test": 0.6620756967790914}, "2": {"train_time": 0.5484771728515625, "pred_time": 0.03662300109863281, "acc_train": 0.9566666666666667, "acc_test": 0.8378109452736319, "f_train": 0.9337349397590362, "f_test": 0.6717389128040845}}}
# results = {"GaussianNB": {"0": {"train_time": 0.002191781997680664, "pred_time": 0.017535924911499023, "acc_train": 0.4, "acc_test": 0.3517965726920951, "f_train": 0.3333333333333333, "f_test": 0.3101343466682625}, "1": {"train_time": 0.01227712631225586, "pred_time": 0.016387224197387695, "acc_train": 0.38, "acc_test": 0.36594803758982863, "f_train": 0.3239051094890511, "f_test": 0.3202198524360008}, "2": {"train_time": 0.0827639102935791, "pred_time": 0.013727903366088867, "acc_train": 0.5933333333333334, "acc_test": 0.5967938087341073, "f_train": 0.4125, "f_test": 0.42027854237705925}}, "DecisionTreeClassifier": {"0": {"train_time": 0.0022771358489990234, "pred_time": 0.004742860794067383, "acc_train": 1.0, "acc_test": 0.77191818684356, "f_train": 1.0, "f_test": 0.5359784216479596}, "1": {"train_time": 0.026633024215698242, "pred_time": 0.0044019222259521484, "acc_train": 0.9966666666666667, "acc_test": 0.8016583747927032, "f_train": 0.997191011235955, "f_test": 0.5938748335552595}, "2": {"train_time": 0.3714487552642822, "pred_time": 0.005837917327880859, "acc_train": 0.97, "acc_test": 0.8185737976782753, "f_train": 0.9638554216867471, "f_test": 0.627939142461964}}, "RandomForestClassifier": {"0": {"train_time": 0.015664100646972656, "pred_time": 0.013602018356323242, "acc_train": 0.9766666666666667, "acc_test": 0.802653399668325, "f_train": 0.9789156626506026, "f_test": 0.5870093782563389}, "1": {"train_time": 0.047672271728515625, "pred_time": 0.020420074462890625, "acc_train": 0.98, "acc_test": 0.8322830292979547, "f_train": 0.9738372093023256, "f_test": 0.6620756967790914}, "2": {"train_time": 0.5289590358734131, "pred_time": 0.03459000587463379, "acc_train": 0.9566666666666667, "acc_test": 0.8378109452736319, "f_train": 0.9337349397590362, "f_test": 0.6717389128040845}}}

accuracy = 0.2478
fscore = 0.2917


evaluate(results, accuracy, fscore)