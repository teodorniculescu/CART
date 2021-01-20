import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import plotly.offline as py
py.init_notebook_mode(connected=True)


def pearson_correlation(data):
    colormap = plt.cm.viridis
    plt.figure(figsize=(12, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white',
                annot=True)
    plt.show()


def setup():
    pd.set_option('display.max_columns', 100)
    #pd.set_option('display.max_rows', None)


def plot_one_column(data):
    what_column = 'flyash'
    data[what_column].plot()
    plt.xlabel('index')
    plt.ylabel(what_column)
    plt.show()


def concrete():
    csv_data_path = 'concrete\\data.csv'
    data = pd.read_csv(csv_data_path, sep=',')

    #print(data.head())
    #print("Contains null ", data.isnull().values.any())
    #pearson_correlation(data)


    # suffle data
    data = shuffle(data)
    data.reset_index(inplace=True, drop=True)

    # get training and testing data
    number_of_rows = len(data.index)
    split_limit = int(number_of_rows / 10)
    test = data[:split_limit]
    test.reset_index(inplace=True, drop=True)
    train = data[split_limit:]
    train.reset_index(inplace=True, drop=True)

    #print(train['cement'].agg(['mean', 'count', 'min', 'max']))
    pearson_correlation(train)

    return train, test


def final_tree(final_data, max_depth):
    train, test = final_data

    compare_column = 'csMPa'
    # Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
    y_train = train[compare_column]
    x_train = train.drop([compare_column], axis=1).values

    y_test = test[compare_column]
    x_test = test.drop([compare_column], axis=1).values

    # Create Decision Tree with max_depth = 3
    decision_tree = tree.DecisionTreeRegressor(max_depth=max_depth)
    decision_tree.fit(x_train, y_train)

    # Export our trained model as a .dot file
    dot_data = tree.export_graphviz(decision_tree,
                                    out_file='tree2.dot',
                                    max_depth=max_depth,
                                    impurity=True,
                                    feature_names=list(train.drop([compare_column], axis=1)),
                                    rounded=True,
                                    filled=True)

    acc_decision_tree = round(decision_tree.score(x_test, y_test) * 100, 2)
    print("Accuracy on testing data ", acc_decision_tree)
    return acc_decision_tree


def cross_validation(cv_data):
    train, test = cv_data
    cv = KFold(n_splits=10)  # Desired number of Cross Validation folds
    accuracies = list()
    max_attributes = len(list(test))
    depth_range = range(1, max_attributes + 1)

    validation_column = 'csMPa'

    # Testing max_depths from 1 to max attributes
    # Uncomment prints for details about each Cross Validation pass
    for depth in depth_range:
        fold_accuracy = []
        tree_model = tree.DecisionTreeRegressor(max_depth=depth)
        # print("Current max depth: ", depth, "\n")
        for train_fold, valid_fold in cv.split(train):
            f_train = train.loc[train_fold]  # Extract train data with cv indices
            f_valid = train.loc[valid_fold]  # Extract valid data with cv indices

            model = tree_model.fit(X=f_train.drop([validation_column], axis=1),
                                   y=f_train[validation_column])  # We fit the model with the fold train data
            valid_acc = model.score(X=f_valid.drop([validation_column], axis=1),
                                    y=f_valid[validation_column])  # We calculate accuracy with the fold validation data
            fold_accuracy.append(valid_acc)

        avg = sum(fold_accuracy) / len(fold_accuracy)
        accuracies.append(avg)
        # print("Accuracy per fold: ", fold_accuracy, "\n")
        # print("Average accuracy: ", avg)
        # print("\n")

    # Just to show results conveniently
    df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
    df = df[["Max Depth", "Average Accuracy"]]
    print(df.to_string(index=False))
    idx = df["Average Accuracy"].argmax()
    return df.iloc[int(idx)]['Max Depth']



if __name__ == '__main__':
    final_sum = 0.0
    total_runs = 1
    for run in range(0, total_runs):
        #print('now at ', run)
        setup()
        data = concrete()
        optimal_depth = cross_validation(cv_data=data)
        print("The optimal depth is ", optimal_depth)
        final_sum += final_tree(final_data=data, max_depth=optimal_depth)
    #print('final', final_sum, total_runs, final_sum/total_runs)


