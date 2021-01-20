import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import KFold
import plotly.offline as py
py.init_notebook_mode(connected=True)


def setup():
    pd.set_option('display.max_columns', 100)
    #pd.set_option('display.max_rows', None)


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


def pearson_correlation(data):
    colormap = plt.cm.viridis
    plt.figure(figsize=(12, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white',
                annot=True)
    plt.show()


def comparing_title_and_sex(train, original_train):
    print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum']))
    print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).agg(['mean', 'count', 'sum']))

    title_and_sex = original_train.copy()[['Name', 'Sex']]
    title_and_sex['Title'] = title_and_sex['Name'].apply(get_title)
    title_and_sex['Sex'] = title_and_sex['Sex'].map({'female': 0, 'male': 1}).astype(int)
    print(title_and_sex[['Title', 'Sex']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum']))


def titanic():
    train_data = 'titanic\\train.csv'
    test_data = 'titanic\\test.csv'

    train = pd.read_csv(train_data, sep=',')
    test = pd.read_csv(test_data, sep=',')
    full_data = [train, test]

    PassengerId = test['PassengerId']

    original_train = train.copy()

    for dataset in full_data:
        dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1


        # Fill null values in Embarked feature
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
        # Fill null values in the Fare feature
        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
        # Fill null values in the Age feature
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)

        # Get the title of the passenger
        dataset['Title'] = dataset['Name'].apply(get_title)
        # Group all non-common titles into one single grouping "Rare"
        dataset['Title'] = dataset['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        # Map Sex
        dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
        # Map Title
        title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
        # Mapping Embarked
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        # Mapping Fare
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)
        # Mapping Age
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    # Feature selection: remove variables no longer containing relevant information
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    train = train.drop(drop_elements, axis=1)
    test = test.drop(drop_elements, axis=1)

    #print(train.head(3))
    #pearson_correlation(train)
    #comparing_title_and_sex(train, original_train)

    return [(train, test), PassengerId]


def cross_validation(cv_data):
    train, test = cv_data
    cv = KFold(n_splits=10)  # Desired number of Cross Validation folds
    accuracies = list()
    max_attributes = len(list(test))
    depth_range = range(1, max_attributes + 1)

    # Testing max_depths from 1 to max attributes
    # Uncomment prints for details about each Cross Validation pass
    for depth in depth_range:
        fold_accuracy = []
        tree_model = tree.DecisionTreeClassifier(max_depth=depth)
        # print("Current max depth: ", depth, "\n")
        for train_fold, valid_fold in cv.split(train):
            f_train = train.loc[train_fold]  # Extract train data with cv indices
            f_valid = train.loc[valid_fold]  # Extract valid data with cv indices

            model = tree_model.fit(X=f_train.drop(['Survived'], axis=1),
                                   y=f_train["Survived"])  # We fit the model with the fold train data
            valid_acc = model.score(X=f_valid.drop(['Survived'], axis=1),
                                    y=f_valid["Survived"])  # We calculate accuracy with the fold validation data
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


def final_tree(final_data, PassengerId):
    train, test = final_data

    # Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
    y_train = train['Survived']
    x_train = train.drop(['Survived'], axis=1).values

    x_test = test.values

    # Create Decision Tree with max_depth = 3
    decision_tree = tree.DecisionTreeClassifier(max_depth=3)
    decision_tree.fit(x_train, y_train)

    # Predicting results for test dataset
    y_pred = decision_tree.predict(x_test)
    submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_pred
    })
    submission.to_csv('submission.csv', index=False)

    # Export our trained model as a .dot file
    dot_data = tree.export_graphviz(decision_tree,
                             out_file='tree1.dot',
                             max_depth=3,
                             impurity=True,
                             feature_names=list(train.drop(['Survived'], axis=1)),
                             class_names=['Died', 'Survived'],
                             rounded=True,
                             filled=True)

    acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
    print("Model accuracy ", acc_decision_tree)


if __name__ == '__main__':
    setup()
    [data, pid] = titanic()
    cross_validation(cv_data=data)
    final_tree(final_data=data, PassengerId=pid)
