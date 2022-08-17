import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load the data
df = pd.read_csv("heart.csv")
df.head()

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

def dataDetail():
    print("Column name: \n{0}".format(df.columns.values))
    print("\nColumns: \n{0}".format(df.shape[1]))
    print("\nRows: \n{0}".format(df.shape[0]))
    print("\nMain statistics: \n{0}".format(df.describe()))
    print("\nTypes of columns: \n{0}".format(df.dtypes))
    print("\nFirst five records: \n{0}".format(df.head(5)))


def pairwise():
    plt.figure(figsize=(15, 10))
    sns.pairplot(df, hue="HeartDisease")
    plt.title("Looking for Insites in Data")
    plt.legend("HeartDisease")
    plt.tight_layout()
    plt.plot()


def histogram():
    columns = [
        "Age",
        "Sex",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "RestingECG",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
        "ST_Slope",
    ]

    for column in columns:
        plt.figure()
        sns.histplot(
            data=df,
            x=column,
            hue="HeartDisease",
            multiple="dodge",
            shrink=0.8,
        )


def feature_histgram(feature):
    plt.figure()
    sns.histplot(
        data=df,
        x=feature,
        hue="HeartDisease",
        multiple="dodge",
        shrink=0.8,
    )


def plot_box():
    columns = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "MaxHR",
        "Oldpeak",
    ]
    plt.figure(figsize=(20, 20))
    for index, column in enumerate(columns):
        plt.subplot(4, 3, index + 1)
        sns.boxplot(
            data=df,
            y=column,
            x="HeartDisease",
        )


def correlation_matrix():
    plt.figure(figsize=(16, 6))
    matrix = np.triu(df.corr())
    heatmap = sns.heatmap(
        df.corr(),
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        mask=matrix,
    )
    heatmap.set_title(
        "Correlation Heatmap",
        fontdict={"fontsize": 12},
        pad=12,
    )


def checkNull():
    print(df.isnull().sum())
    # no null value


def normalization(df):
    return (df - df.min()) / (df.max() - df.min())


def split_dataset(df):
    # Split the dataset - train test split
    x = df.iloc[:, 0:11]
    y = df.iloc[:, 11]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    return X_train, X_test, y_train, y_test


def logistic_regression(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # clf.predict(X_test)
    score = clf.score(X_test, y_test)

    print("logistic_regression score:", score)

    # print("X_test", X_test)
    # y_pred = clf.predict(X_test)
    # print("y_pred", y_pred)
    return clf


def predict(clf, d):
    y_predict = clf.predict(d)
    y_predict_proba = clf.predict_proba(d)[:, 1]
    # print(y_pred)
    print("===============logistic=====================")
    for predict in y_predict:
        if predict == 1:
            print("Heart Failure!!")
        else:
            print("Save!")
    for proba in y_predict_proba:
        print(f"Percentage of patient will have a HeartDisease: {proba:.2%}")
   
def knn_predict(X_train, y_train, X_test, y_test, d):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_predict = knn.predict(d)
    knn_predict_proba = knn.predict_proba(d)[:, 1]
    
    print("===============KNN=====================")
    score = knn.score(X_test, y_test)
    print("KNN score:", score)
    
    for predict in knn_predict:
        if predict == 1:
            print("Heart Failure!!")
        else:
            print("Save!")
    
    for proba in knn_predict_proba:
        print(f"Percentage of patient will have a HeartDisease: {proba:.0%}")


def handlingOutlier():
    # print("========================================")
    # feature_histgram('Cholesterol')
    df["Cholesterol"].replace(0, df["Cholesterol"].median(), inplace=True)
    # print("df['Cholesterol']")
    # print(df['Cholesterol'])
    # feature_histgram('Cholesterol')
    # print("========================================")


def encoding():
    df_encode = df.apply(LabelEncoder().fit_transform)
    df_encode.head()
    return df_encode


def decision_tree(df):
    X_train, X_test, y_train, y_test = split_dataset(df)
    target_name = "HeartDisease"
    feature_names=df.columns.to_list()
    feature_names.remove(target_name)
        
    # Build the tree
    dt = DecisionTreeClassifier(min_samples_split=30, criterion="entropy", random_state=42)
    # Fit the training data
    dt.fit(X_train, y_train)
    plt.figure(figsize=(100, 100))
    tree.plot_tree(dt,feature_names=feature_names,  
                          class_names=['0','1'], filled ='True')

    testY_predict = dt.predict(X_test)
    testY_scores = accuracy_score(y_test, testY_predict)
    print("Decision Tree Scores: ", testY_scores)

    score = np.mean(cross_val_score(dt, X_train, y_train, cv=10))
    print("Decision Tree cross_val_score:", score)


def svm(X_train, X_test, y_train, y_test):
    model1 = SVC(kernel="rbf", random_state=0, probability=True)
    model1.fit(X_train, y_train)
    print("accurancy score of linear :", model1.score(X_test, y_test))

    model3 = SVC(kernel="rbf", random_state=0, probability=True, C=3)
    model3.fit(X_train, y_train)
    print("accurancy score of rbf c=3:", model3.score(X_test, y_test))


def clustering(df):
    model = KMeans(n_clusters=6)  # 6 clusters?
    model.fit(df)
    print(model.labels_)
    df["clust"] = pd.Series(model.labels_)
    df.head()
    print(model.cluster_centers_)
    print(model.inertia_)
    plt.figure()
    plt.xlabel("clust")
    plt.ylabel("number")
    plt.hist(df["clust"])


if __name__ == "__main__":
    dataDetail()
    pairwise()
    histogram()
    plot_box()
    checkNull()
    handlingOutlier()
    dfEncode = MultiColumnLabelEncoder(columns = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']).fit_transform(df)
    # dfEncode = encoding()
    df_norm = normalization(dfEncode)
    X_train, X_test, y_train, y_test = split_dataset(df_norm)
    clf = logistic_regression(X_train, X_test, y_train, y_test)

    # clustering(df_norm)
    decision_tree(dfEncode)
    svm(X_train, X_test, y_train, y_test)

    # predict
    d = [
        {
            "Age": 19,
            "Sex": 1,
            "ChestPainType": 3,
            "RestingBP": 120,
            "Cholesterol": 0,
            "FastingBS": 0,
            "RestingECG": 0,
            "MaxHR": 122,
            "ExerciseAngina": 0,
            "Oldpeak": 1,
            "ST_Slope": 0,
        },
        {
            "Age": 80,
            "Sex": 0,
            "ChestPainType": 0,
            "RestingBP": 39,
            "Cholesterol": 72,
            "FastingBS": 0,
            "RestingECG": 1,
            "MaxHR": 150,
            "ExerciseAngina": 1,
            "Oldpeak": 3,
            "ST_Slope": 1,
        },
        {
            "Age": 80,
            "Sex": 1,
            "ChestPainType": 0,
            "RestingBP": 45,
            "Cholesterol": 51,
            "FastingBS": 1,
            "RestingECG": 1,
            "MaxHR": 67,
            "ExerciseAngina": 0,
            "Oldpeak": 42,
            "ST_Slope": 1,
        },  # dead!
    ]
    df_test = pd.DataFrame(d)
    predict(clf, df_test)
    knn_predict(X_train, y_train, X_test, y_test, df_test)
