# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

# Load the data
df = pd.read_csv("heart.csv")
df.head()


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


def plot_box():
    columns = [
        "Age",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "MaxHR",
        "Oldpeak",
    ]

    for column in columns:
        plt.figure()
        sns.boxplot(
            data=df,
            y=column,
            x="HeartDisease",
        )


def checkNull():
    df.isnull().sum()
    # no null value


def normalization():
    return (df - df.min()) / (df.max() - df.min())


def split_dataset(df):
    # Split the dataset - train test split
    x = df.iloc[:, 0:11]
    y = df.iloc[:, 11]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    return X_train, X_test, y_train, y_test


def logistic_regression(dfEncode):
    X_train, X_test, y_train, y_test = split_dataset(dfEncode)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    
    #print("the score:", score)

    #print("X_test", X_test)
    #y_pred = clf.predict(X_test)
    #print(y_pred)
    return clf


def predict(clf, d):
    y_pred = clf.predict(d)

    if(y_pred[0] == 1):
        print("Heart Failure!!")
    else:
        print("Save!")


def encoding():
    df_encode = df.apply(LabelEncoder().fit_transform)
    df_encode.head()
    # df = df_encode
    # print("df_encode", df_encode)
    return df_encode


def decision_tree():
    X_train, X_test, y_train, y_test = split_dataset()
    # Build the tree
    df = DecisionTreeClassifier(
        min_samples_split=20, criterion="entropy", random_state=42
    )
    # Fit the training data
    df.fit(X_train, y_train)


def clustering():
    data_norm = normalization()
    model = KMeans(n_clusters=6)  # 6 clusters?
    model.fit(data_norm)
    print(model.labels_)
    data_norm["clust"] = pd.Series(model.labels_)
    data_norm.head()
    print(model.cluster_centers_)
    print(model.inertia_)
    plt.figure()
    plt.xlabel("clust")
    plt.ylabel("number")
    plt.hist(data_norm["clust"])


if __name__ == "__main__":
    # dataDetail()
    # pairwise()
    # histogram()
    # checkNull()
    # plot_box()
    dfEncode = encoding()
    d = [
        {
            "Age": 29,
            "Sex": 1,
            "ChestPainType": 0,
            "RestingBP": 19,
            "Cholesterol": 0,
            "FastingBS": 1,
            "RestingECG": 2,
            "MaxHR": 60,
            "ExerciseAngina": 0,
            "Oldpeak": 45,
            "ST_Slope": 2
        }
    ]
    df = pd.DataFrame(d)
    clf = logistic_regression(dfEncode)
    predict(clf, df)

# %%
