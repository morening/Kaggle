#-*-coding:utf-8-*-
# @Time    : 2018/3/28 下午8:42
# @Author  : morening
# @File    : titanic.py
# @Software: PyCharm


# train.csv： Age缺失少量数据（714/891）；Cabin缺失较大（204/891）
# test.csv: Age缺失数据相对少些（332/418）；Cabin缺失较大（91/418）
# 针对上述情况：补全Age数据；忽略Cabin数据
# PassengerId 和 Name 无关特征，故忽略

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def load_df(path):
    df = pd.read_csv(path)
    return df


def plot_df(df):
    plt.subplot2grid((2, 3), (0, 0))
    df.Survived.value_counts().plot(kind='bar')
    plt.title("Survived Distribution")
    plt.ylabel("Number")

    plt.subplot2grid((2, 3), (0, 1))
    df.Pclass.value_counts().plot(kind='bar')
    plt.title("Class Distribution")
    plt.ylabel("Number")

    plt.subplot2grid((2, 3), (0, 2))
    plt.scatter(df.Survived, df.Age)
    plt.title("Age Distribution")
    plt.ylabel("Age")
    plt.grid(b=True, which='major', axis='y')

    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    df.Age[df.Pclass == 1].plot(kind='kde')
    df.Age[df.Pclass == 2].plot(kind='kde')
    df.Age[df.Pclass == 3].plot(kind='kde')
    plt.xlabel("Age")
    plt.ylabel("Density")
    plt.title("Class and Age Distribution")
    plt.legend(('First', 'Second', 'Third'), loc='best')

    plt.subplot2grid((2, 3), (1, 2))
    df.Embarked.value_counts().plot(kind='bar')
    plt.title("Embarked Distribution")
    plt.ylabel("Number")

    plt.show()


def plot_df_Class(df):
    survived = df.Pclass[df.Survived == 1].value_counts()
    unsurvived = df.Pclass[df.Survived == 0].value_counts()
    dfp = pd.DataFrame({'Survived': survived, 'unSurvived': unsurvived})
    dfp.plot(kind='bar', stacked=True)
    plt.title("Class and Survived Status")
    plt.xlabel("Class")
    plt.ylabel("Number")
    plt.show()


def plot_df_Sex(df):
    survived = df.Sex[df.Survived == 1].value_counts()
    unsurvived = df.Sex[df.Survived == 0].value_counts()
    dfp = pd.DataFrame({'Survived': survived, 'unSurvived': unsurvived})
    dfp.plot(kind='bar', stacked=True)
    plt.title("Sex and Survived Status")
    plt.xlabel("Sex")
    plt.ylabel("Number")
    plt.show()


def plot_df_Sex_Class(df):
    fig = plt.figure()
    plt.title("Sex/Class and Survived Status")

    ax1 = fig.add_subplot(141)
    df.Survived[df.Sex == 'female'][df.Pclass != 3].value_counts().plot(kind='bar',
                                                                        label='female with first/second class')
    ax1.legend(['female with first/second class'], loc='best')

    ax2 = fig.add_subplot(142)
    df.Survived[df.Sex == 'female'][df.Pclass == 3].value_counts().plot(kind='bar', label='female with third class')
    ax2.legend(['female with third class'], loc='best')

    ax3 = fig.add_subplot(143)
    df.Survived[df.Sex == 'male'][df.Pclass != 3].value_counts().plot(kind='bar', label='male with first/second class')
    ax3.legend(['male with first/second class'], loc='best')

    ax4 = fig.add_subplot(144)
    df.Survived[df.Sex == 'male'][df.Pclass == 3].value_counts().plot(kind='bar', label='male with third class')
    ax4.legend(['male with third class'], loc='best')

    plt.show()


def plot_df_Embarked_Class(df):
    first = df.Embarked[df.Pclass == 1].value_counts()
    sencond = df.Embarked[df.Pclass == 2].value_counts()
    third = df.Embarked[df.Pclass == 3].value_counts()
    dfp = pd.DataFrame({'first': first, 'sencond': sencond, 'third': third})
    dfp.plot(kind='bar', stacked=True)
    plt.title("Embarked and Class Status")
    plt.xlabel("Embarked")
    plt.ylabel("Number")
    plt.show()


def plot_df_Embarked(df):
    Survived = df.Embarked[df.Survived == 1].value_counts()
    Unsurvived = df.Embarked[df.Survived == 0].value_counts()
    dfp = pd.DataFrame({'Survived': Survived, 'Unsurvived': Unsurvived})
    dfp.plot(kind='bar', stacked=True)
    plt.title("Embarked and Survived Status")
    plt.xlabel("Embarked")
    plt.ylabel("Number")
    plt.show()


def print_df_sibSp_parch(df):
    g = df.groupby(['SibSp', 'Survived'])
    dfp = pd.DataFrame(g.count()['PassengerId'])
    print(dfp)

    g = df.groupby(['Parch', 'Survived'])
    dfp = pd.DataFrame(g.count()['PassengerId'])
    print(dfp)


def print_df_Cabin(df):
    print(df.Cabin.value_counts())


def plot_df_Cabin_noCabin_Class(df):
    cabin = df.Pclass[pd.notnull(df.Cabin)].value_counts()
    no_cabin = df.Pclass[pd.isnull(df.Cabin)].value_counts()
    dfp = pd.DataFrame({'cabin': cabin, 'no_cabin': no_cabin})
    dfp.plot(kind='bar', stacked=True)
    plt.show()


def plot_df_Age_noAge_Class(df):
    age = df.Pclass[pd.notnull(df.Age)].value_counts()
    no_age = df.Pclass[pd.isnull(df.Age)].value_counts()
    dfp = pd.DataFrame({'age': age, 'no_age': no_age})
    dfp.plot(kind='bar', stacked=True)
    plt.show()


def plot_df_Mr_Mrs_Miss_Survived(df):
    Mr = df.Survived[df.Name.str.contains('Mr.')].value_counts()
    Mrs = df.Survived[df.Name.str.contains('Mrs.')].value_counts()
    Miss = df.Survived[df.Name.str.contains('Miss.')].value_counts()
    dfp = pd.DataFrame({'Mr': Mr, 'Mrs': Mrs, 'Miss': Miss})
    dfp.plot(kind='bar', stacked=True)
    plt.show()


def plot_df_Fare_Survived(df):
    Survived = df.Fare[df.Survived == 1].value_counts()
    unSurvived = df.Fare[df.Survived == 0].value_counts()
    dfp = pd.DataFrame({'Survived': Survived, 'unSurvived': unSurvived})
    dfp.plot(kind='bar', stacked=True)
    plt.show()
    pass


def fix_datas(df):
    df.drop(['PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'Fare'], axis=1, inplace=True)

    from sklearn.preprocessing import LabelBinarizer
    new_sex = LabelBinarizer().fit_transform(df.Sex)
    df.drop(['Sex'], axis=1, inplace=True)
    df.insert(0, 'Sex', pd.DataFrame(new_sex))

    title_list = list()
    for name in df.Name:
        title_list.append(name.split(', ')[1].split('. ')[0])
    df.insert(0, 'Title', pd.DataFrame(title_list))
    df.drop(['Name'], axis=1, inplace=True)

    title_dict = {}
    count = 0
    for title in df.Title:
        if title not in title_dict:
            title_dict[title] = count
            count += 1

    title_vect = []
    for title in df.Title:
        title_vect.append(title_dict[title])
    df.drop(['Title'], axis=1, inplace=True)
    df.insert(0, 'Title', pd.DataFrame(title_vect))

    child_list = list()
    for age in df.Age:
        if np.isnan(age):
            child_list.append(np.nan)
        elif age > 50:
            child_list.append(2)
        elif age > 15:
            child_list.append(1)
        else:
            child_list.append(0)
    df.insert(0, 'Adult', pd.DataFrame(child_list))
    df.drop(['Age'], axis=1, inplace=True)

    X_train = [[df['Title'][index], df['Sex'][index]] for index in range(df.shape[0]) if
               np.isnan(df['Adult'][index]) == False]
    y_train = [[df['Adult'][index]] for index in range(df.shape[0]) if np.isnan(df['Adult'][index]) == False]
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    for index in range(df.shape[0]):
        if np.isnan(df['Adult'][index]):
            df['Adult'][index] = gnb.predict([[df['Title'][index], df['Sex'][index]]])

    df.drop(['Title'], axis=1, inplace=True)

    return df


def predict(train_df, test_df):
    from sklearn.svm import SVC
    svc = SVC()
    svc.fit([[train_df['Adult'][index], train_df['Sex'][index], train_df['Pclass'][index]] for index in
             range(train_df.shape[0])], [train_df['Survived'][index] for index in range(train_df.shape[0])])
    return svc.predict([[test_df['Adult'][index], test_df['Sex'][index], test_df['Pclass'][index]] for index in
                        range(test_df.shape[0])])


train_path = '../data/titanic/train.csv'
train_df = load_df(train_path)
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# plot_df(train_df)
# plot_df_Class(train_df) #高等舱获救率高于低等舱
# plot_df_Sex(train_df) #女性获救率高于男性
# plot_df_Sex_Class(train_df) #男性/女性高等舱获取率高于男性/女性低等舱
# plot_df_Embarked_Class(train_df) #规律不明显
# plot_df_Embarked(train_df) #规律不明显
# print_df_sibSp_parch(train_df) #规律不明显
# print_df_Cabin(train_df)
# plot_df_Cabin_noCabin_Class(train_df) #一等舱记录比二、三等舱Cabin记录更完整
# plot_df_Age_noAge_Class(train_df) #高等舱Age记录更完整
# plot_df_Mr_Mrs_Miss_Survived(train_df) #Mrs可能为已婚，有孩子，可能会影响结果
# plot_df_Fare_Survived(train_df)

# 综合上述分析，提取特征 Survived Pclass Sex Age(Adult/Adult) Fare
fixed_train_df = fix_datas(train_df)
test_path = '../data/titanic/test.csv'
test_df = load_df(test_path)
fixed_test_df = fix_datas(test_df)
Survived = predict(fixed_train_df, fixed_test_df)
result_path = '../data/titanic/gender_submission.csv'
result_df = load_df(result_path)
PassengerId = [id for id in result_df.PassengerId]
submission_path = '../data/titanic/submission.csv'
submission_df = pd.DataFrame({'PassengerId': PassengerId, 'Survived': Survived})
submission_df.to_csv(submission_path, index=False, sep=',')
from sklearn.metrics import accuracy_score

print(accuracy_score([survived for survived in result_df.Survived], Survived))