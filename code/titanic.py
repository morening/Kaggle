import warnings
warnings.filterwarnings("ignore")

import pandas as pd

path_train = '../data/titanic/train.csv'
df_train = pd.read_csv(path_train)
path_test = '../data/titanic/test.csv'
df_test = pd.read_csv(path_test)
PassengerId = [id for id in df_test.PassengerId]
df_all = pd.concat([df_train, df_test], ignore_index=True)
df_train_copy = df_train.copy()

import seaborn as sns
import matplotlib.pyplot as plt


# sns.barplot(x='Pclass', y='Survived', data=df_train, palette='Set3')
# plt.show()
# 高等级生存率高于低等级

# sns.barplot(x='Sex', y='Survived', data=df_train, palette='Set3')
# plt.show()
# 女性生存率高于男性

# sns.barplot(x='SibSp', y='Survived', data=df_train, palette='Set3')
# plt.show()
# 兄弟姐妹数量适中生存率较高 1&2 > 0&3&4 > 5&8

# sns.barplot(x='Parch', y='Survived', data=df_train, palette='Set3')
# plt.show()
# 父母与子女数量适中生存率较高 1&2&3 > 0&5 > 4&6

# facet = sns.FacetGrid(df_train, hue='Survived', aspect=2)
# facet.map(sns.kdeplot, 'Age', shade=True)
# facet.set(xlim=(0, df_train['Age'].max()))
# facet.add_legend()
# plt.show()
# 儿童生存率大于成年人

# facet = sns.FacetGrid(df_train, hue='Survived', aspect=2)
# facet.map(sns.kdeplot, 'Fare', shade=True)
# facet.set(xlim=(0, df_train['Fare'].max()))
# facet.add_legend()
# plt.show()
# 票价越高，生存率越高


# sns.barplot(x='Embarked', y='Survived', data=df_train, palette='Set3')
# plt.show()
# 登船城市C的生存率大于其他

df_all['Title'] = df_all['Name'].apply(lambda name: name.split(', ')[1].split('. ')[0])
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
df_all['Title'] = df_all['Title'].map(Title_Dict)
# sns.barplot(x='Title', y='Survived', data=df_all[df_all['Survived'].notnull()], palette='Set3')
# plt.show()
# 不同的称谓也会影响生存率

df_all['FamiliySize'] = df_all['SibSp'] + df_all['Parch'] + 1
# sns.barplot(x='FamiliySize', y='Survived', data=df_all[df_all['Survived'].notnull()], palette='Set3')
# plt.show()
def get_family_type(num):
    if num >= 2 and num <= 4:
        return 'high'
    elif num == 1 or (num >= 5 and num <= 7):
        return 'mid'
    else:
        return 'low'
df_all['FamiliyType'] = df_all['FamiliySize'].apply(get_family_type)

df_all['Cabin'].fillna('U', inplace=True)
df_all['Deck'] = df_all['Cabin'].apply(lambda cabin: str(cabin)[0])
# sns.barplot(x='Deck', y='Survived', data=df_all[df_all['Survived'].notnull()], palette='Set3')
# plt.show()
# 船舱所在位置对生存率影响较大

ticket_group_counts = df_all['Ticket'].value_counts()
df_all['TicketGroupSize'] = df_all['Ticket'].apply(lambda ticket: ticket_group_counts[ticket])
# sns.barplot(x='TicketGroupSize', y='Survived', data=df_all[df_all['Survived'].notnull()], palette='Set3')
# plt.show()

def get_ticket_group_type(num):
    if num >=2 and num <= 4:
        return 'high'
    elif num == 1 or num == 7:
        return 'mid'
    else:
        return 'low'
df_all['TicketGroupType'] = df_all['TicketGroupSize'].apply(get_ticket_group_type)



# print(df_train[df_train['Embarked'] == 'C'][df_train['Pclass'] == 1].median())
# print(df_train[df_train['Embarked'].isnull()])
# Embarked 缺失的数据更接近于C
df_all['Embarked'].fillna('C', inplace=True)
# print(df_all[df_all['Embarked'] == 'S'][df_all['Pclass'] == 3].median())
# print(df_all[df_all['Fare'].isnull()])
df_all['Fare'].fillna(8.05, inplace=True)
df_all['FareType'] = df_all['Fare'].apply(lambda fare: 'low' if fare <= 18 else 'high')

features_age = ['Age', 'Pclass', 'Sex', 'Title']
df_age = df_all[features_age]
df_age = pd.get_dummies(df_age)
known_age = df_age[df_age['Age'].notnull()].as_matrix()
unknown_age = df_age[df_age['Age'].isnull()].as_matrix()
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=0, n_estimators=100)
rfr.fit(X=known_age[:, 1:], y=known_age[:, 0])
predicted_age = rfr.predict(unknown_age[:, 1:])
df_all.loc[(df_all['Age'].isnull()), 'Age'] = predicted_age
def get_age_type(age):
    if age <= 15:
        return 'child'
    elif age <= 50:
        return 'adult'
    else:
        return 'old'
df_all['AgeType'] = df_all['Age'].apply(get_age_type)

df_all['Surname'] = df_all['Name'].apply(lambda name:name.split(',')[0].strip())

import numpy as np
features = ['Survived', 'Pclass', 'Sex', 'AgeType', 'FareType', 'Embarked', 'Title', 'FamiliyType', 'Deck',
            'TicketGroupType']
df_all_dummies = pd.get_dummies(df_all[features])
df_train = df_all_dummies[df_all_dummies['Survived'].notnull()]
df_test = df_all_dummies[df_all_dummies['Survived'].isnull()].drop('Survived', axis=1)
X_train = df_train.as_matrix()[:, 1:]
y_train = df_train.as_matrix()[:, 0]
X_test = df_test.as_matrix()
# print(X_train.shape, y_train.shape, X_test.shape)

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
def build_model(model, search_params, with_select=True):
    if with_select:
        pipe = Pipeline([('select', SelectKBest()), ('classify', model)])
        if 'select__k' not in search_params:
            search_params['select__k'] = range(2, 32, 1)
    else:
        pipe = Pipeline([('classify', model)])
    gsearch = GridSearchCV(estimator=pipe, param_grid=search_params, scoring='roc_auc', cv=10, n_jobs=-1)
    gsearch.fit(X_train, y_train)
    clf = gsearch.best_estimator_
    clf.fit(X=X_train, y=y_train)
    predicted_train = clf.predict(X_train)
    print('%s accuracy_score: %f using %s' % (str(model).split('(')[0], accuracy_score(y_train, predicted_train), gsearch.best_estimator_))
    return clf

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier

# lr = build_model(LogisticRegression(random_state=1, n_jobs=-1), {'classify__C':np.logspace(-1, 1, num=3)})
# rfc = build_model(RandomForestClassifier(random_state=1, max_features='sqrt', n_jobs=-1), {'classify__max_depth':range(3, 15, 3)})
# gbc = build_model(GradientBoostingClassifier(random_state=1), {'classify__learning_rate':np.logspace(-2, 0, num=3), 'classify__n_estimators':range(50, 150, 20), 'classify__subsample':np.arange(0.5, 1.0, 0.1), 'classify__max_depth':range(3, 15, 3)})
# dtc = build_model(DecisionTreeClassifier(random_state=1), {'classify__max_depth':range(2, 10, 2), 'classify__min_samples_split':range(2, 10, 2), 'classify__min_samples_leaf':range(1, 5, 2)})
# svc = build_model(SVC(random_state=1, probability=True), {'classify__C':np.logspace(-1, 1, 3), 'classify__degree':list(range(3, 9, 2))})
# knn = build_model(KNeighborsClassifier(n_jobs=-1), {'classify__n_neighbors':range(5, 30, 5)})
# gnb = build_model(GaussianNB(), {})
# ada = build_model(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=1), random_state=1), {'classify__learning_rate':np.logspace(-1, 1, 3)})
# xg = build_model(XGBClassifier(n_jobs=-1, random_state=1), {'classify__max_depth':range(3, 15, 3), 'classify__learning_rate':np.logspace(-2, 0, 3), 'classify__n_estimators':range(50, 150, 50)})
bag = build_model(BaggingClassifier(random_state=1, max_samples=0.6, max_features=0.6, base_estimator=DecisionTreeClassifier(random_state=1, max_depth=5, min_samples_leaf=2, min_samples_split=2)), {'select__k':[19], 'classify__n_estimators':[13]})

clf = bag
clf.fit(X=X_train, y=y_train)
predicted = clf.predict(X_test)
submission_path = '../data/titanic/submission.csv'
submission_df = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predicted.astype(np.int32)})
submission_df.to_csv(submission_path, index=False, sep=',')

from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(clf, X_train, y_train, cv=10)
print('cv_score.mean: %f, cv_score.std: %f' % (cv_score.mean(), cv_score.std()))

from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator=bag, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 100), cv=10, n_jobs=-1)
plt.plot(train_sizes, train_scores.mean(axis=1), label='train_scores')
plt.plot(train_sizes, test_scores.mean(axis=1), label='test_scores')
plt.legend()
plt.show()
