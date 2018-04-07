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
print('load data done!')

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

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_all['Title'] = df_all['Name'].apply(lambda name: name.split(', ')[1].split('. ')[0])
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
df_all['Title'] = df_all['Title'].map(Title_Dict)
df_all['Title'] = le.fit_transform(df_all['Title'])
# sns.barplot(x='Title', y='Survived', data=df_all[df_all['Survived'].notnull()], palette='Set3')
# plt.show()
# 不同的称谓也会影响生存率

df_all['FamiliySize'] = df_all['SibSp'] + df_all['Parch']
# sns.barplot(x='FamiliySize', y='Survived', data=df_all[df_all['Survived'].notnull()], palette='Set3')
# plt.show()
df_all['IsAlone'] = 1
df_all.loc[df_all['FamiliySize']>0, 'IsAlone'] = 0

df_all['Cabin'].fillna('U', inplace=True)
df_all['Deck'] = df_all['Cabin'].apply(lambda cabin: str(cabin)[0])
df_all['Deck'] = le.fit_transform(df_all['Deck'])
# sns.barplot(x='Deck', y='Survived', data=df_all[df_all['Survived'].notnull()], palette='Set3')
# plt.show()
# 船舱所在位置对生存率影响较大

ticket_group_counts = df_all['Ticket'].value_counts()
df_all['TicketGroupSize'] = df_all['Ticket'].apply(lambda ticket: ticket_group_counts[ticket])
# sns.barplot(x='TicketGroupSize', y='Survived', data=df_all[df_all['Survived'].notnull()], palette='Set3')
# plt.show()
df_all['HasMate'] = 1
df_all.loc[df_all['TicketGroupSize'] == 0, 'HasMate'] = 0

df_all['Embarked'].fillna('C', inplace=True)
df_all['Embarked'] = le.fit_transform(df_all['Embarked'])

df_all['Fare'].fillna(8.05, inplace=True)
df_all['FareType'] = pd.qcut(df_all['Fare'], 6, labels=range(0, 6, 1))
df_all['FareType'] = le.fit_transform(df_all['FareType'])

title_value_counts = df_all['Title'].value_counts()
mean_age_dict = dict()
for title in title_value_counts.keys():
    mean_age_dict[title] = df_all['Age'][df_all['Age'].notnull()][df_all['Title'] == title].mean()
df_all.loc[df_all['Age'].isnull(), 'Age'] = df_all[df_all['Age'].isnull()]['Title'].apply(lambda title: mean_age_dict[title])
df_all['AgeType'] = pd.qcut(df_all['Age'], 4, labels=['child', 'young', 'midlife', 'aged'])
df_all['AgeType'] = le.fit_transform(df_all['AgeType'])

roll_list = list()
for index in range(df_all.shape[0]):
    if df_all.loc[index, 'AgeType'] == 'child':
        roll_list.append('child')
    elif df_all.loc[index, 'AgeType'] == 'aged':
        roll_list.append('grandparent')
    else:
        if df_all.loc[index, 'Parch'] > 0:
            if df_all.loc[index, 'Sex'] == 'male':
                roll_list.append('father')
            else:
                roll_list.append('mother')
        else:
            if df_all.loc[index, 'SibSp'] == 0:
                roll_list.append('single')
            else:
                if df_all.loc[index, 'Sex'] == 'male':
                    roll_list.append('husband')
                else:
                    roll_list.append('wife')
df_all['Roll'] = pd.DataFrame(roll_list)
df_all['Roll'] = le.fit_transform(df_all['Roll'])

df_all['Sex'] = le.fit_transform(df_all['Sex'])

df_all['Surname'] = df_all['Name'].apply(lambda name:name.split(',')[0].strip())
df_all['Surname'] = le.fit_transform(df_all['Surname'])
print('F E done!')

import numpy as np
features = ['Survived', 'Pclass', 'Sex', 'AgeType', 'FareType', 'Embarked', 'Title', 'Deck',
            'Roll', 'HasMate', 'IsAlone']
df_all_dummies = pd.get_dummies(df_all[features])
df_train = df_all_dummies[df_all_dummies['Survived'].notnull()]
df_test = df_all_dummies[df_all_dummies['Survived'].isnull()].drop('Survived', axis=1)
X_train = df_train.as_matrix()[:, 1:]
y_train = df_train.as_matrix()[:, 0]
X_test = df_test.as_matrix()
# print(X_train.shape, y_train.shape, X_test.shape)
# sns.heatmap(df_all_dummies[df_all_dummies['Survived'].notnull()].corr(), annot=True, square=True)
# plt.show()

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
def build_model(model, search_params, with_select=True):
    if with_select:
        pipe = Pipeline([('select', SelectKBest()), ('classify', model)])
        if 'select__k' not in search_params:
            search_params['select__k'] = range(2, 10, 1)
    else:
        pipe = Pipeline([('classify', model)])
    gsearch = GridSearchCV(estimator=pipe, param_grid=search_params, scoring='roc_auc', cv=10, n_jobs=-1)
    gsearch.fit(X_train, y_train)
    clf = gsearch.best_estimator_
    clf.fit(X=X_train, y=y_train)
    predicted_train = clf.predict(X_train)
    score = accuracy_score(y_train, predicted_train)
    print('%s accuracy_score: %f using %s' % (str(model).split('(')[0], score, gsearch.best_params_))
    return clf

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier

# lr = build_model(LogisticRegression(random_state=1, n_jobs=-1), {'classify__C':np.logspace(-1, 1, num=3)})
# rfc = build_model(RandomForestClassifier(random_state=1, max_features='sqrt', n_jobs=-1), {'classify__min_samples_leaf':range(1, 5, 1), 'classify__min_samples_split':range(2, 5, 1), 'classify__n_estimators':range(5, 15, 1), 'classify__max_depth':range(3, 15, 3)})
# gbc = build_model(GradientBoostingClassifier(random_state=1), {'classify__min_samples_leaf':range(1, 5, 1), 'classify__min_samples_split':range(2, 5, 1), 'classify__learning_rate':np.logspace(-2, 0, num=3), 'classify__n_estimators':range(50, 150, 20), 'classify__subsample':np.arange(0.5, 1.0, 0.1), 'classify__max_depth':range(3, 15, 3)})
# dtc = build_model(DecisionTreeClassifier(random_state=1), {'classify__max_depth':range(2, 7, 1), 'classify__min_samples_split':range(2, 5, 1), 'classify__min_samples_leaf':range(1, 5, 1)})
# svc = build_model(SVC(random_state=1, probability=True), {'classify__C':np.logspace(-1, 1, 3), 'classify__degree':list(range(3, 9, 2))})
# knn = build_model(KNeighborsClassifier(n_jobs=-1), {'classify__n_neighbors':range(5, 10, 1)})
# gnb = build_model(GaussianNB(), {})
# ada = build_model(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=1), random_state=1), {'classify__n_estimators':range(30, 70, 10), 'classify__learning_rate':np.logspace(-1, 1, 3)})
# xg = build_model(XGBClassifier(n_jobs=-1, random_state=1), {'classify__max_depth':range(3, 15, 3), 'classify__learning_rate':np.logspace(-2, 0, 3), 'classify__n_estimators':range(80, 120, 10)})
# bag = build_model(BaggingClassifier(random_state=1, base_estimator=DecisionTreeClassifier(random_state=1)), {'classify__max_features':np.arange(0.5, 1.0, 0.1), 'classify__max_samples':np.arange(0.5, 1.0, 0.1), 'classify__n_estimators':range(5, 15, 1)})

lr = build_model(LogisticRegression(random_state=1, n_jobs=-1), {'select__k':[9], 'classify__C':[10]})
rfc = build_model(RandomForestClassifier(random_state=1, max_features='sqrt', n_jobs=-1), {'select__k':[9], 'classify__min_samples_leaf':[1], 'classify__min_samples_split':[4], 'classify__n_estimators':[11], 'classify__max_depth':[6]})
gbc = build_model(GradientBoostingClassifier(random_state=1), {'select__k':[8], 'classify__min_samples_leaf':[4], 'classify__min_samples_split':[2], 'classify__learning_rate':[0.1], 'classify__n_estimators':[50], 'classify__subsample':[0.9], 'classify__max_depth':[3]})
dtc = build_model(DecisionTreeClassifier(random_state=1), {'select__k':[8], 'classify__max_depth':[5], 'classify__min_samples_split':[2], 'classify__min_samples_leaf':[2]})
svc = build_model(SVC(random_state=1, probability=True), {'select__k':[9], 'classify__C':[1.0], 'classify__degree':[3]})
knn = build_model(KNeighborsClassifier(n_jobs=-1), {'select__k':[8], 'classify__n_neighbors':[8]})
gnb = build_model(GaussianNB(), {'select__k':[2]})
ada = build_model(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=1), random_state=1), {'select__k':[3], 'classify__n_estimators':[50], 'classify__learning_rate':[1.0]})
xg = build_model(XGBClassifier(n_jobs=-1, random_state=1), {'select__k':[8], 'classify__max_depth':[3], 'classify__learning_rate':[0.1], 'classify__n_estimators':[110]})
bag = build_model(BaggingClassifier(random_state=1, base_estimator=DecisionTreeClassifier(random_state=1), n_jobs=-1), {'select__k':[9], 'classify__max_features':[0.7], 'classify__max_samples':[0.7], 'classify__n_estimators':[14]})

print('build model done!')

predicted_list = list()
estimator_list = [lr, rfc, gbc, dtc, svc, knn, gnb, ada, xg, bag]
for estimator in estimator_list:
    estimator.fit(X_train, y_train)
    predicted_list.append(estimator.predict(X_test))
predicted_array = np.array(predicted_list).transpose()
predicted_mean = predicted_array.mean(axis=1)
predicted = list()
for num in predicted_mean:
    if num > 0.5:
        predicted.append(1)
    else:
        predicted.append(0)
predicted = np.array(predicted)
submission_path = '../data/titanic/submission.csv'
submission_df = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predicted.astype(np.int32)})
submission_df.to_csv(submission_path, index=False, sep=',')
print('predict test done!')

df_all_path = '../data/titanic/df_all.csv'
df_all.to_csv(df_all_path, index=False, sep=',')
print('df_all write done!')
