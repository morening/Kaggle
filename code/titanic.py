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
def get_sibsp_type(num):
    if num == 1 or num == 2:
        return 'high'
    elif num == 0 or num == 3 or num == 4:
        return 'mid'
    else:
        return 'low'
df_all['SibSpType'] = df_all['SibSp'].apply(get_sibsp_type)

# sns.barplot(x='Parch', y='Survived', data=df_train, palette='Set3')
# plt.show()
# 父母与子女数量适中生存率较高 1&2&3 > 0&5 > 4&6
def get_parch_type(num):
    if num >= 1 and num <= 3:
        return 'high'
    elif num == 0 or num == 5:
        return 'mid'
    else:
        return 'low'
df_all['ParchType'] = df_all['Parch'].apply(get_parch_type)

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

def get_title(name):
    return name.split(', ')[1].split('. ')[0]
df_all['Title'] = df_all['Name'].apply(get_title)
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
# 家庭规模适中生存率大于其他 2&3&4 > 1&5&6&7 > 8&11
def get_family_type(num):
    if num >= 2 and num <= 4:
        return 'high'
    elif num == 1 or (num >= 5 and num <= 7):
        return 'mid'
    else:
        return 'low'
df_all['FamiliyType'] = df_all['FamiliySize'].apply(get_family_type)

def get_deck(cabin):
    return str(cabin)[0]
df_all['Cabin'].fillna('U', inplace=True)
df_all['Deck'] = df_all['Cabin'].apply(get_deck)

# sns.barplot(x='Deck', y='Survived', data=df_all[df_all['Survived'].notnull()], palette='Set3')
# plt.show()
# 船舱所在位置对生存率影响较大

ticket_group_counts = df_all['Ticket'].value_counts()
df_all['TicketGroupSize'] = df_all['Ticket'].apply(lambda ticket: ticket_group_counts[ticket])
# sns.barplot(x='TicketGroupSize', y='Survived', data=df_all[df_all['Survived'].notnull()], palette='Set3')
# plt.show()
# 适中的同号船票数量生存率比其他高 2&3&4 > 1&7 > 5&6
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
def get_fare_type(fare):
    if fare <= 18:
        return 'low'
    else:
        return 'high'
df_all['FareType'] = df_all['Fare'].apply(get_fare_type)

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

# df_all['Surname'] = df_all['Name'].apply(lambda name:name.split(',')[0].strip())
# surname_group_counts = df_all['Surname'].value_counts()
# df_all['SurnameGroupSize'] = df_all['Surname'].apply(lambda surname:surname_group_counts[surname])
# sns.barplot(x='SurnameGroupSize', y='Survived', data=df_all[df_all['Survived'].notnull()], palette='Set3')
# plt.show()
# surname group size不同，生存率也不同


features = ['Survived', 'Pclass', 'Sex', 'AgeType', 'FareType', 'Embarked', 'Title', 'FamiliyType', 'Deck',
            'TicketGroupType']
df_train = df_all[features][df_all['Survived'].notnull()]
df_test = df_all[features][df_all['Survived'].isnull()].drop('Survived', axis=1)
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)
import numpy as np
df_test.insert(27, 'Deck_T', pd.DataFrame([np.nan]))
df_test['Deck_T'].fillna(0, inplace=True)
X_train = df_train.as_matrix()[:, 1:]
y_train = df_train.as_matrix()[:, 0]
X_test = df_test.as_matrix()
# print(X_train.shape, y_train.shape, X_test.shape)

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

# {'classify__max_depth': 9, 'classify__n_estimators': 38, 'select__k': 16} 0.881563393193
# pipe = Pipeline([('select', SelectKBest()), ('classify', RandomForestClassifier(random_state=1, max_features='sqrt'))])
# param_search = {'select__k':list(range(2, 24, 2)), 'classify__n_estimators':list(range(20, 50, 2)), 'classify__max_depth':list(range(3, 60, 3))}
# from sklearn.model_selection import GridSearchCV
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_search, scoring='roc_auc', cv=10)
# gsearch.fit(X_train, y_train)
# print(gsearch.best_params_, gsearch.best_score_)

# skb = SelectKBest(k=16)
# rfc = RandomForestClassifier(max_depth=9, n_estimators=38)
# from sklearn.pipeline import make_pipeline
# clf = make_pipeline(skb, rfc)
# clf.fit(X=X_train, y=y_train)
# predicted_train = clf.predict(X_train)
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_train, predicted_train))
# print(df_train_copy[df_train_copy['Survived'] != predicted_train])

# {'classify__learning_rate': 0.1, 'classify__max_depth': 3, 'classify__n_estimators': 50, 'select__k': 16} 0.881036534602
# pipe = Pipeline([('select', SelectKBest()), ('classify', GradientBoostingClassifier(random_state=1))])
# param_search = {'select__k':list(range(2, 24, 2)), 'classify__n_estimators':list(range(20, 200, 30)), 'classify__learning_rate':[10**rate for rate in range(-3, 3)], 'classify__max_depth':list(range(3, 30, 3))}
# from sklearn.model_selection import GridSearchCV
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_search, scoring='roc_auc', cv=10)
# gsearch.fit(X_train, y_train)
# print(gsearch.best_params_, gsearch.best_score_)

# skb = SelectKBest(k=16)
# gbc = GradientBoostingClassifier(max_depth=3, n_estimators=50, learning_rate=0.1, random_state=1)
# from sklearn.pipeline import make_pipeline
# clf = make_pipeline(skb, gbc)
# clf.fit(X=X_train, y=y_train)
# predicted_train = clf.predict(X_train)
# from sklearn.metrics import accuracy_score
# print('accuracy_score: %f' % accuracy_score(y_train, predicted_train))


# {'classify__max_depth': 5, 'classify__min_samples_leaf': 2, 'classify__min_samples_split': 2, 'select__k': 14} 0.882011672812
# pipe = Pipeline([('select', SelectKBest()), ('classify', DecisionTreeClassifier(random_state=1))])
# param_search = {'select__k':list(range(2, 32, 2)), 'classify__max_depth':list(range(1, 10, 1)), 'classify__min_samples_split':list(range(2, 10, 1)), 'classify__min_samples_leaf':list(range(1, 5, 1))}
# from sklearn.model_selection import GridSearchCV
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_search, scoring='roc_auc', cv=10, n_jobs=-1)
# gsearch.fit(X_train, y_train)
# print(gsearch.best_params_, gsearch.best_score_)

# skb = SelectKBest(k=14)
# dtc = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, min_samples_split=2, random_state=1)
# from sklearn.pipeline import make_pipeline
# clf = make_pipeline(skb, dtc)
# clf.fit(X=X_train, y=y_train)
# predicted_train = clf.predict(X_train)
# from sklearn.metrics import accuracy_score
# print('accuracy_score: %f' % accuracy_score(y_train, predicted_train))


# {'classify__learning_rate': 1.5, 'classify__n_estimators': 90, 'select__k': 20} 0.86143874843
# pipe = Pipeline([('select', SelectKBest()), ('classify', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=1), random_state=1))])
# param_search = {'select__k':list(range(2, 24, 2)), 'classify__n_estimators':list(range(10, 100, 10)), 'classify__learning_rate':list(np.arange(0.5, 5.0, 0.5))}
# from sklearn.model_selection import GridSearchCV
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_search, scoring='roc_auc', cv=10)
# gsearch.fit(X_train, y_train)
# print(gsearch.best_params_, gsearch.best_score_)

# skb = SelectKBest(k=20)
# abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=1), random_state=1, learning_rate=1.5, n_estimators=90)
# from sklearn.pipeline import make_pipeline
# clf = make_pipeline(skb, abc)
# clf.fit(X=X_train, y=y_train)
# predicted_train = clf.predict(X_train)
# from sklearn.metrics import accuracy_score
# print('accuracy_score: %f' % accuracy_score(y_train, predicted_train))

# {'classify__C': 0.1, 'classify__degree': 3, 'select__k': 7} 0.842453759157
# pipe = Pipeline([('select', SelectKBest()), ('classify', SVC(random_state=1))])
# param_search = {'select__k':list(range(3, 24, 2)), 'classify__C':[0.1, 1.0, 10], 'classify__degree':list(range(3, 10, 1))}
# from sklearn.model_selection import GridSearchCV
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_search, scoring='roc_auc', cv=10, n_jobs=-1)
# gsearch.fit(X_train, y_train)
# print(gsearch.best_params_, gsearch.best_score_)

# skb = SelectKBest(k=7)
# svc = SVC(random_state=1, C=0.1, degree=3)
# from sklearn.pipeline import make_pipeline
# clf = make_pipeline(skb, svc)
# clf.fit(X=X_train, y=y_train)
# predicted_train = clf.predict(X_train)
# from sklearn.metrics import accuracy_score
# print('accuracy_score: %f' % accuracy_score(y_train, predicted_train))

# {'classify__n_neighbors': 18, 'classify__weights': 'uniform', 'select__k': 21} 0.874060025443
# pipe = Pipeline([('select', SelectKBest()), ('classify', KNeighborsClassifier())])
# param_search = {'select__k':list(range(3, 24, 2)), 'classify__n_neighbors':list(range(1, 30, 1)), 'classify__weights':['uniform', 'distance']}
# from sklearn.model_selection import GridSearchCV
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_search, scoring='roc_auc', cv=10, n_jobs=-1)
# gsearch.fit(X_train, y_train)
# print(gsearch.best_params_, gsearch.best_score_)

# skb = SelectKBest(k=21)
# svc = KNeighborsClassifier(weights='uniform', n_neighbors=18)
# from sklearn.pipeline import make_pipeline
# clf = make_pipeline(skb, svc)
# clf.fit(X=X_train, y=y_train)
# predicted_train = clf.predict(X_train)
# from sklearn.metrics import accuracy_score
# print('accuracy_score: %f' % accuracy_score(y_train, predicted_train))


# {'classify__base_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=2, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, presort=False, random_state=1,
#             splitter='best'), 'classify__n_estimators': 3, 'select__k': 19} 0.883268764886
base_estimator = [DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, min_samples_split=2, random_state=1),
                  RandomForestClassifier(max_depth=9, n_estimators=38),
                  SVC(random_state=1, C=0.1, degree=3, probability=True),
                  GradientBoostingClassifier(max_depth=3, n_estimators=50, learning_rate=0.1, random_state=1),
                  KNeighborsClassifier(weights='uniform', n_neighbors=18)]
# pipe = Pipeline([('select', SelectKBest()), ('classify', BaggingClassifier(random_state=1, max_samples=0.6, max_features=0.6))])
# param_search = {'select__k':list(range(3, 24, 2)), 'classify__base_estimator':base_estimator, 'classify__n_estimators':list(range(3, 15, 2))}
# from sklearn.model_selection import GridSearchCV
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_search, scoring='roc_auc', cv=10, n_jobs=-1)
# gsearch.fit(X_train, y_train)
# print(gsearch.best_params_, gsearch.best_score_)

skb = SelectKBest(k=19)
bc = BaggingClassifier(random_state=1, max_samples=0.6, max_features=0.6, n_estimators=3, base_estimator=DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, min_samples_split=2, random_state=1))
from sklearn.pipeline import make_pipeline
clf = make_pipeline(skb, bc)
clf.fit(X=X_train, y=y_train)
predicted_train = clf.predict(X_train)
from sklearn.metrics import accuracy_score
print('accuracy_score: %f' % accuracy_score(y_train, predicted_train))


# {'select__k': 21} 0.877402843825
# estimators = [('dt', DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, min_samples_split=2, random_state=1)),
#               ('rfc', RandomForestClassifier(max_depth=9, n_estimators=38)),
#               ('svc', SVC(random_state=1, C=0.1, degree=3, probability=True)),
#               ('gbc', GradientBoostingClassifier(max_depth=3, n_estimators=50, learning_rate=0.1, random_state=1)),
#               ('knn', KNeighborsClassifier(weights='uniform', n_neighbors=18))]
# pipe = Pipeline([('select', SelectKBest()), ('classify', VotingClassifier(estimators=estimators, voting='soft'))])
# param_search = {'select__k':list(range(3, 24, 2))}
# from sklearn.model_selection import GridSearchCV
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_search, scoring='roc_auc', cv=10, n_jobs=-1)
# gsearch.fit(X_train, y_train)
# print(gsearch.best_params_, gsearch.best_score_)

# skb = SelectKBest(k=21)
# vc = VotingClassifier(estimators=estimators, voting='soft')
# from sklearn.pipeline import make_pipeline
# clf = make_pipeline(skb, vc)
# clf.fit(X=X_train, y=y_train)
# predicted_train = clf.predict(X_train)
# from sklearn.metrics import accuracy_score
# print('accuracy_score: %f' % accuracy_score(y_train, predicted_train))


from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(clf, X_train, y_train, cv=10)
print('cv_score.mean: %f, cv_score.std: %f' % (cv_score.mean(), cv_score.std()))

predicted = clf.predict(X_test)
submission_path = '../data/titanic/submission.csv'
submission_df = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predicted.astype(np.int32)})
submission_df.to_csv(submission_path, index=False, sep=',')

# print(df_train_copy[df_train_copy['Survived'] != predicted_train])