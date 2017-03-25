# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:16:28 2017

@author: 呵呵
"""
from sklearn.preprocessing import LabelEncoder

# pands,numpy,seaborn,matplot
import re
import pandas as pd
from pandas import DataFrame,Series
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#data load

Titan = pd.read_csv("train.csv")
Test = pd.read_csv("test.csv")

traindata = pd.DataFrame()
testdata = pd.DataFrame()

#data info 

print("------------------------Titan.info------------------------------------")
Titan.info()
print("----------------------------------------------------------------------")
print(Titan.head())
print("-------------------------columns--------------------------------------")
print(Titan.columns)
print("----------------------------------------------------------------------")



#drop waste data

Titan = Titan.drop(["PassengerId",],axis = 1)
Test = Test.drop(["PassengerId"],axis = 1)
full = pd.concat([Titan,Test],axis=0)

# fill Embarked data in Titan
Titan.Embarked.value_counts()
Titan.Embarked = Titan.Embarked.fillna("S")

# fill Embarked data in test
Test['Fare'] = Test['Fare'].fillna(Test['Fare'].mean())


#check data
Titan.isnull().sum()
Test.isnull().sum()


#Cabin
#------------------------------------------------------------------------------
Titan['Cabin'] = Titan.Cabin.apply(lambda x: x[0] if pd.notnull(x) else 'X')
Titan['Cabin'] = LabelEncoder().fit_transform(Titan.Cabin)

Test['Cabin'] = Test.Cabin.apply(lambda x: x[0] if pd.notnull(x) else 'X')
Test['Cabin'] = LabelEncoder().fit_transform(Test.Cabin)


##out put traindata & testdata
traindata['Cabin'] = Titan['Cabin']
testdata['Cabin'] = Test['Cabin']


"""

值得琢磨一下的语句

def greeting_search(words):
    for word in words.split():
        if word[0].isupper() and word.endswith('.'):
            return word
            
train['Greeting'] = train.Name.apply(greeting_search)
train['Greeting'] = train.groupby('Greeting')['Greeting'].transform(lambda x: 'Rare' if x.count() < 9 else x)
del train['Name']
train['Greeting'] = LabelEncoder().fit_transform(train.Greeting)


"""


#we found that Name had some relations with Age,deal with it 
#------------------------------------------------------------------------------


#name
#------------------------------------------------------------------------------
sign_name = Series(Titan["Name"].sort_index())
test_sign_name = Series(Test["Name"].sort_index())

print ('名称处理中')
#------------------------------------------------------------------------------
#sign_name procession
home = list([0]*len(sign_name))
name = list([0]*len(sign_name))
i = 0
for val in sign_name:
    
    home[i] = sign_name[i].split(',')
    name[i] = home[i][1].strip().split('.')
    i = i + 1

home = DataFrame(home);
name = DataFrame(name);
home = home.drop(1,axis=1)
name = name.drop(2,axis=1)


sign_name = pd.merge(home,name,left_index=True,right_index=True)
sign_name = sign_name.rename(columns={'0_x':'Home','0_y':'Title',1:'Name'})

#------------------------------------------------------------------------------
#test_sign_name procession
print ('test名称处理中')
test_home = list([0]*len(test_sign_name))
test_name = list([0]*len(test_sign_name))

j = 0
for tval in test_sign_name:
    
    test_home[j] = test_sign_name[j].split(',')
    test_name[j] = test_home[j][1].strip().split('.')
    j = j + 1

test_home = DataFrame(test_home);
test_name = DataFrame(test_name);
test_home = test_home.drop(1,axis=1)
#test_name = test_name.drop(2,axis=1)


test_sign_name = pd.merge(test_home,test_name,left_index=True,right_index=True)
test_sign_name = test_sign_name.rename(columns={'0_x':'Home','0_y':'Title',1:'Name'})



#------------------------------------------------------------------------------
#以上应该写个函数,不过先这样吧
#马蛋！！要学正则表达式！


"""
frames = [Titan]
for df in frames:
    df["Title"] = df.Name.str.replace('(.*, )|(\\..*)', '')
"""
#一句话解决问题。。擦
"""
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
"""
#正则表达式取出name

#------------------------------------------------------------------------------
#Title procession

#sign_name['Sex'] = Titan['Sex']
#sign_name['SexSurvived'] = Titan['Survived']


titledict = {"Dr"   : "Mr",
             "Rev"  : "Mr",
             "Col" : "rare",
             "Mlle": "rare",
             "Major" : "rare",
             "Capt" : "rare",
             "Jonkheer"  : "rare",
             "Sir"  : "rare",
             "Don": "rare",
             "Dona": 'rare',
             "Mme": "rare",
             "Ms" : "Miss",
             "Lady"  : "rare",
             "the Countess" : "rare",
             'Mr':'Mr',
             "Miss" : "Miss",
             'Mrs' : 'Mrs',
             'Master':'Master'
             }
             

             
sign_name['Title'] = sign_name['Title'].map(titledict)
test_sign_name['Title'] = test_sign_name['Title'].map(titledict)

test_sign_name['Title'].value_counts()
sign_name['Title'].value_counts()

draw = sign_name.join([Titan['Survived']]).drop(['Home','Name'],axis=1)


"""
for df in re_between:
    for key,val in titledict.items():
        re_between.loc[re_between["Title"]==key, "Title"] = val    
        
"""


#out put traindata & testdata
traindata['Title'] = sign_name['Title']
testdata['Title'] = test_sign_name['Title']

numberdict = {'Mr' : 1,
              'Mrs' : 2,
              'Miss' : 3,
              'Master': 4,
              'rare': 5}


              
              
traindata['Title'] = traindata['Title'].map(numberdict)
testdata['Title'] = test_sign_name['Title'].map(numberdict)



#Age 
#------------------------------------------------------------------------------

#sign_Age procession
sign_age = pd.merge(DataFrame(Titan['Age']),sign_name,left_index=True,right_index=True)
test_sign_age = pd.merge(DataFrame(Test['Age']),test_sign_name,left_index=True,right_index=True)

#test_sign_Age procession
empty_sign_age= sign_age[sign_age['Age'].isnull()]
empty_test_sign_age= test_sign_age[test_sign_age['Age'].isnull()]

#sign_Age Mean,Std
Sum_of_Title = sign_age.groupby(['Title']).sum()
Mean_of_Title = sign_age.groupby(['Title']).mean()
Std_of_Title = sign_age.groupby(['Title']).std()
Number_of_title = empty_sign_age['Title'].value_counts()

#test_sign_AgeMean,Std
test_Sum_of_Title = test_sign_age.groupby(['Title']).sum()
test_Mean_of_Title = test_sign_age.groupby(['Title']).mean()
test_Std_of_Title = test_sign_age.groupby(['Title']).std()
test_Number_of_title = empty_test_sign_age['Title'].value_counts()


#Title list
Title_list = empty_sign_age['Title'].unique()
test_Title_list = empty_test_sign_age['Title'].unique()
#Calculate random numbers
def RandomAge(sex,mean,std,Noftit):
    low = mean.loc[sex,'Age'] - std.loc[sex,'Age']
    high = mean.loc[sex,'Age'] + std.loc[sex,'Age']
    random = np.random.randint(low,high,Noftit[sex])
    return random

    
for tit in Title_list:  
    empty_sign_age['Age'][empty_sign_age['Title'] == tit] = RandomAge(tit,Mean_of_Title,Std_of_Title,Number_of_title)
    

for tittest in test_Title_list:  
    empty_test_sign_age['Age'][empty_test_sign_age['Title'] == tittest] = RandomAge(tittest,test_Mean_of_Title,test_Std_of_Title,test_Number_of_title)
    
    
    
#fill randomnumber in sign_age
sign_age['Age'][sign_age['Age'].isnull()] = empty_sign_age['Age']
test_sign_age['Age'][test_sign_age['Age'].isnull()] = empty_test_sign_age['Age']      
#check info 
sign_age.count()
#drop info
sign_age= sign_age.drop(['Home','Title','Name'],axis=1)
test_sign_age= test_sign_age.drop(['Home','Title','Name'],axis=1)

"""
empty_sign_age['Age'][empty_sign_age['Title'] == 'Mr'] = RandomAge('Mr')
empty_sign_age['Age'][empty_sign_age['Title'] == 'Mrs'] = RandomAge('Mrs')
empty_sign_age['Age'][empty_sign_age['Title'] == 'Miss'] = RandomAge('Miss')
empty_sign_age['Age'][empty_sign_age['Title'] == 'Master'] = RandomAge('Master')
empty_sign_age['Age'][empty_sign_age['Title'] == 'rare'] = RandomAge('rare')
"""


draw = sign_name.join([Titan['Survived']]).drop(['Home','Name'],axis=1)
draw_mean = draw.groupby(['Title'],as_index = False).mean()


fig , (axis1,axis2) = plt.subplots(1,2,figsize= (10,4))
sns.countplot(x='Title', data=sign_name, ax=axis1)
sns.barplot(x='Title', y = 'Survived' , data=draw_mean, ax=axis2)
#sns.barplot(x= Mean_of_Title.index, y=Mean_of_Title.values, data=Mean_of_Title, ax=axis2)

##out put traindata & testdata

traindata['Age'] = sign_age
testdata['Age'] = test_sign_age



#Pclass -
#------------------------------------------------------------------------------


sign_Pclass = Titan[['Pclass','Survived','Sex']]
test_sign_Pclass = Test[['Pclass','Sex']]

Sur_sign_Pclass = sign_Pclass.groupby(sign_Pclass['Pclass'],as_index=False).mean()
female_Sur_sign_Pclass = sign_Pclass[sign_Pclass['Sex']=='female'].groupby(sign_Pclass['Pclass'],as_index=False).mean()
male_Sur_sign_Pclass = sign_Pclass[sign_Pclass['Sex']=='male'].groupby(sign_Pclass['Pclass'],as_index=False).mean()


fig , (axis1,axis2,axis3) = plt.subplots(1,3,figsize= (10,4))
sns.barplot(x= Sur_sign_Pclass.Pclass, y=Sur_sign_Pclass['Survived'], data=Sur_sign_Pclass, ax=axis1)
sns.barplot(x= female_Sur_sign_Pclass.Pclass, y=female_Sur_sign_Pclass['Survived'], data=female_Sur_sign_Pclass, ax=axis2)
sns.barplot(x= male_Sur_sign_Pclass.Pclass, y=male_Sur_sign_Pclass['Survived'], data=male_Sur_sign_Pclass, ax=axis3)


##out put traindata & testdata
traindata['Pclass'] = sign_Pclass.drop(['Survived','Sex'],axis = 1)
testdata['Pclass'] = test_sign_Pclass.drop(['Sex'],axis=1)




#Family
#------------------------------------------------------------------------------
sign_Family = pd.DataFrame()
sign_Family['Family'] = Titan['SibSp'] + Titan['Parch']
sign_Family['Survived'] = Titan['Survived']

#testdata 
test_sign_Family = pd.DataFrame()
test_sign_Family['Family'] = Test['SibSp'] + Test['Parch']






Sur_sign_Family = sign_Family.groupby(['Family'],as_index= False).mean()
fig , (axis1,axis2) = plt.subplots(1,2,figsize= (10,4))
sns.barplot(x= Sur_sign_Family.Family, y=Sur_sign_Family.Survived, data=Sur_sign_Family, ax=axis1)
sns.countplot(x=sign_Family.Family, data=sign_Family, ax=axis2)


# we see that person'family number = 0 and >4 with low Survive rates 1-3with high Survive rates
# so we divied the family by 3 types
sign_Family['Family'][sign_Family['Family'] == 1] =2
sign_Family['Family'][sign_Family['Family'] == 2] =2
sign_Family['Family'][sign_Family['Family'] == 3] =2
sign_Family['Family'][sign_Family['Family'] == 0 ] = 1
sign_Family['Family'][sign_Family['Family'] >3] = 3


test_sign_Family['Family'][test_sign_Family['Family'] == 1] =2
test_sign_Family['Family'][test_sign_Family['Family'] == 2] =2
test_sign_Family['Family'][test_sign_Family['Family'] == 3] =2
test_sign_Family['Family'][test_sign_Family['Family'] == 0 ] = 1
test_sign_Family['Family'][test_sign_Family['Family'] >3] = 3


##out put traindata & testdata
traindata['Family'] = sign_Family['Family']
testdata['Family'] = test_sign_Family['Family']


#Fare

#------------------------------------------------------------------------------

#Fare
sign_Fare =  Titan[['Fare','Pclass','Survived']]
test_sign_Fare =  Test[['Fare','Pclass']]
average_sign_Fare = sign_Fare.groupby(['Pclass'],as_index = False).mean()
empty_sign_Fare = sign_Fare[sign_Fare['Fare'] == 0]

# fill empty Fare
empty_sign_Fare['Fare'][empty_sign_Fare['Pclass'] == 1] = average_sign_Fare.iloc[0,1]
empty_sign_Fare['Fare'][empty_sign_Fare['Pclass'] == 2] = average_sign_Fare.iloc[1,1]
empty_sign_Fare['Fare'][empty_sign_Fare['Pclass'] == 3] = average_sign_Fare.iloc[2,1]

#refillin
sign_Fare[sign_Fare['Fare'] == 0] = empty_sign_Fare


##out put traindata & testdata
traindata['Fare'] = sign_Fare['Fare']
testdata['Fare'] = test_sign_Fare['Fare']






#Embarked

#------------------------------------------------------------------------------
#Embarked
sign_Embarked = Titan[['Embarked','Survived']]
#testEmbarked
Test_sign_Embarked = Test[['Embarked']]

#plot
average_sign_Embarked = sign_Embarked.groupby(['Embarked'],as_index = False).mean()

fig , (axis1,axis2) = plt.subplots(1,2,figsize= (10,4))
sns.barplot(x= average_sign_Embarked.Embarked, y=average_sign_Embarked.Survived, data=average_sign_Embarked, ax=axis1)
sns.countplot(x=sign_Embarked.Embarked, data=sign_Embarked, ax=axis2)
#Embarked turn into numner
sign_Embarked['Embarked'][sign_Embarked['Embarked']== 'S'] =1
sign_Embarked['Embarked'][sign_Embarked['Embarked']== 'Q'] =2
sign_Embarked['Embarked'][sign_Embarked['Embarked']== 'C'] =3
#testEmbarked turn into numner
Test_sign_Embarked['Embarked'][Test_sign_Embarked['Embarked']== 'S'] =1
Test_sign_Embarked['Embarked'][Test_sign_Embarked['Embarked']== 'Q'] =2
Test_sign_Embarked['Embarked'][Test_sign_Embarked['Embarked']== 'C'] =3


##out put traindata & testdata
#traindata['Embarked'] = sign_Embarked['Embarked']
#testdata['Embarked'] = Test_sign_Embarked['Embarked']




#Sex

#-----------------------------------------------------------------------------

traindata['Sex'] = Titan['Sex']
traindata['Sex'][traindata['Sex'] == 'female'] =2
traindata['Sex'][traindata['Sex'] == 'male'] =1

testdata['Sex'] = Test['Sex']
testdata['Sex'][testdata['Sex'] == 'female'] =2
testdata['Sex'][testdata['Sex'] == 'male'] =1


#-----------------------------------------------------------------------------
#getdummies 

def feature_dummies(dataset,title):
    dummy = pd.get_dummies(dataset[title],prefix=title)
    dataset = dataset.join(dummy)
    del dataset[title]
    return dataset

#Family
traindata = feature_dummies(traindata,'Family')
testdata = feature_dummies(testdata,'Family')

#traindata['Family'] = (pd.get_dummies(traindata['Family'],prefix='fam'))['fam_2']
#testdata['Family'] = (pd.get_dummies(testdata['Family'],prefix='fam'))['fam_2']

#pclass
traindata = feature_dummies(traindata,'Pclass')
testdata = feature_dummies(testdata,'Pclass')
#sex
traindata = feature_dummies(traindata,'Sex')
testdata = feature_dummies(testdata,'Sex')
#Title
traindata = feature_dummies(traindata,'Title')
testdata = feature_dummies(testdata,'Title')
#embarked
#traindata = feature_dummies(traindata,'Embarked')
#testdata = feature_dummies(testdata,'Embarked')



traindata  = traindata.drop(['Family_1','Family_3','Pclass_3','Title_1'],axis = 1)
testdata  = testdata.drop(['Family_1','Family_3','Pclass_3','Title_1'],axis = 1)


#Mean normalization
#------------------------------------------------------------------------------


def data_Mean_normalization(Dataset,feature):
    Dataset[feature] = (Dataset.loc[:,feature] - Dataset.loc[:,feature].mean())/(Dataset.loc[:,feature].max() - Dataset.loc[:,feature].min())
    

data_Mean_normalization(traindata,'Age')
data_Mean_normalization(traindata,'Fare')

data_Mean_normalization(testdata,'Age')
data_Mean_normalization(testdata,'Fare')


#ML 
#------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

"""
LabelEncoder().fit_transform(Titan.Pclass)
Out[269]: array([2, 0, 2, ..., 2, 0, 2], dtype=int64)
神语句

train['Cabin'] = train.Cabin.apply(lambda x: x[0] if pd.notnull(x) else 'X')
train['Cabin'] = LabelEncoder().fit_transform(train.Cabin)

train['Age'] = train.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.replace(np.nan, x.median()))
train.iloc[1043, 6] = 7.90

"""

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
"""
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import BaggingClassifier
"""

Ytraindata = Titan['Survived']

def draw_best_features():
    clf=LogisticRegression()
    clf.fit(traindata,Ytraindata)
    importances = clf.feature_importances_
    names=traindata.columns.values

    pd.Series(importances*100, index=names).plot(kind="bar")
    plt.show()
    



    
#Cross Validation
#------------------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(traindata, Ytraindata, train_size = 0.6)

clf = LogisticRegression()

def find_C(X, y):
    Cs = np.logspace(-10, 10, 10)
    score = []  
    for C in Cs:
        clf.C = C
        clf.fit(X_train, y_train)
        score.append(clf.score(X, y))
  
    plt.figure()
    plt.semilogx(Cs, score, marker='x')
    plt.xlabel('Value of C')
    plt.ylabel('Accuracy on Cross Validation Set')
    plt.title('What\'s the Best Value of C?')
    plt.show()
    clf.C = Cs[score.index(max(score))]
    print("Ideal value of C is %g" % (Cs[score.index(max(score))]))
    print('Accuracy: %g' % (max(score)))

#Analyzing the Results
#------------------------------------------------------------------------------

#coef = pd.DataFrame({'Variable': traindata.columns, 'Coefficient': clf.coef_[0]})
#coef


#Precision and Recall
#------------------------------------------------------------------------------



results = y_val.tolist()
predict = clf.predict(X_val)

def precision_recall(predictions, results):
    
    tp, fp, fn, tn, i = 0.0, 0.0, 0.0, 0.0, 0
    
    while i < len(results):
        
            if predictions[i] == 1 and results[i] == 1:
                tp = tp + 1
            elif predictions[i] == 1 and results[i] == 0:
                fp = fp + 1
            elif predictions[i] == 0 and results[i] == 0:
                tn = tn + 1
            else: 
                fn = fn + 1
            i = i + 1
    
    precision = tp / (tp + fp)
    recall = tn / (tn + fn)
    f1 = 2*precision*recall / (precision + recall)
    print("Precision: %g, Recall: %g, f1: %g" % (precision, recall, f1))

















#------------------------------------------------------------------------------
logreg = clf

logreg.fit(traindata, Ytraindata,)

Y_pred = logreg.predict(testdata)

logreg.score(traindata, Ytraindata)




#https://www.kaggle.com/jonathanbechtel/titanic/mastering-the-basics-on-the-rms-titanic