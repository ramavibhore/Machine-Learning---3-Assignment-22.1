#I decided to treat this as a classification problem by creating a new binary variable affair (did the woman
#have at least one affair?) and trying to predict the classification for each woman.
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

#creating data set
dta = sm.datasets.fair.load_pandas().data
#Adding new column affair . If 1 means affair , 0 means no affair
dta['affair'] = (dta.affairs > 0).astype(int)

#Checking if there is any null column
dta.isnull().sum()

'''
rate_marriage      0
age                0
yrs_married        0
children           0
religious          0
educ               0
occupation         0
occupation_husb    0
affairs            0
affair             0
dtype: int64
'''

# histogram of education
dta.educ.hist()
plt.title('Histogram of Education')
plt.xlabel('Education Level')
plt.ylabel('Frequency')

# histogram of marriage rating
dta.rate_marriage.hist()
plt.title('Histogram of Marriage Rating')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')

# distribution of marriage ratings for those having affairs versus those not having affairs.
# barplot of marriage rating grouped by affair (True or False)
pd.crosstab(dta.rate_marriage, dta.affair.astype(bool)).plot(kind='bar')
plt.title('Marriage Rating Distribution by Affair Status')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')

#barplot to look at the percentage of women having affairs by number of years of marriage.
affair_yrs_married = pd.crosstab(dta.yrs_married, dta.affair.astype(bool))
affair_yrs_married.div(affair_yrs_married.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Affair Percentage by Years Married')
plt.xlabel('Years Married')
plt.ylabel('Percentage')

# create dataframes with an intercept column and dummy variables for occupation and occupation_husb
y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + religious + educ + C(occupation) + C(occupation_husb)',
dta, return_type="dataframe")
X.columns

# Assigining column name for X
X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
'C(occupation)[T.3.0]':'occ_3',
'C(occupation)[T.4.0]':'occ_4',
'C(occupation)[T.5.0]':'occ_5',
'C(occupation)[T.6.0]':'occ_6',
'C(occupation_husb)[T.2.0]':'occ_husb_2',
'C(occupation_husb)[T.3.0]':'occ_husb_3',
'C(occupation_husb)[T.4.0]':'occ_husb_4',
'C(occupation_husb)[T.5.0]':'occ_husb_5',
'C(occupation_husb)[T.6.0]':'occ_husb_6'})

# flatten y into a 1-D array
y = np.ravel(y)
print(y)
#[1. 1. 1. ... 0. 0. 0.]

#creating logistic regression model
model = LogisticRegression()
model = model.fit(X, y)
# check the accuracy on the training set
model.score(X, y)
#0.7258875274897895

#It seems 72% are accurate

y.mean()
#0.3224945020420987
#So 32% of the women had affair and 68% we could obtain accurately by always predicting "no"

# examine the coefficients
X.columns, np.transpose(model.coef_)

'''
(Index(['Intercept', 'occ_2', 'occ_3', 'occ_4', 'occ_5', 'occ_6', 'occ_husb_2',
        'occ_husb_3', 'occ_husb_4', 'occ_husb_5', 'occ_husb_6', 'rate_marriage',
        'age', 'yrs_married', 'children', 'religious', 'educ'],
       dtype='object'), array([[ 1.48986185],
        [ 0.18804138],
        [ 0.49891918],
        [ 0.25064145],
        [ 0.83897685],
        [ 0.83400823],
        [ 0.19057971],
        [ 0.29777979],
        [ 0.16135331],
        [ 0.18771782],
        [ 0.19394856],
        [-0.70312118],
        [-0.05841717],
        [ 0.10567634],
        [ 0.01691946],
        [-0.37113544],
        [ 0.00401654]]))
'''

## From the co-efficient relation we can observe that occupation like students are likely more in affirs

#Model evaluation using test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)

#predictive value using test data
predicted = model2.predict(X_test)
print(predicted)
#[1. 0. 0. ... 1. 0. 1.]

#Generate predictive probability
probs = model2.predict_proba(X_test)
print(probs)

'''
[[0.3489832  0.6510168 ]
 [0.90831773 0.09168227]
 [0.73455863 0.26544137]
 ...
 [0.33347785 0.66652215]
 [0.68652213 0.31347787]
 [0.35427667 0.64572333]]
'''

#the classifier is predicting a 1 (having an affair) any time the probability in the second column
#is greater than 0.5.

#evaluation metrics.
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[:, 1]))

#0.7425431711145997
#0.7493733025430991

#Here accuracy is 72% which is same as using score initially
print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))

'''
[[804  81]
 [247 142]]
             precision    recall  f1-score   support

        0.0       0.76      0.91      0.83       885
        1.0       0.64      0.37      0.46       389

avg / total       0.73      0.74      0.72      1274
'''
#Model evaluation using cross validation with 10 folds
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)

print(scores)
print(scores.mean())

#Using 10 folds , we are also getting 72% accurate result

'''
[0.72100313 0.70219436 0.73824451 0.70597484 0.70597484 0.72955975
 0.7327044  0.70440252 0.75157233 0.75      ]
0.7241630685514876
'''