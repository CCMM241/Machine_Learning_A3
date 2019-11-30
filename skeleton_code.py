"""
This script can be used as skelton code to read the challenge train and test
csvs, to train a trivial model, and write data to the submission file.
"""
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

## Read csvs
train_f = pd.read_csv('train.csv', index_col=0)
test_df = pd.read_csv('test.csv', index_col=0)

## Handle missing values
train_df.fillna('NA', inplace=True)
test_df.fillna('NA', inplace=True)


## Convert date from string format to datetime format

#remove all time zone data by stripping + & GMT sign
new_train_df = train_df['date'].str.replace('-','+')
new_train_df = train_df['date'].str.replace('GMT','+')
new = new_train_df.str.split("+",n=1,expand=True)
train_df['date']=new[0] 

#remove all weekday data
new_date = train_df['date'].str.split(",",n=1,expand=True)

#find the index of the rows without weekday data
index_null=pd.isnull(new_date).any(1).nonzero()[0]

#shift the date from column 0 to column 1 to get the complete list of date
new_date[1][index_null]=new_date[0][index_null]

#final data for date
train_df['date']=new_date[1]

#strip all space in date data
train_df['date']=train_df['date'].str.strip()

#convert string to time format
train_df['date'] = pd.to_datetime(train_df['date'], format='%d %b %Y %H:%M:%S')

#regenerate weekdays
train_df['weekday']=train_df['date'].dt.dayofweek

## Filtering column "mail_type"
train_x = train_df[['mail_type']]
train_y = train_df[['label']]

test_x = test_df[['mail_type']]

## Do one hot encoding of categorical feature
feat_enc = OneHotEncoder()
feat_enc.fit(train_x)
train_x_featurized = feat_enc.transform(train_x)
test_x_featurized = feat_enc.transform(test_x)

## Train a simple KNN classifier using featurized data
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_x_featurized, train_y)
pred_y = neigh.predict(test_x_featurized)

## Save results to submission file
pred_df = pd.DataFrame(pred_y, columns=['label'])
pred_df.to_csv("knn_sample_submission.csv", index=True, index_label='Id')
