import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# %% Load data

data = pd.read_csv('../input/train.csv')
data.head()
dataTest = pd.read_csv('../input/test.csv')
dataTest.head()

# %% Preprocess data

le = LabelEncoder()
le.fit(data.Sex)
data.loc[:, 'Sex'] = le.transform(data.Sex)
dataTest.loc[:, 'Sex'] = le.transform(dataTest.Sex)

# Fill in the missing values
mAge = pd.concat((data.Age, dataTest.Age), axis=0).mean()
data.loc[:, 'Age'] = data.Age.fillna(mAge)
dataTest.loc[:, 'Age'] = dataTest.Age.fillna(mAge)

# %% Prepare data for fitting
X = data.loc[:, ['Age', 'Sex', 'Pclass']]
Y = data.loc[:, 'Survived']
XTest = dataTest.loc[:, ['Age', 'Sex', 'Pclass']]
XTrain, XValid, YTrain, YValid = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# %% Fit logistic regression using scikit

LR = LogisticRegression(C=10000)
LR.fit(X=XTrain, y=YTrain)

# Use model to predict on training and validation sets

yPredTrain = LR.predict(XTrain)
yPredValid = LR.predict(XValid)


def acc(Y: np.array, yPred: np.array) -> float:
    return np.sum(yPred == Y) / len(Y)


print('Train set accuracy', acc(YTrain, yPredTrain), '%')
print('Validation set accuracy', acc(YValid, yPredValid), '%')

# Predict for test set

yPredTest = LR.predict(XTest)

# Create a Kaggle submission
sub = pd.DataFrame({'PassengerId': dataTest['PassengerId'],
                    'Survived': yPredTest})

sub.to_csv('scikitLRExample.csv',
           index=False)
