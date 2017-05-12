# -*- coding: utf-8 -*-
"""
Created on Thu May  4 20:14:20 2017

@author: Gareth
"""


#%% Imports

# The usuals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Regular expressions for handelling string features
import re

# LabelEncoder for handelling categorical features (and names)
from sklearn.preprocessing import LabelEncoder

# LightGBM
import lightgbm as lgb

# sklearn tools for model training and assesment
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.model_selection import GridSearchCV


#%% Import data

# Import both data sets
trainRaw = pd.read_csv('train.csv')
testRaw = pd.read_csv('test.csv')

# And concatonate together
nTrain = trainRaw.shape[0]
full = pd.concat([trainRaw, testRaw], axis=0)


#%% Cabins

def ADSplit(s):
    """
    Function to try and extract cabin letter and number from the cabin column.
    Runs a regular expression (urgh) that finds letters and numbers in the 
    string. These are held in match.group, if they exist.
    """

    match = re.match(r"([a-z]+)([0-9]+)", s, re.I)
    
    try:
        letter = match.group(1)
    except:
        letter = ''
    
    try:
        number = match.group(2)
    except:
        number = 9999

    return letter, number


def DR(s):
    """
    From the cabin string, try and extract letter, number, and number of cabins
    """
    
    # Check contents
    if isinstance(s, (int, float)):
        # If field is empty, return nothing
        letter = ''
        number = ''
        nRooms = 9999
    else:
        # If field isn't empty, split sting on space. Some strings contain 
        # multiple cabins.
        s = s.split(' ')
        # Count the cabins based on number of splits
        nRooms = len(s)
        # Just take first cabin for letter/number extraction
        s = s[0]
        
        letter, number = ADSplit(s)
   
    return [letter, number, nRooms]

# Apply DR function to each cell in Cabin column using pandas apply method.    
out = full['Cabin'].apply(DR)
# Outout tuple with 3 values for each row, convert this to pandas df
out = out.apply(pd.Series)
# And name the columns
out.columns = ['CL', 'CN', 'nC']
    
# Then concatenate these columns to the dataset
full = pd.concat([full, out], axis=1)
    

#%% Family 

# Add some family features directly to new columns in the dataset
# Size
full['fSize'] = full['SibSp'] + full['Parch'] + 1
# Ratio
full['fRatio'] = (full['Parch']+1) / (full['SibSp']+ 1)
# Adult?
full['Adult'] = full['Age']>18    

    
#%% Names
# Extract titles from Name column, standardise

titleDict = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Sir",
    "Don": "Sir",
    "Sir": "Sir",
    "Dr": "Dr",
    "Rev": "Rev",
    "theCountess": "Lady",
    "Dona": "Lady",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Lady"
}


def splitName(s, titleDict):
    """
    Extract title from name, replace with value in title dictionary. Also 
    return surname.
    """
    
    # Remove '.' from name string
    s = s.replace('.', '')
    # Split on spaces
    s = s.split(' ')
    # get surname
    surname = s[0]

    # Get title - loop over titleDict, if s matches a key, take the 
    # corresponding value as the title
    title = [t for k,t in titleDict.items() if str(k) in s]
   
    # If no matching keys in title dict, use 'Other'.
    if title == []:
        title = 'Other'
    else:
        # Title is a list, so extract contents
        title = title[0]
    
    # Return surname (stripping remaining ',') and title as string
    return surname.strip(','), title


# Apply functions to df and concatenate new columns as before
out = full['Name'].apply(splitName, args=[titleDict])
out = out.apply(pd.Series)
out.columns = ['Surname', 'Title']
  
full = pd.concat([full, out], axis=1)

    
#%% Categorical columns

# List of categorical columns to recode
catCols = ['Sex', 'Embarked', 'CL', 'CN', 'Surname', 'Title']

# Recode
for c in catCols:
    # Convert column to pd.Categotical
    full[c] = pd.Categorical(full[c])
    # Extract the cat.codes and replace the column with these
    full[c] = full[c].cat.codes
    # Convert the cat codes to categotical...    
    full[c] = pd.Categorical(full[c])


# Generate a logical index of categorical columns to use with LightGBM later
catCols = [i for i,v in enumerate(full.dtypes) if str(v)=='category']


#%% Age

# Replace missing age values with median. 
# See ither kernels for more sophisticated ways of doing this!
full.loc[full.Age.isnull(), 'Age'] = np.median(full['Age'].loc[full.Age.notnull()])


#%% Tickets (not using here)
"""
def tix_clean(j):
    j = j.replace(".", "")
    j = j.replace("/", "")
    j = j.replace(" ", "")
    return j


def tix_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0
    
    
out = full['Ticket'].apply(tix_clean)
out = full['TicketLab'].apply(tix_label)
full = pd.concat([full, out], axis=1)
"""

#%% 
train = full.iloc[0:nTrain,:]
test = full.iloc[nTrain::,:]


#%% 

def prepLGB(data, classCol = '', IDCol = '', fDrop = []):
    
        # Drop class column
        if classCol != '':
            labels = data[classCol]
            fDrop = fDrop + [classCol]
        else:
            labels = []
    
        if IDCol != '':
            IDs = data[IDCol]

        if fDrop != []:
           data =  data.drop(fDrop, axis=1)
       
        # Create LGB mats        
        lData = lgb.Dataset(np.array(data), label=labels, free_raw_data=False, 
                            feature_name=list(data.columns),
                            categorical_feature = 'auto')
        
        return lData, labels, IDs, data
        

"""
train.columns
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
"""

fDrop = ['Ticket', 'Cabin', 'Name']


trainData, validData = train_test_split(train, test_size=0.4)


trainDataL, trainLabels, trainIDs, trainData = prepLGB(trainData, 
                                                 classCol = 'Survived', 
                                                 IDCol = 'PassengerId',
                                                 fDrop = fDrop)
                                                 

validDataL, validLabels, validIDs, validData = prepLGB(validData, 
                                                 classCol = 'Survived', 
                                                 IDCol = 'PassengerId',
                                                 fDrop = fDrop)

allTrainDataL, allTrainLabels, _ , allTrainData = prepLGB(train, 
                                                 classCol = 'Survived', 
                                                 IDCol = 'PassengerId',
                                                 fDrop = fDrop)

testDataL, _, _ , testData = prepLGB(test, 
                                 classCol = 'Survived', 
                                 IDCol = 'PassengerId',
                                 fDrop = fDrop)


#%% 

# Create a set if parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : 4,
          'min_data_in_leaf' : 20,
          'objective': 'binary', 
          'nthread': 5, 
          'silent': True,
          'num_leaves': 128, 
          'learning_rate': 0.05, 
          'max_bin': 512, 
          'subsample_for_bin': 200,
          'subsample': 1, 
          'subsample_freq': 1, 
          'colsample_bytree': 0.8, 
          'reg_alpha': 5, 
          'reg_lambda': 10,
          'min_split_gain': 0.5, 
          'min_child_weight': 1, 
          'min_child_samples': 5, 
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'binary_error'}
          
params = {'metric' : 'binary_error'}
# Run k-fold crossvalidation on all the training data to see how they perform
cv_results = lgb.cv(params, 
                     allTrainDataL, # Using all training data
                     num_boost_round = 10000, 
                     categorical_feature = catCols, # Specify categorical cols
                     nfold = 20, 
                     stratified = False, 
                     shuffle = True, 
                     early_stopping_rounds = 20, 
                     verbose_eval = 1, 
                     show_stdv = True, 
                     seed = 0)

plt.errorbar(x=range(0, len(cv_results['binary_error-mean'])),
             y=cv_results['binary_error-mean'], 
             yerr=cv_results['binary_error-stdv'])
plt.show()

#%% 
    
# Scores ~0.784    
params = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'binary', 
          'nthread': 5, 
          'silent': True,
          'num_leaves':  6,
          'learning_rate': 0.05, 
          'max_bin': 512, 
          'subsample_for_bin': 200,
          'subsample': 1, 
          'subsample_freq': 1, 
          'colsample_bytree': 0.8, 
          'reg_alpha': 5, 
          'reg_lambda': 10,
          'min_split_gain': 0.5, 
          'min_child_weight': 1, 
          'min_child_samples': 5, 
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'binary_error'}
            
gbm = lgb.train(params, 
                trainDataL, 
                100000, 
                valid_sets=[trainDataL, validDataL],
                early_stopping_rounds = 16,
                verbose_eval=4)

lgb.plot_importance(gbm)
predsValid = gbm.predict(validData, num_iteration=gbm.best_iteration)
predsTrain = gbm.predict(trainData, num_iteration=gbm.best_iteration)
predsTest = gbm.predict(testData, num_iteration=gbm.best_iteration)


#%% Different params

# Scores?
params = {'boosting_type': 'gbdt',
          'max_depth' : 50,
          'min_data_in_leaf' : 12,
          'objective': 'binary', 
          'nthread': 5, 
          'silent': True,
          'num_leaves': 128, 
          'learning_rate': 0.05, 
          'max_bin': 512, 
          'subsample_for_bin': 200,
          'subsample': 1, 
          'subsample_freq': 1, 
          'colsample_bytree': 0.8, 
          'reg_alpha': 5, 
          'reg_lambda': 10,
          'min_split_gain': 0.5, 
          'min_child_weight': 1, 
          'min_child_samples': 5, 
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'binary_error'}

gbm = lgb.train(params, 
                trainDataL, 
                100000, 
                valid_sets=[trainDataL, validDataL],
                early_stopping_rounds = 50,
                verbose_eval=4)

lgb.plot_importance(gbm)
predsValid = gbm.predict(validData, num_iteration=gbm.best_iteration)
predsTrain = gbm.predict(trainData, num_iteration=gbm.best_iteration)
predsTest = gbm.predict(testData, num_iteration=gbm.best_iteration)



#%%

def assessMod(predsTrain, yTrain, predsValid=[], yValid=[], 
              report=True, plot=True):
    """
    Using sklearn functions, return accuracy and ROC metrics (tpr, fpr, auc)
    for training data and validation data (if included).
    
    preds should be model preditions, yTrain and yValid should be labels.
    
    """
    trainAcc = accuracy_score(yTrain, np.round(predsTrain))
    fprTrain, tprTrain, thresholdsTrain = roc_curve(yTrain, predsTrain)
    trainAUC =  auc(fprTrain, tprTrain)
    
    if predsValid != []:
        accuracy_score(yValid, np.round(predsValid))
        fprValid, tprValid, thresholdsValid = roc_curve(yValid, predsValid)
        validAcc = accuracy_score(yValid, np.round(predsValid))
        validAUC = auc(fprValid, tprValid)
    else: 
        validAcc = np.nan
        fprValid = np.nan
        tprValid = np.nan
        validAUC = np.nan
   
    if report:
        print('Train accuracy:', trainAcc, '| Train AUC:', 
             trainAUC)
        if not isinstance(predsValid, list):
            print('Validation accuracy:', validAcc, '| Test AUC:', 
                  validAUC)
        
        print('-'*30)
    
    # Plot
    if plot:
        plotROC(tprTrain, fprTrain, label='Train')
        if not isinstance(predsValid, list):
            plotROC(tprValid, fprValid, label='Valid')
      
    # Stats output
    stats = {'fprTrain' : fprTrain,
             'fprValid' : fprValid,
             'tprTrain' : tprTrain,
             'tprValid' : tprValid,
             'trainAcc' : trainAcc,
             'validAcc' : validAcc,
             'trainAUC' : trainAUC,
             'validAUC' : validAUC}

    return stats
  
    
def plotROC(tpr, fpr, label=''):
    """
    Plot ROC curve from tpr and fpr.
    """
    plt.plot(fpr, tpr, label=label)
    plt.legend()
    plt.ylabel('True positive rate.')
    plt.xlabel('False positive rate')
    plt.show()


# Report model performance on training and validation sets                       
assessMod(predsTrain, trainLabels, 
          predsValid = predsValid, yValid = validLabels, 
          report=True, plot=True)               
                       
#%% Save sub


sub = pd.DataFrame()
sub['PassengerId'] = test['PassengerId']
sub['Survived'] = np.int32(predsTest>0.5)
sub.to_csv('sub.csv', index=False)
                       

#%% 



gridParams = {
    'learning_rate': [0.0005],
    'n_estimators': [12,24],
    'num_leaves': [20,30],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'seed' : [500],
    'colsample_bytree' : [0.7,0.8],
    'subsample' : [0.7,1],
    'reg_alpha' : [0,0.5],
    'reg_lambda' : [0,1,5],
    }

gridParams = {
    'learning_rate': [0.1, 0.02],
    'num_leaves': [5,6,7,20],
    'max_depth': [-1,12,24,48],
    'n_estimators' : [20, 40, 45, 50, 55, 60, 75],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'seed' : [500],
    'colsample_bytree' : [0.65, 0.7, 0.75],
    'subsample' : [0.65, 0.7, 0.75],
    'reg_alpha' : [1, 10],
    'reg_lambda' : [1, 6, 10],
    'subsample_for_bin': [100, 150, 200, 220, 300],
    'max_bin': [32,64] 
    }



mdl = lgb.LGBMClassifier(boosting_type= 'gbdt', 
          objective='binary', 
          nthread= 5, 
          silent= True,
          max_depth= -1,
          max_bin= 128, 
          subsample_for_bin= 500,
          subsample= 1, 
          subsample_freq= 1, 
          min_split_gain = 0.5, 
          min_child_weight = 1, 
          min_child_samples = 5, 
          scale_pos_weight = 1)

mdl.get_params().keys()
grid = GridSearchCV(mdl, gridParams, verbose=1, cv=4, n_jobs=-1)
grid.fit(allTrainData, allTrainLabels)


grid.grid_scores_, grid.best_params_, grid.best_score_

params2 =  grid.best_params_


#%% Retrain


best_params_   = {'boosting_type': 'gbdt',
 'colsample_bytree': 0.7,
 'learning_rate': 0.1,
 'max_bin': 32,
 'max_depth': -1,
 'n_estimators': 50,
 'num_leaves': 6,
 'objective': 'binary',
 'reg_alpha': 1,
 'reg_lambda': 1,
 'seed': 500,
 'subsample': 0.7,
 'subsample_for_bin': 200}


params = {'boosting_type': 'gbdt', 
          'objective': 'binary', 
          'nthread': 5, 
          'silent': True,
          'num_leaves': 128, 
          'learning_rate': 0.005, 
          'max_depth': -1,
          'max_bin': 512, 
          'subsample_for_bin': 200,
          'subsample': 1, 
          'subsample_freq': 1, 
          'colsample_bytree': 0.8, 
          'reg_alpha': 5, 
          'reg_lambda': 10,
          'min_split_gain': 0.5, 
          'min_child_weight': 1, 
          'min_child_samples': 5, 
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'binary_error'}

params['colsample_bytree'] = best_params_['colsample_bytree']
params['learning_rate'] = best_params_['learning_rate'] 
params['max_bin'] = best_params_['max_bin']
params['num_leaves'] = best_params_['num_leaves']
params['reg_alpha'] = best_params_['reg_alpha']
params['reg_lambda'] = best_params_['reg_lambda']
params['subsample'] = best_params_['subsample']
params['subsample_for_bin'] = best_params_['subsample_for_bin']
 

      
gbm = lgb.train(params, 
                trainDataL, 
                100000, 
                valid_sets=[trainDataL, validDataL],
                early_stopping_rounds = 50,
                verbose_eval=4)

"""
gbm = lgb.train(params, 
                allTrainDataL, 
                100, 
                valid_sets=[trainDataL, validDataL],
                #early_stopping_rounds = 12,
                verbose_eval=4)
"""

lgb.plot_importance(gbm)
plt.show()
predsValid = gbm.predict(validData, num_iteration=gbm.best_iteration)
predsTrain = gbm.predict(trainData, num_iteration=gbm.best_iteration)
predsTest = gbm.predict(testData, num_iteration=gbm.best_iteration)

assessMod(predsTrain, trainLabels, predsValid=predsValid, yValid= validLabels, 
          report=True, plot=True)               


# Scores 0.79 
sub = pd.DataFrame()
sub['PassengerId'] = test['PassengerId']
sub['Survived'] = np.int32(predsTest>0.5)
sub.to_csv('sub.csv', index=False)                       


