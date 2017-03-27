################################################################################
# Description: Preprocessing of Shelter Animal dataset
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        gio 02 giu 2016 17:14:11 CEST
################################################################################

import numpy as np
import pandas as pd

############
# Settings #
############

from settings import *

############
# Mappings #
############

from mappings import *

##########################
# User-defined functions #
##########################

from PreprocessingAuxiliaryFunctions import *

#################
# Preprocessing #
#################

def preprocessing(inputDir):
    """ Preprocessing of shelter animals training and test set. The preprocessed
        datasets are written to the input Directory for future reuse.

    Args:
        inputDir (str): path to input directory
    """

    # 1) Read training set (use DateTime as index)
    train = pd.read_csv(inputDir + 'train.csv',
                        parse_dates=['DateTime'])

    # 2) Read training set (use DateTime as index)
    test = pd.read_csv(inputDir + 'test.csv',
                       index_col=0,
                       parse_dates=['DateTime'])

    # 3) Preprocessing

    # 3.1) Removing useless attributes
    #   - AnimalID does not contain duplicates
    #   - OutcomeSubtype is not used for evaluation and is not given in test set
    train.drop(['AnimalID', 'OutcomeSubtype'], axis=1, inplace=True)

    # 3.2) Replace Name attribute with boolean: 1 if animal is named, 0 if not.
    # Named animals could be more likely to be adopted or returned to their
    # owners.
    if preprocessingFlags['Name']:
        train['NameLength'] = train['Name'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        test['NameLength'] = test['Name'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        # train['Named'] = (train['NameLength'] > 0).astype(int)
        # test['Named'] = (test['NameLength']).astype(int)
    train.drop('Name', axis=1, inplace=True)
    test.drop('Name', axis=1, inplace=True)

    # 3.3) Remove instances with NaN values from training set (only 19)
    train.dropna(inplace=True)

    # Add missing columns to training or test set (for successive concatenation)
    test['OutcomeType'] = 'Test'

    # Concatenate DataFrames to perform preprocessing
    df = pd.concat([train, test])
    df.sort_values(by='DateTime', inplace=True)

    # 3.5) Process dates
    #   - Split dates in Year, Month, Day, Hour Attributes
    #   - Check day of the week
    #   - Check for holidays

    # referenceDate = datetime.date(2013, 01, 01)
    # train['Date'] = (train['DateTime'] -
    #                 referenceDate).astype('timedelta64[D]').astype(float)
    # train['Year'] = train['DateTime'].apply(lambda x: x.year)
    if preprocessingFlags['Year']:
        df['Year'] = df['DateTime'].apply(lambda x: x.year)

    if preprocessingFlags['DayOfYear']:
        df['DayOfYear'] = df['DateTime'].apply(lambda x: x.dayofyear)

    if preprocessingFlags['DayMonth']:
        df['Month'] = df['DateTime'].apply(lambda x: x.month)
        df['Day'] = df['DateTime'].apply(lambda x: x.day)
        df['WeekDay'] = df['DateTime'].apply(lambda x: x.isoweekday())

    if preprocessingFlags['MinuteHour']:
        df['Hour'] = df['DateTime'].apply(lambda x: x.hour)
        # df['0or9'] = df['Hour'].apply(lambda x: x == 0 or x == 9)
        # df = pd.get_dummies(df['Hour'].apply(hourTrend),columns=['ZeroNine', 'Rest', 'Afternoon', 'Evening'])
        # df[df.columns] = df
        df['Minute'] = df['DateTime'].apply(lambda x: x.minute)
        df['MinuteOfDay'] = df['Hour'] * 60 + df['Minute']
        # df.drop(['Hour','Minute'], axis=1, inplace=True)

    # if preprocessingFlags['TransferTogether']:
    #    df = df.reset_index()
    #    transferTimes = df.ix[df['OutcomeType'] == 'Transfer', 'DateTime'].values
    #    print transferTimes[1:] - transferTimes[:-1]
    #    df['TransferTogether'] = 0




 #        togetherIndexes = [j for i,j in zip(deltaTransferTimes, deltaTransferTimes.index)
                           # if i < 5 and df.ix[j, 'OutcomeType'] ]


    # transferTimes = train.ix[train['OutcomeType'] == 'Transfer', 'DateTime']
    # decalage1 = transferTimes[1:].reset_index(drop=True)
    # decalage2 = transferTimes[:-1].reset_index(drop=True)
    # deltaTransferTimes=((decalage1-decalage2)/np.timedelta64(1,'m')).astype(int)
    # train['TransferTogether']=0
    # togetherIndexes = [j for i,j in zip(deltaTransferTimes,range(len(deltaTransferTimes))) if i <= 5]
    # togetherIndexes = list (set (togetherIndexes + [i+1 for i in togetherIndexes]))
    # train.ix[transferTimes.reset_index().ix[togetherIndexes,'index'], 'TransferTogether'] = 1

    # test['TransferTogether']=0
    # df1=pd.DataFrame(transferTimes)
    # df1['Marker']=0
    # df2=pd.DataFrame(test['DateTime'].sort_values())
    # df2['Marker']=1
    # mergedDfs=df1.append(df2).reset_index().sort_values('DateTime')
    # transferTimes = mergedDfs['DateTime']
    # decalage1 = transferTimes[1:].reset_index(drop=True)
    # decalage2 = transferTimes[:-1].reset_index(drop=True)
    # deltaTransferTimes = ((decalage1 - decalage2) / np.timedelta64(1, 'm')).astype(int)
    # test['TransferTogether'] = 0
    # togetherIndexes = [j for i, j in zip(deltaTransferTimes, range(len(deltaTransferTimes))) if i <= 5]
    # togetherIndexes = list(set(togetherIndexes + [i + 1 for i in togetherIndexes]))
    # mergedDfs.reset_index(inplace=True)
    # testTransferIndexes = [i for i,j in zip(togetherIndexes, mergedDfs.ix[togetherIndexes,'Marker']) if j == 1]
    # test.ix[mergedDfs.ix[testTransferIndexes,'index'], 'TransferTogether'] = 1


    df.drop('DateTime', axis=1, inplace=True)

    # 3.6) Group animal ages in tranches and convert ages to days
    if preprocessingFlags['Age']:
        df['AgeDays'] = df['AgeuponOutcome'].apply(convertAges, 547.5)
        temp = pd.get_dummies(df['AgeDays'].apply(groupAges))
        df[temp.columns] = temp

    df.drop('AgeuponOutcome', axis=1, inplace=True)

    # 3.7) Binarize AnimalType: 0 for cat, 1 for dog
    if preprocessingFlags['AnimalType']:
        df.ix[df['AnimalType'] == 'Cat', 'AnimalType'] = 0
        df.ix[df['AnimalType'] == 'Dog', 'AnimalType'] = 1

    # 3.8) Process SexuponOutcome attribute
    #   - Extract Male/Female boolean variable
    #   - Extract Intact/Neutered boolean variable
    if preprocessingFlags['Sex']:
        df['Gender'] = df['SexuponOutcome'].apply(getGender)

    if preprocessingFlags['Status']:
        df['Status'] = df['SexuponOutcome'].apply(getStatus)

    df.drop('SexuponOutcome', axis=1, inplace=True)

    # 3.9) Process Breed information
    #   - Check if animal is a mix
    if preprocessingFlags['Mix']:
        df['Mix'] = df['Breed'].apply(isMix)

    #   - Check if animal is a pitbull
    if preprocessingFlags['Breeds']:
        # 3.9.1) Extract breeds macro-groups
        # TODO: at the moment, only dogs are considered. Process also cats
        # Training set
        temp = pd.DataFrame(0.0,
                        index=df.index,
                        columns=uniqueBreedGroupDog)

        belongToBreeds = df['Breed'].apply(lambda x: getBreedGroups(x,
                                                     mappingBreedGroupDog)).values

        for (i, j) in zip(df.index, range(len(temp))):
            print (i, j)
            temp.ix[i, belongToBreeds[j]] = 1.0
        df = df.join(temp)

        # Test if breed is considered dangerous/aggressive
        def isDangerous(x):
            return ('Pit Bull' in x or
                    'Cane Corso' in x or
                    'Bull Terrier' in x or
                    'Rhodesian Ridgeback' in x or
                    'Basenji' in x or
                    'American Bulldog' in x or
                    'Great Dane' in x or
                    'Wolf-Dog Hybrid' in x or
                    'Boxer' in x or
                    'Rottweiler' in x or
                    'Chow Chow' in x or
                    'Doberman Pinscher' in x or
                    'Siberian Husky' in x or
                    'Alaskan Malamute' in x)

        df['Dangerous'] = df['Breed'].apply(isDangerous)
        df['Domestic'] = df['Breed'].apply(lambda x: 'Domestic' in x).astype(int)

    df.drop(['Breed'], axis=1, inplace=True)

    # 3.10) Process color information
    #   - Check for multiple colors
    #   - Check for patterns
    if preprocessingFlags['Colors']:
        df['numColors'] = df['Color'].apply(numColors)

    if preprocessingFlags['ColorsRappr']:
        df['Colors'] = extractColor(df['Color'])
        for i in colorsList:
            df[i] = df['Colors'].apply(lambda x: checkColor(x, i))

        # Macro-colors
        for i in colorsAssociations:
           df[i[0]+'Rappr']=(df[i].sum(axis=1)>0).astype(int)
           df.drop(i,axis=1,inplace=True)

        df.drop(['Colors'], axis=1, inplace=True)

    if preprocessingFlags['Patterns']:
        df['Pattern'] = df['Color'].apply(hasPattern)

    df.drop(['Color'], axis=1, inplace=True)

    # 4) Reorder columns in training and test set
    cols = df.columns.tolist()
    cols.remove('OutcomeType')
    cols += ['OutcomeType']
    df = df[cols]

    # 5) Split training and test set
    train = df.ix[df['OutcomeType'] != 'Test', :]
    test = df.ix[df['OutcomeType'] == 'Test', :]
    test.drop('OutcomeType', axis=1, inplace=True)
    train.sort_index(inplace=True)
    test.sort_index(inplace=True)

    print train.head()
    print test.head()


    # 5) Write training set and test set to .csv files
    train.to_csv(inputDir + 'preprocessed_train.csv', index=False)
    test.to_csv(inputDir + 'preprocessed_test.csv')


if __name__ == "__main__":

    # Input directory
    inputDir = '../Input/'

    # Perform preprocessing of training and test sets
    preprocessing(inputDir)
