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
#        train['Named'] = (train['NameLength'] > 0).astype(int)
#        test['Named'] = (test['NameLength']).astype(int)
    train.drop(['Name'], axis=1, inplace=True)
    test.drop(['Name'], axis=1, inplace=True)

    # 3.3) Remove instances with NaN values from training set (only 19)
    train.dropna(inplace=True)

    # 3.4) Sort for increasing timedate and reindex
    train.sort_values(by='DateTime', inplace=True)
    train.reset_index(drop=True, inplace=True)

    # 3.5) Process dates
    #   - Split dates in Year, Month, Day, Hour Attributes
    #   - Check day of the week
    #   - Check for holidays

    # referenceDate = datetime.date(2013, 01, 01)
    # train['Date'] = (train['DateTime'] -
    #                 referenceDate).astype('timedelta64[D]').astype(float)
    # train['Year'] = train['DateTime'].apply(lambda x: x.year)
    if preprocessingFlags['Year']:
        train['Year'] = train['DateTime'].apply(lambda x: x.year)
        test['Year'] = test['DateTime'].apply(lambda x: x.year)

    if preprocessingFlags['DayOfYear']:
        train['DayOfYear'] = train['DateTime'].apply(lambda x: x.dayofyear)
        test['DayOfYear'] = test['DateTime'].apply(lambda x: x.dayofyear)

    if preprocessingFlags['DayMonth']:
        train['Month'] = train['DateTime'].apply(lambda x: x.month)
        train['Day'] = train['DateTime'].apply(lambda x: x.day)
        train['WeekDay'] = train['DateTime'].apply(lambda x: x.isoweekday())
        test['Month'] = test['DateTime'].apply(lambda x: x.month)
        test['Day'] = test['DateTime'].apply(lambda x: x.day)
        test['WeekDay'] = test['DateTime'].apply(lambda x: x.isoweekday())

    if preprocessingFlags['MinuteHour']:
        train['Hour'] = train['DateTime'].apply(lambda x: x.hour)
        # train['0or9'] = train['Hour'].apply(lambda x: x == 0 or x == 9)
        # df = pd.get_dummies(train['Hour'].apply(hourTrend),columns=['ZeroNine', 'Rest', 'Afternoon', 'Evening'])
        # train[df.columns] = df
        train['Minute'] = train['DateTime'].apply(lambda x: x.minute)
        train['MinuteOfDay'] = train['Hour'] * 60 + train['Minute']
        test['Hour'] = test['DateTime'].apply(lambda x: x.hour)
        # test['0or9'] = test['Hour'].apply(lambda x: x == 0 or x == 9)
        # df = pd.get_dummies(test['Hour'].apply(hourTrend),columns=['ZeroNine', 'Rest', 'Afternoon', 'Evening'])
        # test[df.columns] = df
        test['Minute'] = test['DateTime'].apply(lambda x: x.minute)
        test['MinuteOfDay'] = test['Hour'] * 60 + test['Minute']
        # train.drop(['MinuteOfDay'], axis=1, inplace=True)
        # test.drop(['MinuteOfDay'], axis=1, inplace=True)

    #---------------------------------------------------------------------------
    # Alberto
    outcomeLabels = train['OutcomeType']

    transferTimes = train['DateTime']
    decalage1 = transferTimes[1:].reset_index(drop=True)
    decalage2 = transferTimes[:-1].reset_index(drop=True)
    deltaTransferTimes = ((decalage1-decalage2)/np.timedelta64(1,'m')).astype(int)

    dfIndexes = countAllOccurrences(deltaTransferTimes, outcomeLabels, 2)
    train[dfIndexes.columns] = dfIndexes

    # Add longer scales
 #    dfIndexes = countAllOccurrences(deltaTransferTimes, outcomeLabels, 30)
    # dfIndexes.columns = [s + 'Long' for s in dfIndexes.columns]
 #    train[dfIndexes.columns] = dfIndexes

    test['OutcomeType'] = 'Test'
    df1 = pd.DataFrame(train[['DateTime','OutcomeType']])
    df2 = pd.DataFrame(test[['DateTime','OutcomeType']])
    mergedDfs=df1.append(df2).reset_index().sort_values('DateTime')

    outcomeMergedLabels = mergedDfs['OutcomeType'].reset_index(drop=True)

    transferTimes = mergedDfs['DateTime']
    decalage1 = transferTimes[1:].reset_index(drop=True)
    decalage2 = transferTimes[:-1].reset_index(drop=True)
    deltaTransferTimes = ((decalage1 - decalage2) / np.timedelta64(1, 'm')).astype(int)

    df2Indexes = countAllOccurrences(deltaTransferTimes, outcomeMergedLabels, 2)

    df2Indexes['INDEX']=mergedDfs['index'].reset_index(drop=True)
    df2Indexes = df2Indexes.ix[outcomeMergedLabels == 'Test',].set_index('INDEX').sort_index()

    test[df2Indexes.columns] = df2Indexes

    # Add longer scales
#     df2Indexes = countAllOccurrences(deltaTransferTimes, outcomeMergedLabels, 30)

    # df2Indexes['INDEX']=mergedDfs['index'].reset_index(drop=True)
    # df2Indexes = df2Indexes.ix[outcomeMergedLabels == 'Test',].set_index('INDEX').sort_index()

    # df2Indexes.columns = [s + 'Long' for s in df2Indexes.columns]
    # test[df2Indexes.columns] = df2Indexes


#     for i in df2Indexes.columns:
#         test[i] = df2Indexes[i]

#TODO: se il valore e troppo alto la predizione potrebbe essere inaccurata: come considerarli?
#Per il momento rimuovo per coerenza rispetto a quanto in training
    test.drop('TestTog',axis=1, inplace=True)

    #---------------------------------------------------------------------------

    train.drop('DateTime', axis=1, inplace=True)
    test.drop('DateTime', axis=1, inplace=True)

    # 3.6) Group animal ages in tranches and convert ages to days
    if preprocessingFlags['Age']:
        train['AgeDays'] = train['AgeuponOutcome'].apply(convertAges)
        # df = pd.get_dummies(train['AgeDays'].apply(groupAges))
        # train[df.columns] = df
        medianAge = train['AgeDays'].median()
        test['AgeDays'] = test['AgeuponOutcome'].apply(lambda x: convertAges(x, medianAge))
        test['GroupAges'] = test['AgeDays'].apply(groupAges)
        # df = pd.get_dummies(test['AgeDays'].apply(groupAges))
        # test[df.columns] = df

    train.drop('AgeuponOutcome', axis=1, inplace=True)
    test.drop('AgeuponOutcome', axis=1, inplace=True)

    # 3.7) Binarize AnimalType: 0 for cat, 1 for dog
    if preprocessingFlags['AnimalType']:
        train.ix[train['AnimalType'] == 'Cat', 'AnimalType'] = 0
        train.ix[train['AnimalType'] == 'Dog', 'AnimalType'] = 1
        test.ix[test['AnimalType'] == 'Cat', 'AnimalType'] = 0
        test.ix[test['AnimalType'] == 'Dog', 'AnimalType'] = 1

    # 3.8) Process SexuponOutcome attribute
    #   - Extract Male/Female boolean variable
    #   - Extract Intact/Neutered boolean variable
    if preprocessingFlags['Sex']:
        train['Gender'] = train['SexuponOutcome'].apply(getGender)
        test['Gender'] = test['SexuponOutcome'].apply(getGender)

    if preprocessingFlags['Status']:
        train['Status'] = train['SexuponOutcome'].apply(getStatus)
        test['Status'] = test['SexuponOutcome'].apply(getStatus)

    train.drop('SexuponOutcome', axis=1, inplace=True)
    test.drop('SexuponOutcome', axis=1, inplace=True)

    # 3.9) Process Breed information
    #   - Check if animal is a mix
    if preprocessingFlags['Mix']:
        train['Mix'] = train['Breed'].apply(lambda x: 'Mix' in x).astype(int)
        train['MultiBreed'] = train['Breed'].apply(lambda x: '/' in x).astype(int)
        test['Mix'] = test['Breed'].apply(lambda x: 'Mix' in x).astype(int)
        test['MultiBreed'] = test['Breed'].apply(lambda x: '/' in x).astype(int)

    #   - Check if animal is a pitbull
    if preprocessingFlags['Breeds']:
        # 3.9.1) Extract breeds macro-groups
        # TODO: at the moment, only dogs are considered. Process also cats
        # Training set
        df = pd.DataFrame(0.0,
                        index=train.index,
                        columns=uniqueBreedGroupDog)

        breedGroupsTrain = train['Breed'].apply(lambda x: getBreedGroups(x,
                                                mappingBreedGroupDog))

        for i in xrange(len(df)):
            if len(breedGroupsTrain[i]) > 0:
                df.ix[i, breedGroupsTrain[i]] = 1.0
        train = train.join(df)

        # Test set
        df = pd.DataFrame(0.0,
                        index=test.index,
                        columns=uniqueBreedGroupDog)

        breedGroupsTest = test['Breed'].apply(lambda x: getBreedGroups(x,
                                            mappingBreedGroupDog))

        for i in df.index:
            if len(breedGroupsTest[i]) > 0:
                df.ix[i, breedGroupsTest[i]] = 1.0
        test = test.join(df)

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

        train['Dangerous'] = train['Breed'].apply(isDangerous).astype(int)
        test['Dangerous'] = test['Breed'].apply(isDangerous).astype(int)

        train['Domestic'] = train['Breed'].apply(lambda x: 'Domestic' in x).astype(int)
        test['Domestic'] = test['Breed'].apply(lambda x: 'Domestic' in x).astype(int)

    train.drop(['Breed'], axis=1, inplace=True)
    test.drop(['Breed'], axis=1, inplace=True)

    # 3.10) Process color information
    #   - Check for multiple colors
    #   - Check for patterns
    if preprocessingFlags['Colors']:
        train['numColors'] = train['Color'].apply(numColors)
        test['numColors'] = test['Color'].apply(numColors)

    if preprocessingFlags['ColorsRappr']:
        train['Colors'] = extractColor(train['Color'])
        for i in colorsList:
            train[i] = train['Colors'].apply(lambda x: checkColor(x, i))

        test['Colors'] = extractColor(test['Color'])
        for i in colorsList:
            test[i] = test['Colors'].apply(lambda x: checkColor(x, i))

        # Macro-colors
        for i in colorsAssociations:
           train[i[0]+'Rappr']=(train[i].sum(axis=1)>0).astype(int)
           train.drop(i,axis=1,inplace=True)

        for i in colorsAssociations:
           test[i[0] + 'Rappr'] = (test[i].sum(axis=1) > 0).astype(int)
           test.drop(i, axis=1, inplace=True)

        train.drop(['Colors'], axis=1, inplace=True)
        test.drop(['Colors'], axis=1, inplace=True)

    if preprocessingFlags['Patterns']:
        train['Pattern'] = train['Color'].apply(hasPattern)
        test['Pattern'] = test['Color'].apply(hasPattern)

    train.drop(['Color'], axis=1, inplace=True)
    test.drop(['Color'], axis=1, inplace=True)

    # 4) Reorder columns in training and test set
    cols = train.columns.tolist()
    cols.remove('OutcomeType')
    cols += ['OutcomeType']
    train = train[cols]
    test = test[cols[:-1]]

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
