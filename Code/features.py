import numpy as np
import pandas as pd

############
# Mappings #
############
from matplotlib import transforms

from mappings import *
from settings import *

##########################
# User-defined functions #
##########################

def convertAges(age, defaultAge=None):
    """ Convert animal ages from string to number of days.

    Args:
        age (str): animal age expressed in string
        defaultAge (float): default age to return if NaN is found

    Returns:
        ageNum (float): animal age expressed in number of days
    """

    if isinstance(age, str):
        [n, u] = age.split()
        n = float(n)
        if 'day' in u:
            return n
        elif 'week' in u:
            return 7.0 * n + 3.5
        elif 'month' in u:
            return 30.0 * n + 15.0
        elif 'year' in u:
            return 365.0 * n + 182.5
    else:
        return defaultAge


def groupAges(age):
    """ Group animal age in four different tranches
        - infant: age < 1 month
        - young: age < 1 year
        - adult: 1 year <= age < 10 years
        - senior: age >= 10 years

    Args:
        age (str): animal age in string

    Returns:
        ageGroup (str): age group
    """
    if age < 30.0:
        return 'Infant'
    elif age < 365.0:
        return 'Young'
    elif age < 3650.0:
        return 'Adult'
    else:
        return 'Senior'


def ageRanges(age,n_ranges):
    """ Assign the elements of the list to the corresponding percentile ranges

    Args:
        age (list): list of age values
        n_ranges (int): number of ranges

    Returns:

    """
    val = np.empty(len(age),dtype=int)

    # Creation of the percentile delimiters
    perc = [0]+list(np.percentile(age,np.linspace(100/n_ranges,100,n_ranges).tolist()))

    # Assignment by comparison
    for i,j in zip(range(len(val)),age):
        for k in range(n_ranges+1):
            if j > perc[k]:
                val[i] = k + 1
    return val


def getGender(sex):
    """ Extract animal gender from SexuponOutcome instance.

    Args:
        sex (str): SexuponOutcome instance

    Returns:
        gender (str): Animal gender
    """
    if sex == 'Unknown':
        return sex
    else:
        [status, gender] = sex.split()
        return gender


def getStatus(sex):
    """ Extract animal status from SexuponOutcome instance.

    Args:
        sex (str): SexuponOutcome instance

    Returns:
        status (str): Animal status
    """
    if sex == 'Unknown':
        return sex
    else:
        [status, gender] = sex.split()
        if status == 'Spayed':
            status = 'Neutered'
        return status


def isMix(breed):
    """ Check if animal is a mix or a cross breed.

    Args:
        breed (str): Animal breed

    Returns:
        flag (int): 1 if mix, 0 otherwise
    """
    if 'Mix' in breed or '/' in breed:
        return 1
    else:
        return 0


def getUniqueBreeds(s):
    """ Generate list of unique breeds from series of breeds (potentially mixed)

    Args:
        s (pd.Series): series of potentially mixed breeds

    Returns:
        uniqueBreeds (list): list of unique breeds
    """
    # Split cross-breeds
    breeds = [i.split('/') for i in s]
    breeds = [j for i in breeds for j in i]

    # Remove 'Mix' from the breeds name and consider 'Mix' as separate breed
    breeds = [b.replace(' Mix', '') for b in breeds]
    breeds += ['Mix']

    # Get unique breeds
    uniqueBreedsDog = list(set(breeds))
    return uniqueBreedsDog


def getBreedGroups(b, mapping):
    """ Split breed combination into individual breeds.

    Args:
        b (string): breed combination

    Returns:
        breeds (list): list of individual breeds

    """
    breeds = b.split('/')
    if 'Mix' in breeds[-1]:
        breeds[-1] = breeds[-1][:-4]
        breeds += ['Mix']
    breedGroups = map(mapping.get, breeds)
    breedGroups = [x for x in breedGroups if x is not None]
    return breedGroups


def numColors(s):
    return len(s.split('/'))


def hasPattern(s):
    colors = s.split()
    for c in colors:
        if c in patternsList:
            return 1
    return 0


def clearAttributes(phrase, keywords):
    """ Remove the keywords contained in the phrase string

    Args:
        phrase (str): initial string
        keywords (list): strings to be removed

    Returns:
        phrase (str): phrase without the keywords

    """
    for i in keywords:
        phrase=phrase.replace(i,'')
    return phrase.replace('/',' ')


def extractColor(colors):
    """

    Args:
        colors (list): list of color strings

    Returns:
        cleanedColors (list): list of lists of distinct colors

    """
    cleanedColors = [clearAttributes(i,patternsList).split() for i in colors]
    cleanedColors = [i if i else ['NoColor'] for i in cleanedColors]
    return cleanedColors


def checkColor(value, check):
    """

    Args:
        value (list): list of colors
        check (str): color to check

    Returns:
        flag (int): 1 if the color is present, 0 otherwise

    """
    for i in value:
        if i == check:
            return 1
    return 0

#TODO try to split 'Rest' into 'Early morning' and 'Night'
def hourTrend(hour):
    """

    Args:
        hour (int): hour passed

    Returns

    """
    if hour == 0 or hour ==9:
        return 'ZeroNine'
    if hour <=10:
        return 'Rest'
    if hour <= 15:
        return 'Afternoon'
    if hour <= 19:
        return 'Evening'
    return 'Rest'


def preprocessing(inputDir):
    """ Preprocessing of shelter animals training and test set. The preprocessed
        datasets are written to the input Directory for future reuse.

    Args:
        inputDir (str): path to input directory
    """

    # 1) Read training set
    train = pd.read_csv(inputDir + 'train.csv',
                        parse_dates=['DateTime'])

    # 2) Read test set
    test = pd.read_csv(inputDir + 'test.csv',
                       index_col=0,
                       parse_dates=['DateTime'])

    # 3) Preprocessing

    # 3.1) Removing useless attributes
    #   - AnimalID does not contain duplicates
    #   - OutcomeSubtype is not used for evaluation and is not given in test set
    train.drop(['AnimalID', 'OutcomeSubtype'], axis=1, inplace=True)

    # 3.2) Name attribute:
    #   - Replace Name attribute with boolean: 1 if animal is named, 0 if not.
    #   - Add column with the length of the name
    if preprocessingFlags['Name']:
        train['Name'].fillna('',inplace=True)
        train['LengthName'] = train['Name'].apply(len)
        train['NumberOfNames'] = train['Name'].apply(lambda x: len(x.split()))
        train['Named'] = train['LengthName'].astype(bool).astype(int)

        test['Name'].fillna('', inplace=True)
        test['LengthName'] = test['Name'].apply(len)
        test['NumberOfNames'] = test['Name'].apply(lambda x: len(x.split()))
        test['Named'] = test['LengthName'].astype(bool).astype(int)

    train.drop('Name', axis=1, inplace=True)
    test.drop('Name', axis=1, inplace=True)

    # 3.3) Remove instances with NaN values from training set (only 19)
    train.dropna(inplace=True)

    # 3.4) Sort for increasing timedate and reindex
    train.sort_values(by='DateTime', inplace=True)
    train.reset_index(drop=True, inplace=True)

    # 3.5.1) Process dates
    #   - Split dates in Year, Month, Day, Hour Attributes
    #   - Check day of the week
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
        train['ZeroMinute'] = (train['DateTime'].apply(lambda x: x.minute) == 0).astype(int)
        df = pd.get_dummies(train['Hour'].apply(hourTrend),columns=['ZeroNine', 'Rest', 'Afternoon', 'Evening'])
        train[df.columns] = df

        test['Hour'] = train['DateTime'].apply(lambda x: x.hour)
        test['ZeroMinute'] = (test['DateTime'].apply(lambda x: x.minute)==0).astype(int)
        df = pd.get_dummies(test['Hour'].apply(hourTrend), columns=['ZeroNine', 'Rest', 'Afternoon', 'Evening'])
        test[df.columns] = df

    # 3.5.2) "Moving" analysis of transfer outcome
    #FIXME: in realtà dovrebbe funzionare tutto fino al punto 3.6), ma se qualcuno volesse controllare a mano alcuni
    #FIXME valori mi farebbe molto piacere. Alberto

    transferTimes = train.ix[train['OutcomeType'] == 'Transfer', 'DateTime']
    decalage1 = transferTimes[1:].reset_index(drop=True)
    decalage2 = transferTimes[:-1].reset_index(drop=True)
    deltaTransferTimes=((decalage1-decalage2)/np.timedelta64(1,'m')).astype(int)
    train['TransferTogether']=0
    togetherIndexes = [j for i,j in zip(deltaTransferTimes,range(len(deltaTransferTimes))) if i <= 5]
    togetherIndexes = list (set (togetherIndexes + [i+1 for i in togetherIndexes]))
    train.ix[transferTimes.reset_index().ix[togetherIndexes,'index'], 'TransferTogether'] = 1

    test['TransferTogether']=0
    df1=pd.DataFrame(transferTimes)
    df1['Marker']=0
    df2=pd.DataFrame(test['DateTime'].sort_values())
    df2['Marker']=1
    mergedDfs=df1.append(df2).reset_index().sort_values('DateTime')
    transferTimes = mergedDfs['DateTime']
    decalage1 = transferTimes[1:].reset_index(drop=True)
    decalage2 = transferTimes[:-1].reset_index(drop=True)
    deltaTransferTimes = ((decalage1 - decalage2) / np.timedelta64(1, 'm')).astype(int)
    test['TransferTogether'] = 0
    togetherIndexes = [j for i, j in zip(deltaTransferTimes, range(len(deltaTransferTimes))) if i <= 5]
    togetherIndexes = list(set(togetherIndexes + [i + 1 for i in togetherIndexes]))
    mergedDfs.reset_index(inplace=True)
    testTransferIndexes = [i for i,j in zip(togetherIndexes, mergedDfs.ix[togetherIndexes,'Marker']) if j == 1]
    test.ix[mergedDfs.ix[testTransferIndexes,'index'], 'TransferTogether'] = 1


    train.drop('DateTime', axis=1, inplace=True)
    test.drop('DateTime', axis=1, inplace=True)

    # 3.6) Convert animal ages to days
    if preprocessingFlags['Age']:
        train['AgeDays'] = train['AgeuponOutcome'].apply(convertAges)
        medianAge = train.loc[train['AnimalType'] == 'Cat','AgeDays'].median()
        test['AgeDays'] = test['AgeuponOutcome'].apply(lambda x: convertAges(x, medianAge))

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
    #   - Extract breed group
    #   - Check if animal is a mix
    #   - Check if animal is a pitbull
    if preprocessingFlags['Breeds']:

        # 3.9.1) Extract breeds macro-groups
        # TODO: at the moment, only dogs are considered. Process also cats
        # Training set
        df = pd.DataFrame(0.0,
                        index=train.index,
                        columns=uniqueBreedGroupDog + ['Mix'])

        breedGroupsTrain = train['Breed'].apply(lambda x: getBreedGroups(x,
                                                mappingBreedGroupDog))

        for i in range(len(df)):
            if len(breedGroupsTrain[i]) > 0:
                df.ix[i, breedGroupsTrain[i]] = 1.0
        train = train.join(df)

        # Test set
        df = pd.DataFrame(0.0,
                        index=test.index,
                        columns=uniqueBreedGroupDog + ['Mix'])

        breedGroupsTest = test['Breed'].apply(lambda x: getBreedGroups(x,
                                            mappingBreedGroupDog))

        for i in df.index:
            if len(breedGroupsTest[i]) > 0:
                df.ix[i, breedGroupsTest[i]] = 1.0
        test = test.join(df)

        # Check if dog is a pitbull
        # TODO: We might generalize this to dogs who are perceived as aggressive
        train['PitBull'] = train['Breed'].apply(lambda x: int('Pit Bull' in x))
        test['PitBull'] = test['Breed'].apply(lambda x: int('Pit Bull' in x))

    train.drop(['Breed'], axis=1, inplace=True)
    test.drop(['Breed'], axis=1, inplace=True)

    # 3.10) Process color information
    #   - Extract colors.
    #   - Check for patterns

    if preprocessingFlags['Colors']:
        train['Colors'] = extractColor(train['Color'])
        # for i in colorsList:
        #     train[i] = train['Colors'].apply(lambda x: checkColor(x, i))
        #
        test['Colors'] = extractColor(test['Color'])
        # for i in colorsList:
        #     test[i] = test['Colors'].apply(lambda x: checkColor(x, i))

        # Macro-colors
        #for i in colorsAssociations:
        #    train[i[0]+'Rappr']=(train[i].sum(axis=1)>0).astype(int)
        #    train.drop(i,axis=1,inplace=True)

        #for i in colorsAssociations:
        #    test[i[0] + 'Rappr'] = (test[i].sum(axis=1) > 0).astype(int)
        #    test.drop(i, axis=1, inplace=True)

    # Pattern
    if preprocessingFlags['Patterns']:
        train['Pattern'] = train['Color'].apply(hasPattern)
        test['Pattern'] = test['Color'].apply(hasPattern)

    # 3.11) Drop Breed and Color columns
    train.drop(['Color', 'Colors'], axis=1, inplace=True)
    test.drop(['Color', 'Colors'], axis=1, inplace=True)


    # 4) Reorder columns in training and test set
    cols = train.columns.tolist()
    cols.remove('OutcomeType')
    cols += ['OutcomeType']
    train = train[cols]
    test = test[cols[:-1]]

    print(train.head())
    print(test.head())

    # 5) Write training set and test set to .csv files
    train.to_csv(inputDir + 'preprocessed_train.csv', index=False)
    test.to_csv(inputDir + 'preprocessed_test.csv')


if __name__ == "__main__":

    # Input directory
    inputDir = '../Input/'

    # Perform preprocessing of training and test sets
    preprocessing(inputDir)

