import pandas as pd

from mappings import *

def convertAges(age, defaultAge=None):
    """ Convert animal ages from string to number of days.

    Args:
        age (str): animal age expressed in string
        defaultAge (float): defalt age to return if NaN is found

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
    if age < 60.0:
        return 0
    elif age < 1000.0:
        return 1
    elif age < 3650.0:
        return 2
    else:
        return 3

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


def getGender(sex):
    """ Extract animal gender from SexuponOutcome instance.

    Args:
        sex (str): SexuponOutcome instance

    Returns:
        gender (str): Animal gender
    """
    if sex == 'Unknown':
        return 2
    else:
        [status, gender] = sex.split()
        if gender == 'Male':
            return 0
        else:
            return 1

def getStatus(sex):
    """ Extract animal status from SexuponOutcome instance.

    Args:
        sex (str): SexuponOutcome instance

    Returns:
        status (str): Animal status
    """
    if sex == 'Unknown':
        return 2
    else:
        [status, gender] = sex.split()
        if status == 'Spayed' or status == 'Neutered':
            return 1
        else:
            return 0

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


def countAllOccurrences(deltaTimes, outcomes, intAmpl):
    togetherIndexes = {x: [] for  x in outcomes.unique()}
    memoryTimes = [0]
    memoryIndexes = [0]

    df = pd.DataFrame()
    for i in togetherIndexes.keys():
        df[i + 'Tog'] = [0]*(len(deltaTimes)+1)

    for i,j in zip(deltaTimes,range(1,len(deltaTimes)+1)):
        # print(memoryIndexes)
        memoryTimes.append(i)
        memoryIndexes.append(j)
        while sum(memoryTimes) > intAmpl:
            memoryTimes.pop(0)
            memoryTimes[0] = 0
            memoryIndexes.pop(0)
        if len(memoryTimes) > 0:
            df.ix[memoryIndexes[:-1],outcomes[j]+ 'Tog'] += 1
            for k in memoryIndexes[:-1]:
                df.ix[j,outcomes[k]+'Tog'] += 1
        # else:
        #     memoryTimes.append(0)
        #     memoryIndexes.append(j)

    return df

