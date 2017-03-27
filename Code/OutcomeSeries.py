import numpy as np
import pandas as pd

############
# Mappings #
############
from matplotlib import transforms

from mappings import *
from settings import *


def intervalOccurrences(deltaTimes, outcomes, intAmpl, nbOcc):
    """
    Args:
        deltaTransferTimes (list): list of time differences
        outcomes (list): list of outcomes
        intAmpl (int): interval amplitude
        nbOcc (int): number of occurrences

    Returns:
        togetherIndexes (dict): dict of lists containing the indexes of the data satisfying
                                the conditions for each outcome
    """
    togetherIndexes = {'Adoption': [], 'Transfer': [], 'Return_to_owner': [], 'Euthanasia': [],'Died': []}
    memoryTimes = {'Adoption': [0], 'Transfer': [0], 'Return_to_owner': [0], 'Euthanasia': [0],'Died': [0]}
    memoryIndexes = {'Adoption': [], 'Transfer': [], 'Return_to_owner': [], 'Euthanasia': [],'Died': []}
    memoryFlags = {'Adoption': 0, 'Transfer': 0, 'Return_to_owner': 0, 'Euthanasia': 0, 'Died': 0}
    memoryIndexes[outcomes[0]] = [0]
    # FF=1 # Flag per il primissimo elemento, ma visto il dataset non é necessario
    for i,j in zip(deltaTimes,range(1,len(deltaTimes)+1)):
        mm = outcomes[j]
        for nn in memoryTimes.keys():
            memoryTimes[nn][-1] += i
        memoryTimes[mm][-1] -= i
        memoryTimes[mm].append(i)
        # if FF:
        #     memoryTimes[mm] = [i]
        #     FF = 0

        if sum(memoryTimes[mm][:(nbOcc-1)]) > intAmpl:
            if memoryFlags[mm] and memoryIndexes[mm]:
                togetherIndexes[mm].append(memoryIndexes[mm][0])
            memoryFlags[mm] = 0
            memoryTimes[mm] = [0]
            memoryIndexes[mm] = [j]
        elif sum(memoryTimes[mm]) <= intAmpl:
            memoryIndexes[mm].append(j)
        else:
            while len(memoryIndexes[mm]) >= nbOcc:
                togetherIndexes[mm].append(memoryIndexes[mm][0])
                memoryTimes[mm].pop(0)
                memoryIndexes[mm].pop(0)
                memoryFlags[mm] = 1
            if memoryFlags[mm]:
                if sum(memoryTimes[mm]) <= intAmpl:
                    memoryIndexes[mm].append(j)
                else:
                    while len(memoryIndexes[mm]) > 0 and sum(memoryTimes[mm]) >= intAmpl:
                        togetherIndexes[mm].append(memoryIndexes[mm][0])
                        memoryTimes[mm].pop(0)
                        memoryIndexes[mm].pop(0)
            else:
                while len(memoryIndexes[mm]) > 0 and sum(memoryTimes[mm]) >= intAmpl:
                    memoryTimes[mm].pop(0)
                    memoryIndexes[mm].pop(0)
    return togetherIndexes


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

    return df

trainFile = '../Input/preprocessed_train.csv'
testFile = '../Input/preprocessed_test.csv'
outputDir = '../Output/'

train = pd.read_csv(trainFile, parse_dates=['DateTime'])

test = pd.read_csv(testFile, index_col=0, parse_dates=['DateTime'])

outcomeLabels = train['OutcomeType']

transferTimes = train['DateTime']
decalage1 = transferTimes[1:].reset_index(drop=True)
decalage2 = transferTimes[:-1].reset_index(drop=True)
deltaTransferTimes = ((decalage1-decalage2)/np.timedelta64(1,'m')).astype(int)

# togetherIndexes = intervalOccurrences(deltaTransferTimes, outcomeLabels, 5, 3)
# for i in togetherIndexes.keys():
#     train[i+'Tog'] = 0
#     train.ix[togetherIndexes[i],i + 'Tog']=1

dfIndexes = countAllOccurrences(deltaTransferTimes, outcomeLabels, 5)

for i in dfIndexes.columns:
    train[i] = dfIndexes[i]

test['OutcomeType'] = 'Test'
df1 = pd.DataFrame(train[['DateTime','OutcomeType']])
df1['Marker'] = 0
df2 = pd.DataFrame(test[['DateTime','OutcomeType']])
df2['Marker'] = 1
mergedDfs=df1.append(df2).reset_index().sort_values('DateTime')

outcomeMergedLabels = mergedDfs['OutcomeType'].reset_index(drop=True)

transferTimes = mergedDfs['DateTime']
decalage1 = transferTimes[1:].reset_index(drop=True)
decalage2 = transferTimes[:-1].reset_index(drop=True)
deltaTransferTimes = ((decalage1 - decalage2) / np.timedelta64(1, 'm')).astype(int)

df2Indexes = countAllOccurrences(deltaTransferTimes, outcomeMergedLabels, 5)

df2Indexes['INDEX']=mergedDfs['index'].reset_index(drop=True)
df2Indexes = df2Indexes.ix[outcomeMergedLabels == 'Test',].set_index('INDEX').sort_index()
for i in df2Indexes.columns:
    test[i] = df2Indexes[i]

#TODO: se il valore é troppo alto la predizione potrebbe essere inaccurata: come considerarli?
#Per il momento rimuovo per coerenza rispetto a quanto in training
test.drop('TestTog',axis=1, inplace=True)

test[['DiedTog','TransferTog','AdoptionTog','Return_to_ownerTog','EuthanasiaTog']].head(50)

# 5) Write training set and test set to .csv files
train[['DiedTog','TransferTog','AdoptionTog','Return_to_ownerTog','EuthanasiaTog']].to_csv(outputDir + 'series_outcomes.csv', index=False)
test[['DiedTog','TransferTog','AdoptionTog','Return_to_ownerTog','EuthanasiaTog']].to_csv(outputDir + 'series_outcomes.csv')

