import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.tree as tr

sns.set_style("whitegrid")
sns.set_style("ticks")


train = pd.read_csv('../Input/train.csv', header=0,
                    names=['ID', 'Name', 'Date', 'Out', 'SubOut', 'Type', 'Sex', 'Age', 'Breed', 'Color'],
                    parse_dates=['Date'])

test = pd.read_csv('../Input/test.csv', header=0,
                   names=['ID', 'Name', 'Date', 'Type', 'Sex', 'Age', 'Breed', 'Color'],
                   parse_dates=['Date'])

# Riduzione di train: elimino Subout, ID, Name
train.drop(['ID', 'Name', 'SubOut'], axis=1, inplace=True)

# Riduzione di test: elimino ID, Name
test.drop(['ID', 'Name'], axis=1, inplace=True)


# Eliminazione missing data
train.dropna(inplace=True)

# Valori unici
Uniq_train = {'Out': train.Out.unique(), 'Type': train.Type.unique(),
              'Sex': train.Sex.unique(), 'Age': train.Age.unique(),
              'Breed': train.Breed.unique(), 'Color': train.Color.unique()}

Uniq_test = {'Type': test.Type.unique(),
             'Sex': test.Sex.unique(), 'Age': test.Age.unique(),
             'Breed': test.Breed.unique(), 'Color': test.Color.unique()}

# Confronto tra valori unici
# for i in Uniq_train.keys():
#     print(i, ':', Uniq_train[i].size)
#
# for i in Uniq_test.keys():
#     print(i, ':', Uniq_test[i].size)

# Riordino per data
train.sort_values('Date', inplace=True)
train.reset_index(drop=True, inplace=True)

# Recupero informazioni temporali
train['Hour'] = [h.hour for h in train.Date]
train['Month'] = [h.month for h in train.Date]

# Giorno della settimana
def compute_day(dates, start_day):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    n = days.index(start_day)
    up_days = days[n:]+days[:n]
    diffs = ((dates-dates[0])/np.timedelta64(1, 'D')).astype(int)
    day_name = []
    for dd in diffs:
        day_name.append(up_days[dd % 7])
    return day_name

# Il 1 ottobre 2013 era un martedì
train['Day of the week']=compute_day(train.Date,'Tue')


# Trasformazione dell'età e delle date su scala numerica
NumAge = [int(s) for age in train.Age for s in age.split() if s.isdigit()]  # List comprehension, python 3
for i in range(len(NumAge)):
    age = train.Age[i]
    if age.find('week') != -1:
        NumAge[i] = NumAge[i] * 7 + 3
    elif age.find('month') != -1:
        NumAge[i] = NumAge[i] * 30 + 15
    elif age.find('year') != -1:
        NumAge[i] = NumAge[i] * 365 + 150
    else:
        NumAge[i] = NumAge[i]

NumDate = ((train.Date-train.Date[0])/np.timedelta64(1, 'h')).astype(int)

train.Age = NumAge
train.Date = (NumDate+9)/24

# Fasce d'età definite sui percentili
def age_ranges(age,n_ranges):
    temp = range(n_ranges+1)
    val = np.empty(len(age),dtype=int)
    perc = [0]+list(np.percentile(age,np.linspace(10,100,n_ranges).tolist()))
    for i,j in zip(range(len(val)),age):
        for k in temp:
            if j > perc[k]:
                val[i] = k + 1
    return val
train['AgeRange'] = age_ranges(train.Age, 10)

# Prime ispezioni grafiche
# plt.figure()
# sns.countplot(x='Type', data=train)
# plt.title('Proporzioni cani/gatti')



plt.figure()
sns.countplot(x='Out', hue='Type', data=train)
plt.title('Outcomes with respect to animal type')
sns.despine()

plt.figure()
sns.countplot(x='Out', hue='Sex', data=train)
plt.title('Outcomes with respect to sex')
sns.despine()

# plt.figure()
# sns.violinplot(x='Out', y='Age', hue='Type', data=train)
# plt.figure()
# sns.violinplot(x='Out', y='Date', hue='Type', data=train)
# plt.figure()
# sns.distplot(train.Age)

# plt.figure()
# sns.countplot(x='AgeRange', hue='Type',data=train)
# plt.figure()
# sns.countplot(x=train.AgeRange, hue=train.Out)
# sns.countplot(x=train.AgeRange[train.Type == 'Dog'], hue=train.Out,palette="dark")
# plt.title('Differenze tra cani e gatti per fasce d\'età e outcome')


# plt.figure()
# sns.countplot(x='Out',hue='Month',data=train)
plt.figure()
sns.countplot(x='Hour',hue='Out',data=train)
plt.title('Outcome trends with respect to hour')
sns.despine()

plt.figure()
for i in train.Out.unique():
    sns.distplot(train.ix[train.Out==i,'Hour'],norm_hist=False,hist=False)
plt.title('Outcome trends with respect to hour')
sns.despine()

sns.set_palette(sns.color_palette('hls',7))
plt.figure()
train.ix[0,'Day of the week']='Mon'
sns.countplot(x='Out',hue='Day of the week',data=train)
plt.title('Outcomes with respect to the day of the week')
sns.despine()

plt.show()


sns.set_context("notebook", font_scale=1.5)
sns.heatmap(df_sign.corr(),cmap="PuOr",mask=mask)
plt.yticks(rotation=0)
plt.show()
plt.title("Correlation matrix between outcome types and subtypes")
