################################################################################
# Description: Shelter Animal Outcomes
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        mer 01 giu 2016 14:45:50 CEST
################################################################################

import seaborn as sns

sns.set(style='ticks',
        color_codes=True,
        context='poster',
        rc={"figure.figsize": (20, 15)})

# 1.3) Exploratory Plots




        sns.violinplot(x='DateTime',
                    y='OutcomeType',
                    data=train,
                    hue='AnimalType',
                    bw=.2)

        sns.violinplot(x='AgeuponOutcome',
                    y='OutcomeType',
                    data=train,
                    hue='AnimalType',
                    bw=.2)

        g = sns.factorplot(x='Name',
                        y='AgeuponOutcome',
                        row='AnimalType',
                        col='OutcomeType',
                        hue='SexuponOutcome',
                        data=train,
                        size=8,
                        kind='bar')



