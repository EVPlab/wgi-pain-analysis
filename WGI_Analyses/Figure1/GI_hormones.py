import pandas as pd
from statsmodels.stats.proportion import proportion_confint
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt

#--------------------------------#
#Correlation of Hormones with Gender Index
# Used in Figure 1F
#--------------------------------#

#Read in data
df = pd.read_csv('/PATH/UKB_500k_Demographics.csv')
gi=pd.read_csv('/PATH/new_GI_probabilities_I.csv')
hormones = pd.read_csv('/PATH/tando.csv')
ft = pd.read_csv('/PATH/Free_Testosterone.csv')

hormones = hormones[['eid','Testosterone_T0','Oestradiol_T0']]
ft = ft[['eid','ft']]

df = df.merge(hormones, on='eid',how='left')
df = df.merge(ft, on='eid',how='left')
df = df.merge(gi, on='eid',how='left')

#Code Gender Index so that higher values are more feminine
df['Gender_Index'] = (df['Meta_Model_Prob']*-1)+1

#Split data into female and male
female = df[df['Sex_T0']==0]
male = df[df['Sex_T0']==1]

#Partial Correlations of hormones with Gender Index
print(pg.partial_corr(female,'Gender_Index','ft',covar=['Age_T0']))
print(pg.partial_corr(male,'Gender_Index','ft',covar=['Age_T0']))
print(pg.partial_corr(female,'Gender_Index','Oestradiol_T0',covar=['Age_T0']))
print(pg.partial_corr(male,'Gender_Index','Oestradiol_T0',covar=['Age_T0']))
print(pg.partial_corr(female,'Gender_Index','Testosterone_T0',covar=['Age_T0']))
print(pg.partial_corr(male,'Gender_Index','Testosterone_T0',covar=['Age_T0']))

