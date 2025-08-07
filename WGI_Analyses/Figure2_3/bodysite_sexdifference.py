import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pingouin as pg

#--------------------------------#
#Code for Determining Sex Differences in Pain Sites
# Used in Figure 2B
#--------------------------------#

#Read in data
df = pd.read_csv('/PATH/UKB_500K_pain_updated.csv')

#invert sex coding so that postive Odds ratios are for females
df['Sex_T0'] = np.abs(df['Sex_T0']-1)

#limit number of chronic pain types to 4+ max
df ['NumberChronicPainTypes_T0'] = df['NumberChronicPainTypes_T0'].apply(lambda x: 4 if x>4 else x)

#list of pain types
pain_list = ['ChronicWidespreadPain_T0','ChronicNeckShoulderPain_T0', 'ChronicHipPain_T0', 'ChronicBackPain_T0','ChronicStomachAbdominalPain_T0', 'ChronicKneePain_T0','ChronicHeadaches_T0', 'ChronicFacialPain_T0']

# Calculate Odds Ratios and Confidence Intervals
#create empty dataframe to store results
df_cols = ['Category','OR','se','pvalue','n','CI lower','CI upper']
results_df = pd.DataFrame(columns=df_cols)

#loop through each pain type
for pain in pain_list:
    #drop rows with missing data for the pain type
    df_new = df.dropna(subset=[pain])
    #run logistic regression
    lr_or = pg.logistic_regression(df_new['Sex_T0'],df_new[pain],remove_na=True)
    new = pd.DataFrame()
    new['Category'] = [pain]
    new['OR'] = [np.exp(lr_or['coef'][1])]
    new['se'] = [lr_or['se'][1]]
    new['pvalue'] = [lr_or['pval'][1]]
    new['n'] = [df_new.dropna(subset=[pain]).shape[0]]
    new['CI lower'] = [np.exp(lr_or['CI[2.5%]'][1])]
    new['CI upper'] = [np.exp(lr_or['CI[97.5%]'][1])]
    results_df = pd.concat([results_df,new])

#print results
print(results_df)
#save results to csv
results_df.to_csv('/PATH/bodysite_sex_diffs.csv')
