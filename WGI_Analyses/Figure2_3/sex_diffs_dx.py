import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pingouin as pg

#--------------------------------#
#Code for Determining Sex Differences in Diagnoses
# Used in Figure 3A
#--------------------------------#

#Read in Data
df=pd.read_csv('/PATH/UKB_500K_pain_updated.csv')
df = df[['eid','Sex_T0']]
dx = pd.read_csv('/PATH/UKB_Fig2/NCI_combined.csv')
df = df.merge(dx,on='eid',how='left')
#invert sex coding so that postive Odds ratios are for females
df['Sex_T0'] = np.abs(df['Sex_T0']-1)
#list of diagnoses
dx_list = ['NCI_fibromyalgia_T0','NCI_polymyalgia rheumatica_T0','NCI_cervical spondylosis_T0','NCI_spine arthritis/spondylitis_T0','NCI_joint pain_T0','NCI_back pain_T0','NCI_disc disorder_T0','NCI_trapped nerve/compressed nerve_T0','NCI_sciatica_T0','NCI_hernia_T0','NCI_irritable bowel syndrome_T0','NCI_gastro-oesophageal reflux (gord) / gastric reflux_T0','NCI_arthritis (nos)_T0','NCI_osteoarthritis_T0','NCI_osteoporosis_T0','NCI_rheumatoid arthritis_T0','NCI_migraine_T0','NCI_headaches (not migraine)_T0','NCI_angina_T0','NCI_carpal tunnel syndrome_T0','NCI_gout_T0','NCI_chronic fatigue syndrome_T0','NCI_ankylosing spondylitis_T0','NCI_trigemminal neuralgia_T0','NCI_crohns disease_T0','NCI_spinal stenosis_T0','NCI_peripheral neuropathy_T0','NCI_ulcerative colitis_T0','NCI_pulmonary embolism +/- dvt_T0','NCI_chronic obstructive airways disease/copd_T0','NCI_stroke_T0','NCI_multiple sclerosis_T0','NCI_psoriatic arthropathy_T0','NCI_parkinsons disease_T0','NCI_peripheral vascular disease_T0']
# Calculate Odds Ratios and Confidence Intervals
df_cols = ['Category','OR','se','pvalue','n','CI lower','CI upper']
results_df = pd.DataFrame(columns=df_cols)
for pain in dx_list:
    df_new = df.dropna(subset=[pain])
    lr_or = pg.logistic_regression(df_new['Sex_T0'],df_new[pain],remove_na=True)
    new = pd.DataFrame()
    new['Category'] = [pain]
    new['OR'] = [np.exp(lr_or['coef'][1])]
    new['se'] = [lr_or['se'][1]]
    new['pvalue'] = [lr_or['pval'][1]]
    new['n'] = [df_new[df_new[pain]==1].shape[0]]
    new['CI lower'] = [np.exp(lr_or['CI[2.5%]'][1])]
    new['CI upper'] = [np.exp(lr_or['CI[97.5%]'][1])]
    results_df = pd.concat([results_df,new])

print(results_df)
#results_df.to_csv('/PATH/UKB_NCI_dx_sexdiff.csv',index=False)

#Nociplastic Longitudinal
dx = pd.read_csv('/PATH/Longitudinal_Nociplastic_cols.csv')
df = df.merge(dx,on='eid')
df_cols = ['Category','OR','se','pvalue','n','CI lower','CI upper']
results_df = pd.DataFrame(columns=df_cols)
dx_list = ['Fibromyalgia','CFS','IBS','Migraine','Headache']
for pain in dx_list:
    df_new = df.dropna(subset=[pain])
    lr_or = pg.logistic_regression(df_new['Sex_T0'],df_new[pain],remove_na=True)
    new = pd.DataFrame()
    new['Category'] = [pain]
    new['OR'] = [np.exp(lr_or['coef'][1])]
    new['se'] = [lr_or['se'][1]]
    new['pvalue'] = [lr_or['pval'][1]]
    new['n'] = [df_new[df_new[pain]==1].shape[0]]
    new['CI lower'] = [np.exp(lr_or['CI[2.5%]'][1])]
    new['CI upper'] = [np.exp(lr_or['CI[97.5%]'][1])]
    results_df = pd.concat([results_df,new])

#save results to csv
results_df.to_csv('/PATH/UKB_Nociplastic_Long_dx_sexdiff.csv',index=False)