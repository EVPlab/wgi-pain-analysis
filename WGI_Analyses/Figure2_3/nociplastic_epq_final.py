import pandas as pd
import numpy as np
import pingouin as pg

#--------------------------------#
#Code for Determining Sex Differences and WGI effects for Symptoms
# and association between diangoses and symptoms
# Used in Figure 3E and Figure 3F
#--------------------------------#

#Read in Data
homepath = "/PATH/"
df = pd.read_csv(homepath + "nociplastic_EPQ.csv")
dx=pd.read_csv(homepath + "UKB_500K_pain_updated.csv")
dx_nci=pd.read_csv(homepath + "UKB_Fig2/NCI_combined_new.csv")
gi=pd.read_csv(homepath + "new_GI_probabilities_I.csv")
df=pd.merge(df,dx,on='eid',how='left')
df=df.merge(dx_nci,on='eid',how='left')
df=df.merge(gi,on='eid',how='left')

#Create PSD variable with levels of severity
df['PSD'] = df.apply(lambda row: 0 if row['FSC'] < 4 else (1 if row['FSC'] < 8 else (2 if row['FSC'] < 12 else (3 if row['FSC'] < 20 else 4))), axis=1)

#Association between PSD levels/DN4 levels and sex differences
df_cols = ['Category','OR','Prop','se','pvalue','n']
or_df = pd.DataFrame(columns=df_cols)
for i in range(1,5):
    df['new'] = df['PSD'].apply(lambda x: 0 if x == 0 else (1 if x == i else np.nan))
    lr_or = pg.logistic_regression(df['new'],df['Sex_T0'],remove_na=True)
    new = pd.DataFrame()
    new['Category'] = [str(i)]
    new['OR'] = [np.exp(lr_or['coef'][1])]
    new['Prop'] = df[df['new'] == 1]['Sex_T0'].value_counts()[1]/df[df['new'] == 1].shape[0]
    new['se'] = [lr_or['se'][1]]
    new['pvalue'] = [lr_or['pval'][1]]
    new['n'] = [df.shape[0]]
    or_df = pd.concat([or_df,new])

df['new'] = df['DN4'].apply(lambda x: 1 if x >= 4 else 0)
lr_or = pg.logistic_regression(df['new'],df['Sex_T0'],remove_na=True)
new = pd.DataFrame()
new['Category'] = [str(i)]
new['OR'] = [np.exp(lr_or['coef'][1])]
new['Prop'] = df[df['new'] == 1]['Sex_T0'].value_counts()[1]/df[df['new'] == 1].shape[0]
new['se'] = [lr_or['se'][1]]
new['pvalue'] = [lr_or['pval'][1]]
new['n'] = [df.shape[0]]
or_df = pd.concat([or_df,new])

or_df.to_csv(homepath + "Sex_PSD_DN4_OR.csv", index=False)

#Association between PSD levels/DN4 levels and WGI

f = df[df['Sex_T0']==0]
m = df[df['Sex_T0']==1]


df_cols = ['Category','OR','Prop','se','pvalue','n']
or_df = pd.DataFrame(columns=df_cols)
for i in range(1,5):
    f['new'] = f['PSD'].apply(lambda x: 0 if x == 0 else (1 if x == i else np.nan))
    m['new'] = m['PSD'].apply(lambda x: 0 if x == 0 else (1 if x == i else np.nan))
    lr_or_f = pg.logistic_regression(f['new'],f['Meta_Model_Prob'],remove_na=True)
    lr_or_m = pg.logistic_regression(m['new'],m['Meta_Model_Prob'],remove_na=True)
    new = pd.DataFrame()
    new['Category'] = [str(i),str(i)]
    new['Score'] = ['Women','Men']
    new['OR'] = [np.exp(lr_or_f['coef'][1]),np.exp(lr_or_m['coef'][1])]
    new['Prop'] = [f[f['new'] == 1]['new'].value_counts()[1]/f[f['new'] == 1].shape[0],m[m['new'] == 1]['new'].value_counts()[1]/m[m['new'] == 1].shape[0]]
    new['se'] = [lr_or_f['se'][1],lr_or_m['se'][1]]
    new['pvalue'] = [lr_or_f['pval'][1],lr_or_m['pval'][1]]
    new['n'] = [f[f['new'] == 1].shape[0],m[m['new'] == 1].shape[0]]
    or_df = pd.concat([or_df,new])

f['new'] = f['DN4'].apply(lambda x: 1 if x >= 4 else 0)
m['new'] = m['DN4'].apply(lambda x: 1 if x >= 4 else 0)
lr_or_f = pg.logistic_regression(f['new'],f['Meta_Model_Prob'],remove_na=True)
lr_or_m = pg.logistic_regression(m['new'],m['Meta_Model_Prob'],remove_na=True)
new = pd.DataFrame()
new['Category'] = [str(i),str(i)]
new['Score'] = ['Women','Men']
new['OR'] = [np.exp(lr_or_f['coef'][1]),np.exp(lr_or_m['coef'][1])]
new['Prop'] = [f[f['new'] == 1]['new'].value_counts()[1]/f[f['new'] == 1].shape[0],m[m['new'] == 1]['new'].value_counts()[1]/m[m['new'] == 1].shape[0]]
new['se'] = [lr_or_f['se'][1],lr_or_m['se'][1]]
new['pvalue'] = [lr_or_f['pval'][1],lr_or_m['pval'][1]]
new['n'] = [f[f['new'] == 1].shape[0],m[m['new'] == 1].shape[0]]
or_df = pd.concat([or_df,new])

or_df.to_csv(homepath + "Sex_PSD_DN4_WGI_OR.csv", index=False)

#list of diangoses
dx_list_T0 = ['NCI_fibromyalgia_T0','NCI_polymyalgia rheumatica_T0','NCI_cervical spondylosis_T0','NCI_spine arthritis/spondylitis_T0','NCI_joint pain_T0','NCI_back pain_T0','NCI_disc disorder_T0','NCI_trapped nerve/compressed nerve_T0','NCI_sciatica_T0','NCI_hernia_T0','NCI_irritable bowel syndrome_T0','NCI_gastro-oesophageal reflux (gord) / gastric reflux_T0','NCI_arthritis (nos)_T0','NCI_osteoarthritis_T0','NCI_osteoporosis_T0','NCI_rheumatoid arthritis_T0','NCI_migraine_T0','NCI_headaches (not migraine)_T0','NCI_angina_T0','NCI_carpal tunnel syndrome_T0','NCI_gout_T0','NCI_chronic fatigue syndrome_T0','NCI_ankylosing spondylitis_T0','NCI_trigemminal neuralgia_T0','NCI_crohns disease_T0','NCI_spinal stenosis_T0','NCI_peripheral neuropathy_T0','NCI_ulcerative colitis_T0','NCI_pulmonary embolism +/- dvt_T0','NCI_chronic obstructive airways disease/copd_T0','NCI_stroke_T0','NCI_multiple sclerosis_T0','NCI_psoriatic arthropathy_T0','NCI_parkinsons disease_T0','NCI_peripheral vascular disease_T0','Nociplastic_T0','Non-Nociplastic_T0']

#Association between diangoses and FSC
df_cols = ['Category','OR','se','pvalue','n']
or_df = pd.DataFrame(columns=df_cols)
df['new'] = df['FSC'].apply(lambda x: 1 if x >= 8 else 0)
for pain in dx_list_T0:
    lr_or = pg.logistic_regression(df[pain],df['new'],remove_na=True)
    new = pd.DataFrame()
    new['Category'] = [pain]
    new['OR'] = [np.exp(lr_or['coef'][1])]
    new['se'] = [lr_or['se'][1]]
    new['pvalue'] = [lr_or['pval'][1]]
    new['n'] = [df.shape[0]]
    or_df = pd.concat([or_df,new])

or_df.to_csv(homepath + "DX_PSDmod_OR.csv", index=False)

#Association between diangoses and DN4
df_cols = ['Category','OR','se','pvalue','n']
or_df = pd.DataFrame(columns=df_cols)
df['new'] = df['DN4'].apply(lambda x: 1 if x >= 4 else 0)
for pain in dx_list_T0:
    lr_or = pg.logistic_regression(df[pain],df['new'],remove_na=True)
    new = pd.DataFrame()
    new['Category'] = [pain]
    new['OR'] = [np.exp(lr_or['coef'][1])]
    new['se'] = [lr_or['se'][1]]
    new['pvalue'] = [lr_or['pval'][1]]
    new['n'] = [df.shape[0]]
    or_df = pd.concat([or_df,new])

or_df.to_csv(homepath + "DX_DN4sig_OR.csv", index=False)



