import pandas as pd
import pingouin as pg
from sklearn.preprocessing import scale
from confounds import Residualize
import numpy as np

#--------------------------------#
#Calculating Odds Ratios for Pain Sites and Diagnoses
# Used in Figure 2 and 3
#--------------------------------#

#Read in data
gi=pd.read_csv('/PATH/new_GI_probabilities_I.csv')
dx=pd.read_csv('/PATH/UKB_500K_pain_updated.csv')
dx_nci=pd.read_csv('/PATH/UKB_Fig2/NCI_combined_new.csv')
df=pd.merge(gi,dx,on='eid',how='left')
df=df.merge(dx_nci,on='eid',how='left')

#Split data into female and male
female = df[df['Sex_T0']==0]
male = df[df['Sex_T0']==1]

#Residualize Gender Index by Age
resid = Residualize()
resid.fit(female[['Meta_Model_Prob']], female[['Age_T0']])
female['Meta_Model_Prob'] = resid.transform(female[['Meta_Model_Prob']], female[['Age_T0']])
resid.fit(male[['Meta_Model_Prob']], male[['Age_T0']])
male['Meta_Model_Prob'] = resid.transform(male[['Meta_Model_Prob']], male[['Age_T0']])

# Scale Meta_Model_Prob for each sex
female['Meta_Model_Prob_scaled'] = scale(female['Meta_Model_Prob'])
male['Meta_Model_Prob_scaled'] = scale(male['Meta_Model_Prob'])

# List of pain sites
pain_list = ['ChronicWidespreadPain_T0','ChronicNeckShoulderPain_T0', 'ChronicHipPain_T0', 'ChronicBackPain_T0','ChronicStomachAbdominalPain_T0', 'ChronicKneePain_T0','ChronicHeadaches_T0', 'ChronicFacialPain_T0','1 site','2 sites','3 sites','4+ sites']

#Loop through pain sites and calculate ORs for T0
df_cols = ['Site','Score','OR','se','pvalue','n']
or_df = pd.DataFrame(columns=df_cols)
for pain in pain_list:
    print(pain)
    g_f = pg.logistic_regression(female['Meta_Model_Prob_scaled'],female[pain],remove_na=True)
    g_m = pg.logistic_regression(male['Meta_Model_Prob_scaled'],male[pain],remove_na=True)
    site_list = [pain,pain]
    sex_list = ['Women','Men']
    or_list = [g_f['coef'][1],g_m['coef'][1]]
    se_list = [g_f['se'][1],g_m['se'][1]]
    p_list = [g_f['pval'][1],g_m['pval'][1]]
    n_list = [female.dropna(subset=[pain]).shape[0],male.dropna(subset=[pain]).shape[0]]
    new = pd.DataFrame()
    new['Site'] = site_list
    new['Score'] = sex_list
    new['OR'] = or_list
    new['se'] = se_list
    new['pvalue'] = p_list
    new['n'] = n_list
    or_df = pd.concat([or_df,new])

or_df.to_csv('/PATH/gi_ors_UKB_I_Resid.csv',index=False)

# Diagnoses T0 and T2
dx_list_T0 = ['NCI_fibromyalgia_T0','NCI_polymyalgia rheumatica_T0','NCI_cervical spondylosis_T0','NCI_spine arthritis/spondylitis_T0','NCI_joint pain_T0','NCI_back pain_T0','NCI_disc disorder_T0','NCI_trapped nerve/compressed nerve_T0','NCI_sciatica_T0','NCI_hernia_T0','NCI_irritable bowel syndrome_T0','NCI_gastro-oesophageal reflux (gord) / gastric reflux_T0','NCI_arthritis (nos)_T0','NCI_osteoarthritis_T0','NCI_osteoporosis_T0','NCI_rheumatoid arthritis_T0','NCI_migraine_T0','NCI_headaches (not migraine)_T0','NCI_angina_T0','NCI_carpal tunnel syndrome_T0','NCI_gout_T0','NCI_chronic fatigue syndrome_T0','NCI_ankylosing spondylitis_T0','NCI_trigemminal neuralgia_T0','NCI_crohns disease_T0','NCI_spinal stenosis_T0','NCI_peripheral neuropathy_T0','NCI_ulcerative colitis_T0','NCI_pulmonary embolism +/- dvt_T0','NCI_chronic obstructive airways disease/copd_T0','NCI_stroke_T0','NCI_multiple sclerosis_T0','NCI_psoriatic arthropathy_T0','NCI_parkinsons disease_T0','NCI_peripheral vascular disease_T0','Nociplastic_T0','Non-Nociplastic_T0']
dx_list_T2 = ['NCI_fibromyalgia_T2','NCI_polymyalgia rheumatica_T2','NCI_cervical spondylosis_T2','NCI_spine arthritis/spondylitis_T2','NCI_joint pain_T2','NCI_back pain_T2','NCI_disc disorder_T2','NCI_trapped nerve/compressed nerve_T2','NCI_sciatica_T2','NCI_hernia_T2','NCI_irritable bowel syndrome_T2','NCI_gastro-oesophageal reflux (gord) / gastric reflux_T2','NCI_arthritis (nos)_T2','NCI_osteoarthritis_T2','NCI_osteoporosis_T2','NCI_rheumatoid arthritis_T2','NCI_migraine_T2','NCI_headaches (not migraine)_T2','NCI_angina_T2','NCI_carpal tunnel syndrome_T2','NCI_gout_T2','NCI_chronic fatigue syndrome_T2','NCI_ankylosing spondylitis_T2','NCI_trigemminal neuralgia_T2','NCI_crohns disease_T2','NCI_spinal stenosis_T2','NCI_peripheral neuropathy_T2','NCI_ulcerative colitis_T2','NCI_pulmonary embolism +/- dvt_T2','NCI_chronic obstructive airways disease/copd_T2','NCI_stroke_T2','NCI_multiple sclerosis_T2','NCI_psoriatic arthropathy_T2','NCI_parkinsons disease_T2','NCI_peripheral vascular disease_T2','Nociplastic_T2','Non-Nociplastic_T2']

#Loop through diagnoses and calculate OR associations with Gender Index T0
df_cols = ['Site','Score','OR','se','pvalue','n']
or_df = pd.DataFrame(columns=df_cols)
for pain in dx_list_T0:
    print(pain)
    female['new'] = female.apply(lambda row: 1 if row[pain]==1 else (0 if row['NCI_T0']==0 else np.nan),axis=1)
    male['new'] = male.apply(lambda row: 1 if row[pain]==1 else (0 if row['NCI_T0']==0 else np.nan),axis=1)
    g_f = pg.logistic_regression(female['Meta_Model_Prob_scaled'],female['new'],remove_na=True)
    g_m = pg.logistic_regression(male['Meta_Model_Prob_scaled'],male['new'],remove_na=True)
    site_list = [pain,pain]
    sex_list = ['Women','Men']
    or_list = [g_f['coef'][1],g_m['coef'][1]]
    se_list = [g_f['se'][1],g_m['se'][1]]
    p_list = [g_f['pval'][1],g_m['pval'][1]]
    n_list = [female.dropna(subset=[pain]).shape[0],male.dropna(subset=[pain]).shape[0]]
    new = pd.DataFrame()
    new['Site'] = site_list
    new['Score'] = sex_list
    new['OR'] = or_list
    new['se'] = se_list
    new['pvalue'] = p_list
    new['n'] = n_list
    or_df = pd.concat([or_df,new])

or_df.to_csv('/PATH/gi_ors_UKB_dx_new_Resid.csv',index=False)

# Loop through pain sites and calculate ORs for T2
pain_list_t2 = ['ChronicWidespreadPain_T2', 'ChronicNeckShoulderPain_T2','ChronicHipPain_T2', 'ChronicBackPain_T2','ChronicStomachAbdominalPain_T2', 'ChronicKneePain_T2','ChronicHeadaches_T2', 'ChronicFacialPain_T2','1 site T2', '2 sites T2', '3 sites T2', '4+ sites T2']
female = female[female['Present_at_T2']==1]
male = male[male['Present_at_T2']==1]

# Re-scale Meta_Model_Prob for T2 subset
female['Meta_Model_Prob_scaled'] = scale(female['Meta_Model_Prob'])
male['Meta_Model_Prob_scaled'] = scale(male['Meta_Model_Prob'])

df_cols = ['Site','Score','OR','se','pvalue','n']
or_df = pd.DataFrame(columns=df_cols)
for pain, baseline in zip(pain_list_t2, pain_list):
    print(pain,baseline)
    female_new = female[female[baseline]==0]
    male_new = male[male[baseline]==0]
    g_f = pg.logistic_regression(female_new['Meta_Model_Prob_scaled'],female_new[pain],remove_na=True)
    g_m = pg.logistic_regression(male_new['Meta_Model_Prob_scaled'],male_new[pain],remove_na=True)
    site_list = [pain,pain]
    sex_list = ['Women','Men']
    or_list = [g_f['coef'][1],g_m['coef'][1]]
    se_list = [g_f['se'][1],g_m['se'][1]]
    p_list = [g_f['pval'][1],g_m['pval'][1]]
    n_list = [female_new.dropna(subset=[pain]).shape[0],male_new.dropna(subset=[pain]).shape[0]]
    new = pd.DataFrame()
    new['Site'] = site_list
    new['Score'] = sex_list
    new['OR'] = or_list
    new['se'] = se_list
    new['pvalue'] = p_list
    new['n'] = n_list
    or_df = pd.concat([or_df,new])

or_df.to_csv('/PATH/gi_ors_UKB_I_t2_Resid.csv',index=False)

#Loop through diagnoses and calculate OR associations with Gender Index T2
df_cols = ['Site','Score','OR','se','pvalue','n']
or_df = pd.DataFrame(columns=df_cols)
for pain, baseline in zip(dx_list_T2, dx_list_T0):
    print(pain,baseline)
    female_new = female[female[baseline]==0]
    male_new = male[male[baseline]==0]
    g_f = pg.logistic_regression(female_new['Meta_Model_Prob_scaled'],female_new[pain],remove_na=True)
    g_m = pg.logistic_regression(male_new['Meta_Model_Prob_scaled'],male_new[pain],remove_na=True)
    site_list = [pain,pain]
    sex_list = ['Women','Men']
    or_list = [g_f['coef'][1],g_m['coef'][1]]
    se_list = [g_f['se'][1],g_m['se'][1]]
    p_list = [g_f['pval'][1],g_m['pval'][1]]
    n_list = [female_new.dropna(subset=[pain]).shape[0],male_new.dropna(subset=[pain]).shape[0]]
    new = pd.DataFrame()
    new['Site'] = site_list
    new['Score'] = sex_list
    new['OR'] = or_list
    new['se'] = se_list
    new['pvalue'] = p_list
    new['n'] = n_list
    or_df = pd.concat([or_df,new])

or_df.to_csv('/PATH/gi_ors_UKB_dx_T2_Resid.csv',index=False)

#Calculate change in pain sites between T0 and T2
female['Diff_Pain_Sites'] = female['NumberChronicPainTypes_T2'] - female['NumberChronicPainTypes_T0']
male['Diff_Pain_Sites'] = male['NumberChronicPainTypes_T2'] - male['NumberChronicPainTypes_T0']
diff_coding = lambda x: 3 if x > 3 else (-3 if x<-3 else x)
female['Diff_Pain_Sites'] = female['Diff_Pain_Sites'].apply(diff_coding)
male['Diff_Pain_Sites'] = male['Diff_Pain_Sites'].apply(diff_coding)
resid.fit(female[['Meta_Model_Prob']], female[['NumberChronicPainTypes_T0']])
female['Meta_Model_Prob'] = resid.transform(female[['Meta_Model_Prob']], female[['NumberChronicPainTypes_T0']])
resid.fit(male[['Meta_Model_Prob']], male[['NumberChronicPainTypes_T0']])
male['Meta_Model_Prob'] = resid.transform(male[['Meta_Model_Prob']], male[['NumberChronicPainTypes_T0']])

# Re-scale Meta_Model_Prob after residualizing with NumberChronicPainTypes_T0
female['Meta_Model_Prob_scaled'] = scale(female['Meta_Model_Prob'])
male['Meta_Model_Prob_scaled'] = scale(male['Meta_Model_Prob'])

#Calculate ORs for change in pain sites between T0 and T2
df_cols = ['Site','Score','OR','se','pvalue','n']
or_df = pd.DataFrame(columns=df_cols)
for i in range(-3,4):
    if i == 0:
        continue
    coding = lambda x: 1 if x ==i else (0 if x==0 else np.nan)
    female['Change'] = female['Diff_Pain_Sites'].apply(coding)
    male['Change'] = male['Diff_Pain_Sites'].apply(coding)
    g_f = pg.logistic_regression(female['Meta_Model_Prob_scaled'],female['Change'],remove_na=True)
    g_m = pg.logistic_regression(male['Meta_Model_Prob_scaled'],male['Change'],remove_na=True)
    site_list = [i,i]
    sex_list = ['Women','Men']
    or_list = [g_f['coef'][1],g_m['coef'][1]]
    se_list = [g_f['se'][1],g_m['se'][1]]
    p_list = [g_f['pval'][1],g_m['pval'][1]]
    n_list = [female.dropna(subset=['Change']).shape[0],male.dropna(subset=['Change']).shape[0]]
    new = pd.DataFrame()
    new['Site'] = site_list
    new['Score'] = sex_list
    new['OR'] = or_list
    new['se'] = se_list
    new['pvalue'] = p_list
    new['n'] = n_list
    or_df = pd.concat([or_df,new])

or_df.to_csv('/Volumes/Seagate_Por/csvs/gi_ors_UKB_I_spread_t2_new_Resid.csv',index=False)

new = df[df['Present_at_T2']==1]
new['Diff_Pain_Sites'] = new['NumberChronicPainTypes_T2'] - new['NumberChronicPainTypes_T0']

for i in range(1,4):
    if i == 0:
        continue
    coding = lambda x: 1 if x ==i else (0 if x==0 else np.nan)
    title = 'Change_'+str(i)
    new[title] = new['Diff_Pain_Sites'].apply(coding)

for pain, baseline in zip(pain_list_t2, pain_list):
    new[f'new_{pain}'] = new.apply(lambda row: 1 if row[pain] == 1 and row[baseline] == 0 else 0 if row[pain] == 0 and row[baseline] == 0 else np.nan, axis=1)

new = new[['eid','new_ChronicWidespreadPain_T2', 'new_ChronicNeckShoulderPain_T2','new_ChronicHipPain_T2', 'new_ChronicBackPain_T2','new_ChronicStomachAbdominalPain_T2', 'new_ChronicKneePain_T2','new_ChronicHeadaches_T2', 'new_ChronicFacialPain_T2', 'Change_1', 'Change_2', 'Change_3']]
new.to_csv('/PATH/Pain_T2_w_change_sept25.csv',index=False)

#Dimension

#### Dimension Scores
df[['Identity_Prob','Relations_Prob','Roles_Prob','Institutional_Prob']]=(df[['Identity_Prob','Relations_Prob','Roles_Prob','Institutional_Prob']]*-1)+1

female = df[df['Sex_T0']==0]
male = df[df['Sex_T0']==1]

resid.fit(female[['Identity_Prob']], female[['Age_T0']])
female['Identity_Prob_r'] = resid.transform(female[['Identity_Prob']], female[['Age_T0']])
resid.fit(male[['Identity_Prob']], male[['Age_T0']])
male['Identity_Prob_r'] = resid.transform(male[['Identity_Prob']], male[['Age_T0']])
resid.fit(female[['Relations_Prob']], female[['Age_T0']])
female['Relations_Prob_r'] = resid.transform(female[['Relations_Prob']], female[['Age_T0']])
resid.fit(male[['Relations_Prob']], male[['Age_T0']])
male['Relations_Prob_r'] = resid.transform(male[['Relations_Prob']], male[['Age_T0']])
resid.fit(female[['Roles_Prob']], female[['Age_T0']])
female['Roles_Prob_r'] = resid.transform(female[['Roles_Prob']], female[['Age_T0']])
resid.fit(male[['Roles_Prob']], male[['Age_T0']])
male['Roles_Prob_r'] = resid.transform(male[['Roles_Prob']], male[['Age_T0']])
resid.fit(female[['Institutional_Prob']], female[['Age_T0']])
female['Institutional_Prob_r'] = resid.transform(female[['Institutional_Prob']], female[['Age_T0']])
resid.fit(male[['Institutional_Prob']], male[['Age_T0']])
male['Institutional_Prob_r'] = resid.transform(male[['Institutional_Prob']], male[['Age_T0']])
# Scale residualized dimension scores
female['Identity_Prob_r_scaled'] = scale(female['Identity_Prob_r'])
male['Identity_Prob_r_scaled'] = scale(male['Identity_Prob_r'])
female['Relations_Prob_r_scaled'] = scale(female['Relations_Prob_r'])
male['Relations_Prob_r_scaled'] = scale(male['Relations_Prob_r'])
female['Roles_Prob_r_scaled'] = scale(female['Roles_Prob_r'])
male['Roles_Prob_r_scaled'] = scale(male['Roles_Prob_r'])
female['Institutional_Prob_r_scaled'] = scale(female['Institutional_Prob_r'])
male['Institutional_Prob_r_scaled'] = scale(male['Institutional_Prob_r'])

df_cols = ['name','Score','OR','se','pvalue','n']
or_df = pd.DataFrame(columns=df_cols)

for pain in dx_list_T0:
    female['new'] = female.apply(lambda row: 1 if row[pain]==1 else (0 if row['NCI_T0']==0 else np.nan),axis=1)
    male['new'] = male.apply(lambda row: 1 if row[pain]==1 else (0 if row['NCI_T0']==0 else np.nan),axis=1)
    f_1 = pg.logistic_regression(female['Identity_Prob_r_scaled'],female['new'],remove_na=True)
    m_1 = pg.logistic_regression(male['Identity_Prob_r_scaled'],male['new'],remove_na=True)
    f_2 = pg.logistic_regression(female['Relations_Prob_r_scaled'],female['new'],remove_na=True)
    m_2 = pg.logistic_regression(male['Relations_Prob_r_scaled'],male['new'],remove_na=True)
    f_3 = pg.logistic_regression(female['Roles_Prob_r_scaled'],female['new'],remove_na=True)
    m_3 = pg.logistic_regression(male['Roles_Prob_r_scaled'],male['new'],remove_na=True)
    f_4 = pg.logistic_regression(female['Institutional_Prob_r_scaled'],female['new'],remove_na=True)
    m_4 = pg.logistic_regression(male['Institutional_Prob_r_scaled'],male['new'],remove_na=True) 
    site_list = [pain,pain,pain,pain,pain,pain,pain,pain]
    score_list = ['Identity_Prob Women','Identity_Prob Men','Relations_Prob Women','Relations_Prob Men','Roles_Prob Women','Roles_Prob Men','Institutional_Prob Women','Institutional_Prob Men']
    or_list = [f_1['coef'][1],m_1['coef'][1],f_2['coef'][1],m_2['coef'][1],f_3['coef'][1],m_3['coef'][1],f_4['coef'][1],m_4['coef'][1]]
    se_list = [f_1['se'][1],m_1['se'][1],f_2['se'][1],m_2['se'][1],f_3['se'][1],m_3['se'][1],f_4['se'][1],m_4['se'][1]]
    p_list = [f_1['pval'][1],m_1['pval'][1],f_2['pval'][1],m_2['pval'][1],f_3['pval'][1],m_3['pval'][1],f_4['pval'][1],m_4['pval'][1]]
    n_list = [female.dropna(subset=[pain]).shape[0],male.dropna(subset=[pain]).shape[0],female.dropna(subset=[pain]).shape[0],male.dropna(subset=[pain]).shape[0],female.dropna(subset=[pain]).shape[0],male.dropna(subset=[pain]).shape[0],female.dropna(subset=[pain]).shape[0],male.dropna(subset=[pain]).shape[0]]
    new = pd.DataFrame()
    new['name'] = site_list
    new['Score'] = score_list
    new['OR'] = or_list
    new['se'] = se_list
    new['pvalue'] = p_list
    new['n'] = n_list
    or_df = pd.concat([or_df,new])

or_df.to_csv('/PATH/residualized_Dimensions_OR_UKB_dx_new_Resid.csv',index=False)