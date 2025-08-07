import pandas as pd
from scipy.stats import norm

#--------------------------------#
#Code for Determining Average ORs across diagnoses of the same "type", stratified by sex
# Used in Figure 3B & 5A
#--------------------------------#

df = pd.read_csv('/PATH/gi_ors_UKB_dx_new_sept25_Forest.csv')
f = df[df['Score'] == 'Women']
m = df[df['Score'] == 'Men']
types = ['Nociplastic','Neuropathic','Nociceptive','Other']

#create empty dataframe to store results
new_df = pd.DataFrame(columns=['name','Score','OR','se','p-val'])
#loop through each pain type
for t in types:
    df_t_f = f[f['type'] == t]
    df_t_m = m[m['type'] == t]
    log_or_f = df_t_f['OR'].mean()
    se_f = df_t_f['se'].mean()
    z_f = log_or_f / se_f
    p_value_f = 2 * (1 - norm.cdf(abs(z_f)))
    log_or_m = df_t_m['OR'].mean()
    se_m = df_t_m['se'].mean()
    z_m = log_or_m / se_m
    p_value_m = 2 * (1 - norm.cdf(abs(z_m)))
    new_df = pd.concat([new_df, pd.DataFrame([{'name':t,'Score':'Women','OR':log_or_f,'se':se_f,'p-val':p_value_f}])], ignore_index=True)
    new_df = pd.concat([new_df, pd.DataFrame([{'name':t,'Score':'Men','OR':log_or_m,'se':se_m,'p-val':p_value_m}])], ignore_index=True)

#Save results to csv
new_df.to_csv('/PATH/UKB_GI_mlm_ave_sexed.csv',index=False)