import pandas as pd
from scipy.stats import norm

#--------------------------------#
#Code for Determining Average ORs across diagnoses of the same "type"
# Used in Figure 3B & 5A
#--------------------------------#

#Read in Data
df = pd.read_csv('/PATH/dn4_eachdx_or_in_disorder_typed.csv')
types = ['Nociplastic','Neuropathic','Nociceptive','Other']

#create empty dataframe to store results
new_df = pd.DataFrame(columns=['name','Score','OR','se','p-val'])
#loop through each pain type
for t in types:
    df_t = df[df['type'] == t]
    log_or = df_t['OR'].mean()
    se = df_t['se'].mean()
    z = log_or / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    new_df = pd.concat([new_df, pd.DataFrame([{'name':t,'Score':'Women','OR':log_or,'se':se,'p-val':p_value}])], ignore_index=True)

#save results to csv
new_df.to_csv('/PATH/DN4_ave_byType.csv',index=False)