# Import necessary libraries
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

# Example DataFrame (You can replace this with your dataset)
# Assuming 'Y' is the binary dependent variable and 'X1', 'X2', 'X3' are independent variables

df = pd.read_csv('/PATH/UKB_NoBrain_500K_selected.csv')
df = df[['eid', 'Age_T0', 'Sex_T0']]
gi=pd.read_csv('/PATH/new_GI_probabilities_I.csv')
gi['Gender_Index'] = (gi['Meta_Model_Prob']*-1)+1
gi=gi[['eid','Identity_Prob', 'Relations_Prob', 'Roles_Prob','Institutional_Prob','Gender_Index']]
hormones = pd.read_csv('/PATH/UKBB_DATA/tando.csv')
hormones = hormones[['eid','Testosterone_T0']]
hormones['Tesosterone_T0'] = scale(hormones['Testosterone_T0'])
df=df.merge(gi,on='eid',how='left')
df=df.merge(hormones,on='eid',how='left')
dx = pd.read_csv('/PATH/UKB_Fig2/NCI_combined_new.csv')
dx = dx[['eid','Nociplastic_T0']]
df = df.merge(dx,on='eid',how='left')
df_m = pd.read_csv('/PATH/ukb45401.csv') 
df_m = df_m[['eid','2724-0.0','2724-1.0','2724-2.0','2724-3.0','3581-0.0','3581-1.0','3581-2.0','3581-3.0','3591-0.0','2834-0.0']]
df_m.columns = ['eid','Menopause_T0','Menopause_T1','Menopause_T2','Menopause_T3','MenopauseAge_T0','MenopauseAge_T1','MenopauseAge_T2','MenopauseAge_T3','Hysterectomy_T0','Oophorectomy_T0']
df_m = df_m[['eid','Menopause_T0']]
df = df.merge(df_m,on='eid',how='left')
f = df[df['Sex_T0']==0]
f = f.dropna(subset=['Gender_Index', 'Age_T0', 'Menopause_T0', 'Testosterone_T0','Nociplastic_T0'])
# Define the dependent and independent variables
X = f[['Meta_Model_Prob', 'Age_T0', 'Menopause_T0', 'Testosterone_T0']]
Y = f[['Nociplastic_T0']]


# Add a constant to the independent variables matrix
X = sm.add_constant(X)

# Perform the logistic regression
logit_model = sm.Logit(Y, X).fit()

# Print the logistic regression summary
print(logit_model.summary())

# 1. Standardized Coefficients: These allow comparison on the same scale
standardized_coefficients = logit_model.params
print("\nStandardized Coefficients (Log Odds):")
print(standardized_coefficients)

# 2. Odds Ratios: Exponentiate the standardized coefficients
standardized_odds_ratios = np.exp(standardized_coefficients)
print("\nStandardized Odds Ratios:")
print(standardized_odds_ratios)

# 3. Marginal Effects: These measure the change in predicted probability for small changes in independent variables
marginal_effects = logit_model.get_margeff().summary_frame()
print("\nMarginal Effects (dy/dx):")
print(marginal_effects)
marginal_effects.to_csv('/PATH/MarginalEffect_Menopause_GI.csv')