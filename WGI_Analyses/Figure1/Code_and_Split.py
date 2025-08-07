import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

#--------------------------------#
#The purpose of this file is to drop NA values, recode the reamaining data for model building, 
#and split the dataframe into non-mri group and mri group
#--------------------------------#

##reading in dataset
df = pd.read_csv('/PATH/GI_Variables_Unrefined_income.csv')
df2 = pd.read_csv('/PATH/40k_id.csv')
df2.columns=['eid']

##Dropping NA values for questionnaire that excludes the most participants (Trauma Questions)
df = df.dropna(subset=['20525-0.0'])

##Coding Array Type Data
#Merging Values into a list
df['Employment_Status'] = df[['6142-0.0','6142-0.1','6142-0.2','6142-0.3','6142-0.4','6142-0.5','6142-0.6']].values.tolist()
df['Qualifications'] = df[['6138-0.0','6138-0.1','6138-0.2','6138-0.3','6138-0.4','6138-0.5']].values.tolist()
df['Leisure'] = df[['6160-0.0','6160-0.1','6160-0.2','6160-0.3','6160-0.4']].values.tolist()

#Creating new columns
one = lambda x: 1 if x.count(1.0) > 0 else 0
two = lambda x: 1 if x.count(2.0) > 0 else 0
three = lambda x: 1 if x.count(3.0) > 0 else 0
four = lambda x:  1 if x.count(4.0) > 0 else 0
five = lambda x: 1 if x.count(5.0) > 0 else 0
six = lambda x: 1 if x.count(6.0) > 0 else 0
seven = lambda x: 1 if x.count(7.0) > 0 else 0
neg_seven = lambda x: 1 if x.count(-7.0) > 0 else 0
#employment
df['Employed'] = df['Employment_Status'].apply(one)
df['Retired'] = df['Employment_Status'].apply(two)
df['Looking_After_Home'] = df['Employment_Status'].apply(three)
df['Unable_to_work'] = df['Employment_Status'].apply(four)
df['Unemployed'] = df['Employment_Status'].apply(five)
df['Volunteering'] = df['Employment_Status'].apply(six)
df['Student'] = df['Employment_Status'].apply(seven)
df['None_Above_E'] = df['Employment_Status'].apply(neg_seven)
#Qualifications
df['University'] = df['Qualifications'].apply(one)
df['A_Level'] = df['Qualifications'].apply(two)
df['O_Level'] = df['Qualifications'].apply(three)
df['CSE'] = df['Qualifications'].apply(four)
df['NVQ'] = df['Qualifications'].apply(five)
df['Other_Professional'] = df['Qualifications'].apply(six)
df['None_Above_Q'] = df['Qualifications'].apply(neg_seven)
#Leisure
df['Sport'] = df['Leisure'].apply(one)
df['Pub'] = df['Leisure'].apply(two)
df['Religion'] = df['Leisure'].apply(three)
df['Adult_Ed'] = df['Leisure'].apply(four)
df['Other'] = df['Leisure'].apply(five)
df['No_Leisure'] = df['Leisure'].apply(neg_seven)
#drop old columns
df = df.drop(columns=['6142-0.0','6142-0.1','6142-0.2','6142-0.3','6142-0.4','6142-0.5','6142-0.6','6138-0.0','6138-0.1','6138-0.2','6138-0.3','6138-0.4','6138-0.5','6160-0.0','6160-0.1','6160-0.2','6160-0.3','6160-0.4','Employment_Status','Qualifications','Leisure'])

#Renaming Columns for consistent column title format
df = df.rename({'Sex_T0': 'Sex', '20490-0.0': 'SM_Child', '20523-0.0': 'Partner_Abuse', \
    '20529-0.0': 'Violent_Crime', '20531-0.0': 'SA_Adult', '20277-0.0': 'Job_Code', '20525-0.0': 'Rent',\
    '20521-0.0': 'Belittlement', '20524-0.0': 'SI_Partner', '2110-0.0': 'Confide', '1031-0.0': 'Family_Visits',\
    '2050-0.0': 'Depressed_Mood','2060-0.0':'Disinterest','2070-0.0':'Restlessness','2080-0.0':'Lethargy',\
    '1920-0.0':'MoodSwings','1930-0.0':'Miserableness','1940-0.0':'Irritability','1950-0.0':'Sensitivity',\
    '1960-0.0':'FedUp','1970-0.0':'Nervous','1980-0.0':'Worrier','1990-0.0':'Tense','2000-0.0':'Embarrassment',\
    '2010-0.0':'Nerves','2020-0.0':'Lonely','2030-0.0':'Guilt','2040-0.0':'Risk_Taking'}, axis=1)

#Creating Lambda functions for recoding data
#ukbb Data-Coding 100349
dc100349 = lambda x: 1 if x == 1 else (0 if x == 0 else np.nan)
df['MoodSwings'] = df.MoodSwings.apply(dc100349)
df['Miserableness'] = df.Miserableness.apply(dc100349)
df['Irritability'] = df.Irritability.apply(dc100349)
df['Sensitivity'] = df.Sensitivity.apply(dc100349)
df['FedUp'] = df.FedUp.apply(dc100349)
df['Nervous'] = df.Nervous.apply(dc100349)
df['Worrier'] = df.Worrier.apply(dc100349)
df['Tense'] = df.Tense.apply(dc100349)
df['Embarrassment'] = df.Embarrassment.apply(dc100349)
df['Nerves'] = df.Nerves.apply(dc100349)
df['Lonely'] = df.Lonely.apply(dc100349)
df['Guilt'] = df.Guilt.apply(dc100349)
df['Risk_Taking'] = df.Risk_Taking.apply(dc100349)	

#Coding for 533
dc533 = lambda x: 1 if x == 2 else (1 if x == 1 else (0 if x==0 else np.nan))
df['Violent_Crime'] = df.Violent_Crime.apply(dc533)
df['SA_Adult'] = df.SA_Adult.apply(dc533)

#Job Coding: jobs were coded in the SOC2000 occupational coding meaning the first digit of each code represents the overarching job category
#We will use this larger job category to define the type of job done by each participant
#those with no job entered will be set to 0
df['Job_Code'] = df.Job_Code.replace(np.NaN,0)
first_str = lambda x: str(x)[0]
df['Job_Category'] = df.Job_Code.apply(first_str)
df = df.drop(['Job_Code'],axis=1)
#Coding for 532/100327/100501: We are maintaining ordinal nature of the variable but replacing "prefer not to answer" with the average score in the column
#if the participant answers "perfer not to answer" on more than two questsion they will be dropped
df['SM_Child'] = df.SM_Child.replace(-818,np.NaN)
df['Partner_Abuse'] = df.Partner_Abuse.replace(-818,np.NaN)
df['SI_Partner'] = df.SI_Partner.replace(-818,np.NaN)
df['Belittlement'] = df.Belittlement.replace(-818,np.NaN)
df['Rent'] = df.Rent.replace(-818,np.NaN)
df['Family_Visits'] = df.Family_Visits.replace([-1,-3],np.NaN)
df['Confide'] = df.Confide.replace([-1,-3],np.NaN)
df['Depressed_Mood'] = df.Depressed_Mood.replace([-1,-3],np.NaN)
df['Disinterest'] = df.Disinterest.replace([-1,-3],np.NaN)
df['Restlessness'] = df.Restlessness.replace([-1,-3],np.NaN)
df['Lethargy'] = df.Lethargy.replace([-1,-3],np.NaN)
df['IncomeHouseholdPreTax_T0'] = df.IncomeHouseholdPreTax_T0.replace([-1,-3],np.NaN)

#dropping participants with more than 20% missing values
df = df.dropna(thresh=(df.shape[1]*0.80), axis=0)

#splitting into mri (follow-up) and no mri groups
df_1_2 = df.merge(df2, on="eid", how="left", indicator=True)
no_mri = df_1_2[df_1_2["_merge"] == "left_only"].drop(columns=["_merge"])
mri = df_1_2[df_1_2["_merge"] == "both"].drop(columns=["_merge"])

#imputing missing values
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
df_median_mri = imp_median.fit_transform(mri)
df_median_mri = pd.DataFrame(data=df_median_mri,columns=df.columns)
df_median_no_mri = imp_median.fit_transform(no_mri)
df_median_no_mri = pd.DataFrame(data=df_median_no_mri,columns=df.columns)

imp_it = IterativeImputer(missing_values=np.nan)
df_iter_mri = imp_it.fit_transform(mri)
df_iter_mri = pd.DataFrame(data=df_iter_mri,columns=df.columns)
df_iter_no_mri = imp_it.fit_transform(no_mri)
df_iter_no_mri = pd.DataFrame(data=df_iter_no_mri,columns=df.columns)

#saving files
df_median_mri.to_csv('/PATH/gender_variables_median_impute_mri_I.csv',index=False)
df_median_no_mri.to_csv('/PATH/gender_variables_median_impute_no_mri_I.csv',index=False)
df_iter_mri.to_csv('/PATH/gender_variables_iterative_impute_mri_I.csv',index=False)
df_iter_no_mri.to_csv('/PATH/gender_variables_iterative_impute_no_mri_I.csv',index=False)
