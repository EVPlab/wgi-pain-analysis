import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pingouin as pg

#--------------------------------#
#Code for Calculating Gender Index
# Used in Figure 1
#--------------------------------#

#Read Training Data
homepath = '/PATH/'
df = pd.read_csv(homepath+'gender_variables_median_impute_no_mri_I.csv')

#Feature names by gender component
gender_Identity_features = ['Depressed_Mood', 'Disinterest', 'Restlessness', 'Lethargy', 'MoodSwings', 'Miserableness', 'Irritability', 'Sensitivity', 'FedUp', 'Nervous', 'Worrier', 'Tense', 'Embarrassment', 'Nerves', 'Lonely', 'Guilt', 'Risk_Taking']
gender_Relations_features = ['Family_Visits', 'SM_Child', 'Partner_Abuse', 'Violent_Crime', 'SA_Adult', 'Belittlement', 'SI_Partner', 'Confide','Sport', 'Pub', 'Religion', 'Adult_Ed', 'Other', 'No_Leisure']
gender_Roles_features = ['Employed', 'Retired', 'Looking_After_Home', 'Unable_to_work', 'Unemployed', 'Volunteering', 'Student', 'Job_Category']
gender_Inst_features = ['Rent','None_Above_E', 'University', 'A_Level', 'O_Level', 'CSE', 'NVQ', 'Other_Professional', 'None_Above_Q','IncomeHouseholdPreTax_T0']

#Split data into features and target
X_Identity = df[gender_Identity_features]
X_Relations = df[gender_Relations_features]
X_Roles = df[gender_Roles_features]
X_Inst = df[gender_Inst_features]
y_train = df.Sex

# Define models and pipeline
column_trans = make_column_transformer((OneHotEncoder(), ['Job_Category']),remainder='passthrough')
model_identity = LogisticRegression(penalty = 'l2', C=0.5, max_iter=1000, solver='saga')
model_relations = LogisticRegression(penalty = 'l2', C=0.5, max_iter=1000, solver='saga')
model_roles = LogisticRegression(penalty = 'l2', C=0.5, max_iter=1000, solver='saga')
model_inst = LogisticRegression(penalty = 'l2', C=0.5, max_iter=1000, solver='saga')
pipe_roles = Pipeline(steps=[('colum_trans',column_trans),  ('model',model_roles)])

#Define cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=2023)

# Predict probabilities using cross-validation for each model
prob_identity = cross_val_predict(model_identity, X_Identity, y_train, cv=kf, method='predict_proba')
prob_relations = cross_val_predict(model_relations, X_Relations, y_train, cv=kf, method='predict_proba')
prob_roles = cross_val_predict(pipe_roles, X_Roles, y_train, cv=kf, method='predict_proba')
prob_inst = cross_val_predict(model_inst, X_Inst, y_train, cv=kf, method='predict_proba')

# Combine probabilities for meta-model training
X_meta = np.hstack([prob_identity[:,1:], prob_relations[:,1:], prob_roles[:,1:], prob_inst[:,1:]])

# Meta-model
meta_model = LogisticRegression(penalty='l2', C=0.5, max_iter=1000, solver='saga')

# Perform cross-validation for meta-model
meta_model_proba = cross_val_predict(meta_model, X_meta, y_train, cv=kf, method='predict_proba')

# Plotting AUC
def plot_roc_curve(y_true, y_probas, model_names, ax, title="ROC Curves"):
    for y_proba, name in zip(y_probas, model_names):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")
    
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)

# Probabilities for ROC curve (taking the probability of the positive class)
probas_train = [
    prob_identity[:, 1],
    prob_relations[:, 1],
    prob_roles[:, 1],
    prob_inst[:, 1],
    meta_model_proba[:, 1]
]
model_names = ["Identity", "Relations", "Roles", "Institutional", "Meta-Model"]

# Create DataFrame with probabilities and participant info
prob_df = pd.DataFrame({
    'eid': df['eid'],
    'Sex': y_train,
    'Identity_Prob': prob_identity[:, 1],
    'Relations_Prob': prob_relations[:, 1],
    'Roles_Prob': prob_roles[:, 1],
    'Institutional_Prob': prob_inst[:, 1],
    'Meta_Model_Prob': meta_model_proba[:, 1]
})

# Dimension Importance
# Function to calculate and return AUC-ROC loss
def calculate_loss(X_full, y, excluded_index):
    # Exclude one set of probabilities
    X_subset = np.delete(X_full, excluded_index, axis=1)
    # Re-train and evaluate the meta-model
    meta_model = LogisticRegression(penalty='l2', C=0.5, max_iter=1000, solver='saga')
    scores = cross_val_score(meta_model, X_subset, y, cv=kf, scoring='roc_auc')
    return np.mean(scores)

# AUC-ROC for the full meta-model
full_meta = np.mean(cross_val_score(meta_model, X_meta, y_train, cv=kf, scoring='roc_auc'))

# Initialize a dictionary to store AUC loss
auc_loss = {}

# Calculate loss in AUC-ROC for each model
for i, name in enumerate(model_names[:-1]):  # Exclude the last one, which is the meta-model
    auc_with_exclusion = calculate_loss(X_meta, y_train, i)
    auc_loss[name] = full_meta - auc_with_exclusion

# Display the AUC loss for each sub-model
for model, loss in auc_loss.items():
    print(f"AUC Loss without '{model}': {loss:.4f}")

########Test
df_test = pd.read_csv(homepath+'gender_variables_median_impute_mri_I.csv')
# Assuming df_test is your test DataFrame with the same structure as df
X_test_identity = df_test[gender_Identity_features]
X_test_relations = df_test[gender_Relations_features]
X_test_roles = df_test[gender_Roles_features]
X_test_inst = df_test[gender_Inst_features]
y_test = df_test['Sex']
#Fit dimension models to train set
model_identity.fit(X_Identity,y_train)
model_relations.fit(X_Relations,y_train)
pipe_roles.fit(X_Roles,y_train)
model_inst.fit(X_Inst,y_train)
meta_model.fit(X_meta, y_train)

#get structured coefficients for each model
coef_identity = pd.DataFrame(model_identity.coef_,columns=gender_Identity_features)
coef_relations = pd.DataFrame(model_relations.coef_,columns=gender_Relations_features)
# Get feature names after one-hot encoding
encoded_feature_names = pipe_roles.named_steps['colum_trans'].get_feature_names_out()
coef_roles = pd.DataFrame(pipe_roles.named_steps['model'].coef_, columns=encoded_feature_names)
coef_inst = pd.DataFrame(model_inst.coef_,columns=gender_Inst_features)
coef_meta = pd.DataFrame(meta_model.coef_,columns=["Identity", "Relations", "Roles", "Institutional"])
coef_meta.to_csv(homepath+'gender_coefs_I.csv',index=False)
coef_identity.to_csv(homepath+'gender_coefs_identity_I.csv',index=False)
coef_relations.to_csv(homepath+'gender_coefs_relations_I.csv',index=False)
coef_roles.to_csv(homepath+'gender_coefs_roles_I.csv',index=False)
coef_inst.to_csv(homepath+'gender_coefs_institutional_I.csv',index=False)

# Generate predictions for the test set
prob_test_identity = model_identity.predict_proba(X_test_identity)[:, 1]
prob_test_relations = model_relations.predict_proba(X_test_relations)[:, 1]
prob_test_roles = pipe_roles.predict_proba(X_test_roles)[:, 1]
prob_test_inst = model_inst.predict_proba(X_test_inst)[:, 1]

#get structured coefficients for each model
# Calculate correlations between features and model probabilities
# Identity correlations
identity_corrs = []
for col in X_test_identity.columns:
    r = pg.corr(X_test_identity[col], prob_test_identity)['r'].iloc[0]
    identity_corrs.append((col, r))
identity_corrs = sorted(identity_corrs, key=lambda x: abs(x[1]), reverse=True)
coefs_identity = pd.DataFrame(identity_corrs,columns=['Feature','Correlation'])
coefs_identity.to_csv(homepath+'gender_coefs_identity_corrs_I.csv',index=False)

# Relations correlations 
relations_corrs = []
for col in X_test_relations.columns:
    r = pg.corr(X_test_relations[col], prob_test_relations)['r'].iloc[0]
    relations_corrs.append((col, r))
relations_corrs = sorted(relations_corrs, key=lambda x: abs(x[1]), reverse=True)
coefs_relations = pd.DataFrame(relations_corrs,columns=['Feature','Correlation'])
coefs_relations.to_csv(homepath+'gender_coefs_relations_corrs_I.csv',index=False)

# Roles correlations
roles_corrs = []
# Get feature names after one-hot encoding
encoded_feature_names = pipe_roles.named_steps['colum_trans'].get_feature_names_out()
# Transform test data using the fitted pipeline's transformer
X_test_roles_encoded = pipe_roles.named_steps['colum_trans'].transform(X_test_roles)
# Convert to DataFrame with proper column names
X_test_roles_encoded = pd.DataFrame(X_test_roles_encoded, columns=encoded_feature_names)

for col in encoded_feature_names:
    r = pg.corr(X_test_roles_encoded[col], prob_test_roles)['r'].iloc[0]
    roles_corrs.append((col, r))
roles_corrs = sorted(roles_corrs, key=lambda x: abs(x[1]), reverse=True)
coefs_roles = pd.DataFrame(roles_corrs,columns=['Feature','Correlation'])
coefs_roles.to_csv(homepath+'gender_coefs_roles_corrs_I.csv',index=False)

# Institutional correlations
inst_corrs = []
for col in X_test_inst.columns:
    r = pg.corr(X_test_inst[col], prob_test_inst)['r'].iloc[0]
    inst_corrs.append((col, r))
inst_corrs = sorted(inst_corrs, key=lambda x: abs(x[1]), reverse=True)
coefs_inst = pd.DataFrame(inst_corrs,columns=['Feature','Correlation'])
coefs_inst.to_csv(homepath+'gender_coefs_institutional_corrs_I.csv',index=False)


# Combine test set probabilities for the meta-model
X_test_meta = np.vstack([prob_test_identity, prob_test_relations, prob_test_roles, prob_test_inst]).T

# Predict with the meta-model
prob_test_meta = meta_model.predict_proba(X_test_meta)[:, 1]

# Calculate AUC scores and plot ROC curves
auc_scores_test = [roc_auc_score(y_test, prob) for prob in [prob_test_identity, prob_test_relations, prob_test_roles, prob_test_inst, prob_test_meta]]
probas_test = [prob_test_identity, prob_test_relations, prob_test_roles, prob_test_inst, prob_test_meta]

# Combine training and test set probabilities
combined_prob_df = pd.concat([
    prob_df, 
    pd.DataFrame({
        'eid': df_test['eid'],
        'Sex': y_test,
        'Identity_Prob': prob_test_identity,
        'Relations_Prob': prob_test_relations,
        'Roles_Prob': prob_test_roles,
        'Institutional_Prob': prob_test_inst,
        'Meta_Model_Prob': prob_test_meta
    })
])

# Save combined DataFrame to CSV
combined_prob_df.to_csv(homepath+'new_GI_probabilities_I.csv', index=False)