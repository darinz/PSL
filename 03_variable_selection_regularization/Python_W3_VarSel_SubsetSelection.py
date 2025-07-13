# PSL: Variable Subset Selection
# This script demonstrates best subset selection, AIC/BIC model selection, and stepwise selection for linear regression.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lm
from sklearn.metrics import mean_squared_error, r2_score
import itertools

# --- Load Data ---
# The Prostate data can be found at:
# https://hastie.su.domains/ElemStatLearn/data.html
# The data examines the correlation between the level of prostate-specific antigen and a number of clinical measures in men who were about to receive a radical prostatectomy.
# The first 8 columns are predictors; column 9 is the outcome/response.

myData = pd.read_csv("https://hastie.su.domains/ElemStatLearn/datasets/prostate.data", sep='\t')
# Drop the 1st and 10th columns (index and train/test indicator)
myData = myData.drop(columns=[myData.columns[0], myData.columns[10]])

# Separate predictors and response
df_X = myData.iloc[:, :-1]
df_Y = myData.iloc[:, -1]

# --- Best Subset Selection ---
def bestsubset(X, Y):
    """
    For each model size, select the best subset of variables (smallest RSS among all models with the same size).
    Returns a DataFrame with model size, RSS, and selected features.
    """
    RSS_list, numb_features, feature_list = [], [], []
    for m in range(1, len(X.columns) + 1):
        best_RSS = np.inf
        for combo in itertools.combinations(X.columns, m):
            tmp_X = X[list(combo)]
            tmp_model = lm()
            tmp_model.fit(tmp_X, Y)
            tmp_RSS = mean_squared_error(Y, tmp_model.predict(tmp_X)) * len(Y)
            if tmp_RSS < best_RSS:
                best_RSS = tmp_RSS
                best_varset = combo
        RSS_list.append(best_RSS)
        feature_list.append(best_varset)
        numb_features.append(len(best_varset))
    return pd.DataFrame({'msize': numb_features, 'RSS': RSS_list, 'features': feature_list})

# Run best subset selection
rs = bestsubset(df_X, df_Y)

# --- Model Selection: AIC and BIC ---
n = len(df_Y)
Aic = n * np.log(rs.RSS / n) + 2 * rs.msize
Bic = n * np.log(rs.RSS / n) + rs.msize * np.log(n)

# Plot AIC and BIC for best subset selection
fig = plt.figure(figsize=(16, 6))
ax = fig.add_subplot(1, 2, 1)
ax.scatter(rs.msize, Aic, alpha=1, color='blue')
ax.set_xlabel('No. of Features')
ax.set_ylabel('AIC')
ax.set_title('AIC - Best subset selection')

ax = fig.add_subplot(1, 2, 2)
ax.scatter(rs.msize, Bic, alpha=.5, color='red')
ax.set_xlabel('No. of Features')
ax.set_ylabel('BIC')
ax.set_title('BIC - Best subset selection')
plt.show()

# --- Top 3 Models for Each Size ---
def bestsubset_3(X, Y):
    """
    For each model size, return the top 3 models (by RSS).
    Returns a DataFrame with lists of model sizes, RSS, and features.
    """
    RSS_dict, numb_features_dict, feature_list_dict = {}, {}, {}
    for m in range(1, len(X.columns) + 1):
        best_RSS = []
        for combo in itertools.combinations(X.columns, m):
            tmp_X = X[list(combo)]
            tmp_model = lm()
            tmp_model.fit(tmp_X, Y)
            tmp_RSS = mean_squared_error(Y, tmp_model.predict(tmp_X)) * len(Y)
            best_RSS.append((tmp_RSS, combo))
            best_RSS = sorted(best_RSS)[:3]
        RSS_dict[m] = [x[0] for x in best_RSS]
        feature_list_dict[m] = [x[1] for x in best_RSS]
        numb_features_dict[m] = [len(x[1]) for x in best_RSS]
    return pd.DataFrame({'msize': numb_features_dict, 'RSS': RSS_dict, 'features': feature_list_dict})

# --- Stepwise Selection Functions ---
def computeAIC(X, Y, k=2):
    """
    Compute AIC or BIC for a linear regression model with design matrix X and response Y.
    k=2 for AIC, k=log(n) for BIC.
    """
    n = len(Y)
    model = lm()
    model.fit(X, Y)
    RSS = mean_squared_error(Y, model.predict(X)) * len(Y)
    return n * np.log(RSS / n) + k * X.shape[1]

def stepAIC(X, Y, features=None, AIC=True):
    """
    Stepwise subset selection based on AIC or BIC.
    Starts with the variable set 'features' (default: all variables).
    At each step, checks whether dropping or adding a variable improves AIC/BIC.
    """
    if features is None:
        features = X.columns
    AIC_list, action_list, feature_list = [], [], []
    best_AIC = np.inf
    best_action = ''
    n = len(Y)
    k = 2 if AIC else np.log(n)
    current_AIC = computeAIC(X[features], Y, k)
    while current_AIC < best_AIC:
        AIC_list.append(current_AIC)
        feature_list.append(list(features))
        action_list.append(best_action)
        best_AIC = current_AIC
        tmp_AIC_list, tmp_action_list, tmp_feature_list = [], [], []
        # Try dropping each feature
        for p in features:
            tmp_features = features.drop(p)
            tmp_AIC = computeAIC(X[tmp_features], Y, k)
            tmp_AIC_list.append(tmp_AIC)
            tmp_feature_list.append(tmp_features)
            tmp_action_list.append('- ' + p)
        # Try adding each remaining feature
        remaining_features = [p for p in X.columns if p not in features]
        for p in remaining_features:
            tmp_features = list(features) + [p]
            tmp_AIC = computeAIC(X[tmp_features], Y, k)
            tmp_AIC_list.append(tmp_AIC)
            tmp_feature_list.append(tmp_features)
            tmp_action_list.append('+ ' + p)
        best_model = np.array(tmp_AIC_list).argmin()
        current_AIC = tmp_AIC_list[best_model]
        features = tmp_feature_list[best_model]
        best_action = tmp_action_list[best_model]
    return pd.DataFrame({'AIC': AIC_list, 'action': action_list, 'features': feature_list})

# Example usage:
# myout = stepAIC(df_X, df_Y)  # Stepwise AIC
# myout = stepAIC(df_X, df_Y, AIC=False)  # Stepwise BIC

# The script above demonstrates variable subset selection, model selection criteria, and stepwise selection for linear regression using Python. 