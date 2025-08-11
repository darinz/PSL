"""
Random Forest for Regression - Python Implementation

This script demonstrates the use of Random Forest for regression using scikit-learn.
It covers data loading, model training, prediction, error analysis, performance plotting,
and feature importance visualization.

Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
random.seed(124)

# Load the housing dataset
url = "https://liangfgithub.github.io/Data/HousingData.csv"
Housing = pd.read_csv(url)
print("Dataset shape:", Housing.shape)
print("\nFirst few rows:")
print(Housing.head())

# Split the housing data into a training set (70%) and a test set (30%)
X = Housing.drop("Y", axis=1)
y = Housing["Y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ============================================================================
# MODEL FITTING
# ============================================================================
"""
To fit the model, we use the RandomForest function from sklearn.

We aim to predict the variable Y using all other columns in the data matrix as potential predictors. 
We set the oob_score to TRUE (which I'll explain later), max_features = 1.0/3, and specify that 
the forest should consist of 400 trees.

max_features, if float, indicates the fraction of features considered at each split. 
The number of variables selected for each split. The recommended value for regression-based 
Random Forest models is 1/3.
"""

rfModel = RandomForestRegressor(n_estimators=400, oob_score=True, max_features=1.0/3)
rfModel.fit(X_train, y_train)

print(f"\nRandom Forest model fitted with {rfModel.n_estimators} trees")
print(f"Out-of-bag score: {rfModel.oob_score_:.4f}")

# ============================================================================
# MAKING PREDICTIONS
# ============================================================================
yhat = rfModel.predict(X_test)
test_mse = np.mean((yhat - y_test) ** 2.0)
print(f"\nTest MSE: {test_mse:.6f}")

# ============================================================================
# UNDERSTANDING TRAINING ERRORS
# ============================================================================
"""
It's worth discussing the training error. Random Forest provides TWO types of training errors.

The first is obtained by calling the prediction function on the training data, usually 
resulting in an error rate substantially lower than the test error. This is not surprising, 
as training errors are generally lower than test errors, especially for complex models.
"""

yhat_train = rfModel.predict(X_train)
train_mse = np.mean((yhat_train - y_train) ** 2.0)
print(f"Training MSE (using training predictions): {train_mse:.6f}")

"""
However, Random Forest also provides a second type of training error—based on so-called 
out-of-bag samples, if setting oob_score to TRUE. This error rate tends to be closer to 
the test error and serves as a more reliable estimator of the model's generalization performance.
"""

yhat_oob = rfModel.oob_prediction_
oob_mse = np.mean((yhat_oob - y_train) ** 2.0)
print(f"Training MSE (using OOB predictions): {oob_mse:.6f}")

# ============================================================================
# OUT-OF-BAG SAMPLES EXPLANATION
# ============================================================================
"""
What exactly are "Out-of-Bag" samples, and what is the meaning of 'bag' in this specific context?

The term "bag" simply refers to a collection or set where the bootstrap samples can be stored. 
Imagine your original training dataset as being stored in a box. A bootstrap sample is then 
formed by drawing, with replacement, from this box. These selected samples are placed into 
a metaphorical "bag."

In the Random Forest procedure, each tree is built using one such bootstrap sample. Importantly, 
this bag is unlikely to contain every unique instance from the original training set. The data 
points not included in a specific bag are termed "Out-of-Bag" (OOB) samples for that tree. 
These samples serve as test points for evaluating that tree's performance.

Here's how it works: Let's say your Random Forest consists of five trees. Assume that your 
first training instance, X_1, appears only in the bootstrap samples for the first three trees. 
When making predictions for X_1, only the last two trees will be used because X_1 is an OOB 
sample for these trees. Likewise, each data point will have its prediction made from an ensemble 
of trees for which it is an Out-of-Bag sample.

This prediction methodology makes the OOB error rate akin to a cross-validation error, 
differing fundamentally from traditional training error.

In R's Random Forest package, you'll find a useful component named 'oob.times'. This feature 
provides valuable insights by indicating the number of times each training instance was used 
as an Out-Of-Bag (OOB) sample. This can be helpful for understanding the generalization 
performance of the model on different subsets of your data.

However, scikit-learn's RandomForest implementation does not directly offer this 'oob.times' 
feature. While scikit-learn does allow you to calculate OOB error and OOB predictions 
(by setting oob_score=True or using the oob_prediction_ attribute, respectively) but doesn't 
provide the specific counts of how many times each training instance was used as an OOB sample.

Since the 'oob.times' information is not readily available in scikit-learn, I recommend 
students reviewing the corresponding section in our Rcode page, which could offer valuable insights.
"""

# ============================================================================
# PERFORMANCE PLOTS
# ============================================================================
"""
Next, let's visually examine the performance of our Random Forest model.

The performance plot features the number of trees on the x-axis and their corresponding 
mean square error (MSE) on the y-axis; each MSE value on the y-axis is calculated based 
on a forest with the specific number of trees marked on the x-axis. These MSE values are 
derived from OOB predictions, making them a reliable estimate more akin to cross-validation 
error than to simple training error.

The plots reveal that the model's performance improves rapidly with the initial addition 
of trees but starts to plateau beyond a certain point. This suggests that adding more trees 
does not lead to overfitting, although one could potentially achieve similar performance 
with fewer trees, say, just the first 100.
"""

# Reference: https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
min_estimators = 20
max_estimators = 400

error_rate = []
clf = RandomForestRegressor(oob_score=True)
for i in range(min_estimators, max_estimators + 1):
    clf.set_params(n_estimators=i, warm_start=True)
    clf.fit(X_train, y_train)

    # Record the OOB error for each `n_estimators=i` setting.
    oob_error = np.mean((clf.oob_prediction_ - y_train) ** 2.0)
    error_rate.append((i, oob_error))

error_rate = np.array([*error_rate])

# Create the performance plot
plt.figure(figsize=(10, 6))
plt.plot(error_rate[:, 0], error_rate[:, 1])
plt.xlabel("Number of trees")
plt.ylabel("OOB error rate")
plt.title("Random Forest Performance: OOB Error vs Number of Trees")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# VARIABLE IMPORTANCE
# ============================================================================
"""
The output feature_importances_ measures importance based on RSS gain.
"""

# Reference: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
importances = rfModel.feature_importances_
std = np.std(
    [tree.feature_importances_ for tree in rfModel.estimators_], axis=0)
forest_importances = pd.DataFrame({'Importance': importances}, index=X.columns)
forest_importances = forest_importances.sort_values(by=["Importance"])

# Create the feature importance plot
fig, ax = plt.subplots(figsize=(10, 8))
forest_importances.plot.barh(y="Importance", ax=ax, legend=False)
ax.set_title("Feature importances using Mean Decrease in Impurity")
ax.set_xlabel("Mean decrease in impurity")
plt.tight_layout()
plt.show()

# Print feature importance summary
print("\nFeature Importance Summary:")
print("=" * 50)
for feature, importance in forest_importances.iterrows():
    print(f"{feature:15s}: {importance['Importance']:.4f}")

"""
For permutation based importance, see:
https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance
"""

print("\n" + "="*60)
print("RANDOM FOREST REGRESSION ANALYSIS COMPLETE")
print("="*60)
print(f"Final Test MSE: {test_mse:.6f}")
print(f"Training MSE (OOB): {oob_mse:.6f}")
print(f"Number of trees used: {rfModel.n_estimators}")
print(f"Number of features: {X.shape[1]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}") 