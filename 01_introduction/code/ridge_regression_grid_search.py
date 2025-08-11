from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Example: Grid Search for Ridge Regression
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_alpha = grid_search.best_params_['alpha']
