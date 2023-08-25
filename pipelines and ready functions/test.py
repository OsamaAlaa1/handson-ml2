# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Create a sample dataset
np.random.seed(42)
X = np.random.rand(100, 5)  # Generating 100 samples with 5 features
y = 2*X[:, 0] + 3*X[:, 1] - 1.5*X[:, 2] + np.random.randn(100)  # True target values with added noise

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
def linear_regression(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Ridge Regression
def ridge_regression(X_train, y_train, X_test):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Lasso Regression
def lasso_regression(X_train, y_train, X_test):
    model = Lasso(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# ElasticNet Regression
def elastic_net_regression(X_train, y_train, X_test):
    model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Decision Tree Regression
def decision_tree_regression(X_train, y_train, X_test):
    model = DecisionTreeRegressor(max_depth=None, min_samples_split=2)
    # max_depth: Maximum depth of the tree. Controls the level of complexity.
    # min_samples_split: Minimum number of samples required to split an internal node.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Random Forest Regression
def random_forest_regression(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2)
    # n_estimators: Number of trees in the forest.
    # max_depth: Maximum depth of each tree in the forest.
    # min_samples_split: Minimum number of samples required to split an internal node.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Gradient Boosting Regression
def gradient_boosting_regression(X_train, y_train, X_test):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    # n_estimators: Number of boosting stages (trees) to be built.
    # learning_rate: Controls the contribution of each tree to the final prediction.
    # max_depth: Maximum depth of each tree in the ensemble.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# XGBoost Regression
def xgboost_regression(X_train, y_train, X_test):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    # n_estimators: Number of boosting stages (trees) to be built.
    # learning_rate: Controls the contribution of each tree to the final prediction.
    # max_depth: Maximum depth of each tree in the ensemble.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# LightGBM Regression
def lightgbm_regression(X_train, y_train, X_test):
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    # n_estimators: Number of boosting stages (trees) to be built.
    # learning_rate: Controls the contribution of each tree to the final prediction.
    # max_depth: Maximum depth of each tree in the ensemble.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# CatBoost Regression
def catboost_regression(X_train, y_train, X_test):
    model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=3)
    # iterations: Number of boosting stages (trees) to be built.
    # learning_rate: Controls the contribution of each tree to the final prediction.
    # depth: Maximum depth of each tree in the ensemble.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Support Vector Regression
def svr_regression(X_train, y_train, X_test):
    model = SVR(kernel='rbf', C=1.0)
    # kernel: Specifies the kernel type used in the algorithm.
    # C: Regularization parameter. Controls the trade-off between fitting to the data and allowing margin violations.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# K-Nearest Neighbors Regression
def knn_regression(X_train, y_train, X_test):
    model = KNeighborsRegressor(n_neighbors=5)
    # n_neighbors: Number of neighbors to use for prediction.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Neural Network Regression
def neural_network_regression(X_train, y_train, X_test):
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, alpha=0.0001)
    # hidden_layer_sizes: Tuple representing the number of neurons in each hidden layer.
    # max_iter: Maximum number of iterations to converge.
    # alpha: L2 regularization term.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Evaluate Models
def evaluate_models(y_true, y_preds):
    for model_name, y_pred in y_preds.items():
        mse = mean_squared_error(y_true, y_pred)
        print(f"{model_name} MSE: {mse:.4f}")

# Perform predictions and evaluate
y_preds = {
    'Linear Regression': linear_regression(X_train, y_train, X_test),
    'Ridge Regression': ridge_regression(X_train, y_train, X_test),
    'Lasso Regression': lasso_regression(X_train, y_train, X_test),
    'ElasticNet Regression': elastic_net_regression(X_train, y_train, X_test),
    'Decision Tree Regression': decision_tree_regression(X_train, y_train, X_test),
    'Random Forest Regression': random_forest_regression(X_train, y_train, X_test),
    'Gradient Boosting Regression': gradient_boosting_regression(X_train, y_train, X_test),
    'XGBoost Regression': xgboost_regression(X_train, y_train, X_test),
    'LightGBM Regression': lightgbm_regression(X_train, y_train, X_test),
    'CatBoost Regression': catboost_regression(X_train, y_train, X_test),
    'SVR Regression': svr_regression(X_train, y_train, X_test),
    'KNN Regression': knn_regression(X_train, y_train, X_test),
    'Neural Network Regression': neural_network_regression(X_train, y_train, X_test)
}

# Evaluate and print results
evaluate_models(y_test, y_preds)
