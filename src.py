import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, cross_validate
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline

# Classes
class RemoveOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
            z_scores = stats.zscore(X[numerical_cols])
            self.outliers_ = (abs(z_scores) > self.threshold).any(axis=1)
        elif isinstance(X, pd.Series):
            z_scores = stats.zscore(X)
            self.outliers_ = abs(z_scores) > self.threshold
        else:
            raise TypeError("Input should be a pandas DataFrame or Series")
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_transformed = X[~self.outliers_]
        elif isinstance(X, pd.Series):
            X_transformed = X[~self.outliers_]
        else:
            raise TypeError("Input should be a pandas DataFrame or Series")
        return X_transformed
    


# Select certain features (columns) from the dataset (top_features) -> (maybe unnecessary)
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, top_features):
        self.top_features = top_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.top_features]
    

# PREPROCESSING FUNCTIONS

# Function to use in method 1
def method1(model, X, y, preprocessing1):
    """
    This function creates a pipeline with a given regression model.
    It applies a logarithmic transformation to the target variable using TransformedTargetRegressor
    to handle skewness and makes the target distribution more Gaussian.
    The function then evaluates the model's performance using 10-fold cross-validation,
    measuring the root mean squared error (RMSE) and R-squared (R2) scores.
    Finally, it returns DataFrames summarizing the RMSE and R2 scores, including statistics such as mean, standard deviation, and percentiles.

    Parameters:
    - model: The regression model to be used in the pipeline.
    - X: The input features for the model.
    - y: The target variable.
    - preprocessing1: The preprocessing step to be applied to the input features.

    Returns:
    - cross_val_scores1: A DataFrame containing the RMSE and R2 scores from cross-validation.
    - rmse1_describe: A DataFrame containing descriptive statistics of the RMSE scores.
    - r2_1describe: A DataFrame containing descriptive statistics of the R2 scores.
    """
    target_regressor = TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1
    )

    pipeline1 = make_pipeline(preprocessing1, target_regressor)

    # Perform cross-validation with multiple scoring metrics
    scores1 = cross_validate(pipeline1, X, y, scoring=["neg_root_mean_squared_error", "r2"], 
                            cv=KFold(n_splits=10, shuffle=True, random_state=42), return_train_score=False)

    # Extract RMSE and R2 scores
    rmse1 = -scores1['test_neg_root_mean_squared_error']
    r2_1 = scores1['test_r2']

    # Describe RMSE and R2 scores
    rmse1_describe = pd.DataFrame(rmse1, index=range(1, len(rmse1)+1), 
                                  columns=[model.__class__.__name__ + " -> Method 1"]).describe().loc[["mean", "std", "min", "max"], :]
    r2_1describe = pd.DataFrame(r2_1, index=range(1, len(r2_1)+1), 
                                columns=[model.__class__.__name__ + " -> Method 1"]).describe().loc[["mean", "std", "min", "max"], :]

    cross_val_scores1 = pd.DataFrame({
        model.__class__.__name__ + " -> RMSE": rmse1,
        model.__class__.__name__ + " -> R2": r2_1
    }, index=range(1, len(rmse1)+1))

    return cross_val_scores1, rmse1_describe, r2_1describe



# Function to use in method 2
def method2(model, X_top, y, preprocessing2):
    """
    This function creates a machine learning pipeline with a given regression model and a preprocessing step.
    It applies a logarithmic transformation to the target variable to handle skewness and make the target distribution more Gaussian.
    The function then evaluates the model's performance using 10-fold cross-validation,
    measuring the root mean squared error (RMSE) and R-squared (R2) scores.
    The pipeline differs from method1 by using a different preprocessing step that utilizes only top features.
    Finally, it returns DataFrames summarizing the cross-validation scores and their descriptive statistics.

    Parameters:
    - model: The regression model to be used in the pipeline.
    - X_top: The input features for the model, consisting of top features.
    - y: The target variable.
    - preprocessing2: The preprocessing step to be applied to the input features.

    Returns:
    - cross_val_scores2: A DataFrame containing the RMSE and R2 scores from cross-validation.
    - rmse2_describe: A DataFrame containing descriptive statistics of the RMSE scores.
    - r2_2describe: A DataFrame containing descriptive statistics of the R2 scores.
    """
    target_regressor = TransformedTargetRegressor(
    regressor=model,
    func=np.log1p,
    inverse_func=np.expm1
    )

    pipeline2 = make_pipeline(preprocessing2, target_regressor)

    scores2 = cross_validate(pipeline2, X_top, y, scoring=["neg_root_mean_squared_error", "r2"],
                            cv=KFold(n_splits=10, shuffle=True, random_state=42), return_train_score=False)

    rmse2 = -scores2['test_neg_root_mean_squared_error']
    r2_2 = scores2['test_r2']

    rmse2_describe = pd.DataFrame(rmse2, index=range(1, len(rmse2)+1), 
                                columns=[model.__class__.__name__ + " -> Method 2"]).describe().loc[["mean", "std", "min", "max"], :]
    r2_2describe = pd.DataFrame(r2_2, index=range(1, len(r2_2)+1),
                                columns=[model.__class__.__name__ + " -> Method 2"]).describe().loc[["mean", "std", "min", "max"], :]

    cross_val_scores2 = pd.DataFrame({
        model.__class__.__name__ + " -> RMSE": rmse2,
        model.__class__.__name__ + " -> R2": r2_2
    }, index=range(1, len(rmse2)+1))

    return cross_val_scores2, rmse2_describe, r2_2describe



# Function to use in method 3
def method3(model, X, y, preprocessing3):
    """
    This function creates a machine learning pipeline with a given regression model and a preprocessing step.
    Unlike method1 and method2, it does not apply any transformation to the target variable.
    The function then evaluates the model's performance using 10-fold cross-validation,
    measuring the root mean squared error (RMSE) and R-squared (R2) scores.
    Finally, it returns DataFrames summarizing the cross-validation scores and their descriptive statistics.

    Parameters:
    - model: The regression model to be used in the pipeline.
    - X: The input features for the model.
    - y: The target variable.
    - preprocessing3: The preprocessing step to be applied to the input features.

    Returns:
    - cross_val_scores3: A DataFrame containing the RMSE and R2 scores from cross-validation.
    - rmse3_describe: A DataFrame containing descriptive statistics of the RMSE scores.
    - r2_3describe: A DataFrame containing descriptive statistics of the R2 scores.
    """
    pipeline3 = make_pipeline(preprocessing3, model)

    scores3 = cross_validate(pipeline3, X, y, scoring=["neg_root_mean_squared_error", "r2"],
                            cv=KFold(n_splits=10, shuffle=True, random_state=42), return_train_score=False)

    rmse3 = -scores3['test_neg_root_mean_squared_error']
    r2_3 = scores3['test_r2']

    rmse3_describe = pd.DataFrame(rmse3, index=range(1, len(rmse3)+1), 
                                columns=[model.__class__.__name__ + " -> Method 3"]).describe().loc[["mean", "std", "min", "max"], :]
    r2_3describe = pd.DataFrame(r2_3, index=range(1, len(r2_3)+1), 
                                columns=[model.__class__.__name__ + " -> Method 3"]).describe().loc[["mean", "std", "min", "max"], :]

    cross_val_scores3 = pd.DataFrame({
        model.__class__.__name__ + " -> RMSE": rmse3,
        model.__class__.__name__ + " -> R2": r2_3
    }, index=range(1, len(rmse3)+1))

    return cross_val_scores3, rmse3_describe, r2_3describe



# OTHER FUNCTIONS

# Plot accuracy curve
def plot_accuracy_curve(train_sizes, train_scores, test_scores, model, preprocessing_name, results_df):
    """
    This function plots the training and validation scores against the training set sizes, and
    calculates the final scores and their difference. The plot is displayed immediately.

    Parameters:
    - train_sizes: Array of training set sizes used to generate the learning curve.
    - train_scores: Array of training scores for each training set size.
    - test_scores: Array of validation scores for each training set size.
    - model: The machine learning model used in the pipeline.
    - preprocessing_name: Name of the preprocessing pipeline used in the model.
    - results_df: DataFrame to store the final training and validation scores, along with the difference.

    Returns:
    - results_df: Updated DataFrame containing the model, preprocessing pipeline, final training score,
                final validation score, and the difference between the two scores.
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig = plt.figure(figsize=(16, 9))

    plt.plot(train_sizes, train_mean, label="Training Score", color="blue", marker="o")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    plt.plot(train_sizes, test_mean, label="Validation Score", color="green", marker="o")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="green")

    last_train_score = train_mean[-1]
    last_validation_score = test_mean[-1]
    mean_score = last_train_score - last_validation_score

    model_preprocessing = f"{str(model)} -> {preprocessing_name}"

    results_df = pd.concat([
        results_df,
        pd.DataFrame({
            'Model': [model_preprocessing],
            'Final Training Score': [last_train_score],
            'Final Validation Score': [last_validation_score],
            'Difference': [mean_score]
        })
    ], ignore_index=True)

    plt.title(f"Learning Curve {model_preprocessing}")
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

    return results_df
