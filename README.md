# House Prices Prediction - Advanced Regression Techniques

## Project Overview

This project was developed as part of the **Multivariate Data Analysis** course in a team of five. Originally hosted on GitLab, this is a re-upload to GitHub. The objective is to apply **regression techniques** to predict house prices using the **Ames Housing dataset**. The project involves feature selection, preprocessing, outlier handling, and model evaluation to achieve high prediction accuracy.

## Ames Housing Dataset

The **Ames Housing dataset** is a rich collection of residential property data from Ames, Iowa. Compiled for educational purposes, it serves as an alternative to the classic **Boston Housing dataset**, offering 79 explanatory variables that describe almost every aspect of residential homes.

### Dataset Features

- **79 explanatory variables**: Covering details such as size, type, construction quality, year built, neighborhood, and physical features.
- **Sales Price Prediction**: The target variable is `SalePrice`, representing the final sale price of a home.
- **Comprehensive dataset**: Provides an opportunity to explore **feature engineering** and experiment with **preprocessing**.

More details about the dataset can be found on Kaggle: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

## Data Preprocessing

### Handling Missing Values

Some categorical features contain `NaN` values, which often represent missing properties rather than missing data. We replaced `NaN` values with explicit categories:

| Feature        | `NaN` Meaning |
|---------------|---------------|
| Alley        | No alley access |
| MasVnrType   | None (Potentially Dropped) |
| FireplaceQu  | No Fireplace |
| PoolQC       | No Pool |
| Fence        | No Fence |
| MiscFeature  | None (May be Important) |

### Outlier Detection and Handling

Setting a default threshold, **28.97%** of data rows were identified as potential outliers. Based on analysis, we chose to remove outliers beyond **5 standard deviations** from the mean, reducing data loss while maintaining integrity:

- **3 std** → 28.98% of data removed
- **4 std** → 14.18% of data removed
- **5 std** → 5.75% of data removed *(Chosen threshold)*

### Feature Engineering and Transformations

- **Skewed Features**: Right-skewed distributions were transformed using **logarithmic and power transformations** to normalize distributions and improve linear regression performance.
- **Correlation Analysis**: Features highly correlated with `SalePrice` were prioritized.
- **Categorical Encoding**: Used **OneHotEncoder** to convert categorical variables into binary features.

### Data Processing Pipelines

1. **Skewed Numerical Features (`skewed_pipeline`)**:
   - Imputation using `SimpleImputer` (median strategy)
   - Standardization using `PowerTransformer`
2. **Non-Skewed Numerical Features (`num_pipeline`)**:
   - Imputation using `SimpleImputer` (median strategy)
   - Standardization using `StandardScaler`
3. **Categorical Features (`cat_pipeline`)**:
   - Imputation using `SimpleImputer` (new categories for NaNs)
   - Encoding using `OneHotEncoder`

## Experimental Setup

To determine the best preprocessing approach, **three methods** were applied:

1. **Method 1:**
   - Uses **all columns**
   - Applies `PowerTransformer` to skewed data
   - Uses `TransformedTargetRegressor` for `SalePrice`
2. **Method 2:**
   - Uses only **top features** (highly correlated with `SalePrice`)
   - Same transformations as Method 1
3. **Method 3:**
   - Uses **all columns**
   - No `PowerTransformer` or `TransformedTargetRegressor`

Each method was evaluated using **cross-validation scores** based on **R2** and **RMSE metrics**.

## Model Selection

A variety of regression models were tested. The **Ridge Regression** model, trained using **Method 1**, produced the best results:

- **Better performance** with skewed transformations
- **Lower cross-validation variance**
- **Best trade-off** between accuracy and stability

### Overfitting Analysis

To check for overfitting, **learning curves** were plotted and compared for different values of `alpha`. The model with `alpha=10` was chosen for its balance between:

- **Final Training Score**: 0.9348
- **Final Validation Score**: 0.9109
- **Difference**: 0.0239 (~2.4%)

## Next Steps

- **Feature Selection Refinements** to remove redundant variables
- **Test Set Predictions** using the final model

## Repository Structure

```
├── data/                  # Original datasets
├── data_description.txt   # Dataset documentation
├── main_notebook          # Main project notebook
├── README.md              # Project documentation
├── src.py                 # All classes and functions used
```

## Conclusion

The **House Prices Prediction** project demonstrates the importance of **feature engineering**, **data preprocessing**, and **model selection** for accurate real-estate price prediction. Using Ridge Regression with transformed data yielded the best results, balancing accuracy and generalization.

Further improvements can be made through **feature refinement**, and **ensemble modeling**.
