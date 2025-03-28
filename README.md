
# LASSO Regression with Homotopy Path

## Team Members:
- 1.Bhavana Sunkari ,CWID : A20543227
- 2.Suraj Ghatage , CWID : A20539840
## Project Overview

This repository implements the LASSO (Least Absolute Shrinkage and Selection Operator) regression technique using the Homotopy method for model fitting. This approach is highly efficient for solving LASSO by providing a piecewise path for all possible regularization strengths, leading to optimal feature selection and regularization in high-dimensional data.

## What is LASSO Regression?

LASSO is a type of regression that imposes an L1 penalty on the coefficients to prevent overfitting and encourage sparsity. By shrinking coefficients, some of them become exactly zero, which can result in automatic feature selection. The LASSO optimization problem is defined as:

$$\min_{\beta} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$

Where:
- $X$ is the input feature matrix.
- $y$ is the output vector.
- $\beta$ is the coefficient vector to be estimated.
- $\lambda$ is the regularization parameter that controls the sparsity of the solution.

## What is the Homotopy Method?

The Homotopy method provides a computationally efficient approach to solve the LASSO regression for multiple values of the regularization parameter ($\lambda$). Instead of solving the LASSO for each value of $\lambda$ independently, it tracks the solution across the entire regularization path by gradually adjusting $\lambda$ from a starting point where all coefficients are zero to values where they may become non-zero.

This allows for a faster calculation of the entire solution path, making it ideal for datasets where we want to explore various regularization strengths without recalculating the model repeatedly.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lasso-homotopy.git
   cd lasso-homotopy
   ```

2. Set up a virtual environment and install the dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Example: Fitting the LASSO Model

```python
import numpy as np
from model.LassoHomotopy import LassoHomotopyModel

# Generate random data
X = np.random.randn(100, 10)  # 100 samples, 10 features
true_coef = np.zeros(10)
true_coef[:3] = [1.0, -0.5, 0.25]  # Non-zero coefficients for first 3 features
y = X @ true_coef + 0.1 * np.random.randn(100)  # Add noise to the data

# Initialize and fit the LASSO model using Homotopy
model = LassoHomotopyModel()
result = model.fit(X, y)

# Output the estimated coefficients
print("Estimated coefficients:", result.coef_)

# Making predictions
X_test = np.random.randn(20, 10)  # Generate new test data
predictions = result.predict(X_test)
print("Predictions:", predictions)
```

### Key Model Parameters

- `max_iter` (int, default=1000): Maximum iterations for the Homotopy algorithm.
- `tol` (float, default=1e-6): Convergence tolerance for the iterative method.
- `lambda_max` (float, optional): Maximum regularization strength at the start. If None, it is set to the maximum feature-target correlation.
- `lambda_min_ratio` (float, default=1e-3): Minimum regularization strength as a fraction of `lambda_max`.

### Feature Selection with Collinear Data

LASSO is particularly effective when dealing with collinear features. Below is an example of how the model performs in such cases:

```python
# Generate collinear data
X = np.random.randn(100, 20)
X[:, 10:] = X[:, :10] + 0.1 * np.random.randn(100, 10)  # Introduce collinearity

# True coefficients for the first 5 features
true_coef = np.zeros(20)
true_coef[:5] = [1.0, -0.5, 0.25, -0.3, 0.1]
y = X @ true_coef + 0.1 * np.random.randn(100)

# Fit the LASSO model
model = LassoHomotopyModel()
result = model.fit(X, y)

# Identify non-zero coefficients
non_zero_indices = np.where(np.abs(result.coef_) > 1e-6)[0]
print("Indices of non-zero coefficients:", non_zero_indices)
```

## Running Tests

To run tests that ensure the model works as expected, use `pytest`:

```bash
python -m pytest tests/
```

### Test Coverage Includes:

- **Synthetic Data**: Verifying correct feature selection by testing on datasets with known sparse coefficients.
- **Collinear Data**: Ensuring that collinearity does not affect the sparsity and accuracy of the model.
- **Regularization Variations**: Confirming that the model reacts as expected to different values of the regularization parameter.
- **General Model Functionality**: Ensuring the model works on various data types and sizes.

## Test Results Output

Here are the results from running the tests:

- **Test Case 1: Basic Functionality** - Passed. The model successfully learned and predicted the sparse coefficients.
- **Test Case 2: Feature Selection on Synthetic Data** - Passed. The model correctly identified the non-zero coefficients.
- **Test Case 3: Handling Collinearity** - Passed. The model effectively handled collinear features and selected the correct features.
- **Test Case 4: Regularization Behavior** - Passed. As expected, increasing regularization led to more coefficients being zeroed.

Sample output from the test results:

```bash
================================================= test session starts =================================================
platform win32 -- Python 3.12.7, pytest-8.3.5, pluggy-1.5.0
rootdir: C:\Users\pamid\OneDrive\Documents\lasso-homotopy\tests
collected 4 items

tests/test_lasso_homotopy.py::test_basic_functionality PASSED                    [ 25%]
tests/test_lasso_homotopy.py::test_feature_selection PASSED                     [ 50%]
tests/test_lasso_homotopy.py::test_collinearity_handling PASSED                  [ 75%]
tests/test_lasso_homotopy.py::test_regularization_behavior PASSED               [100%]

================================================= 4 passed in 1.25 seconds =================================================
```

These results confirm that the LASSO regression model with the Homotopy method works correctly across different test cases.

## Answering Key Questions

### What does the model do, and when should it be used?

This model performs LASSO regression via the Homotopy method. It is best used when you need:

1. Feature selection in high-dimensional datasets.
2. Regularization to prevent overfitting, especially when some features are collinear or irrelevant.
3. A sparse solution where only a subset of the features is selected by the model.

This method is particularly valuable for exploring the solution across various regularization strengths efficiently.

### How did you validate your model?

We validated the model with multiple tests, including:

1. **Synthetic data with known sparse solutions** to check if the model accurately identifies the active features.
2. **Collinear data** to ensure that LASSO can effectively handle highly correlated predictors and still select a sparse set of features.
3. **Varying the regularization strength** to observe the expected sparsity pattern and ensure proper convergence.
4. **Multiple datasets of different sizes** to confirm robustness across various data dimensions.

### What parameters can be adjusted to optimize performance?

The following parameters allow users to tune the model:

1. `max_iter`: Controls the number of iterations the algorithm runs for.
2. `tol`: Sets the convergence threshold for stopping the algorithm.
3. `lambda_max`: Defines the maximum value of the regularization parameter, which influences the shrinkage of coefficients.
4. `lambda_min_ratio`: Determines the minimum regularization value as a proportion of `lambda_max`.

### What challenges did you encounter, and could they be overcome?

Some challenges include:

1. **High-dimensional data**: The algorithm may become slow or memory-intensive for very high-dimensional datasets. This could be mitigated with optimizations such as sparse matrix handling or more efficient matrix operations.
   
2. **Numerical instability**: In cases of extreme collinearity, the model might face difficulties with matrix inversion. This could be addressed by implementing more robust techniques like regularized inversions.

3. **Large datasets**: Handling datasets larger than memory can be tricky. An online learning or mini-batch approach could be considered for scalability.

4. **Convergence speed**: For certain datasets, the algorithm may require many iterations. This could be improved with adaptive stopping criteria.

## Project Structure

```
/LassoHomotopy
├── model/
│   └── LassoHomotopy.py    # Core implementation of the LASSO Homotopy algorithm
├── tests/
│   ├── test_LassoHomotopy.py  # Test cases for the model
│   ├── small_test.csv      # Small test dataset
│   └── collinear_data.csv  # Dataset with collinear features
└── requirements.txt        # Project dependencies
```

## Visualizing the Coefficient Path

One of the advantages of the Homotopy method is the ability to visualize how coefficients change with different regularization strengths:

```python
import matplotlib.pyplot as plt
import numpy as np
from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel

# Generate data and fit model as in previous examples
# ...

# After fitting the model
plt.figure(figsize=(10, 6))
for i in range(result.coef_path_.shape[1]):
    # Only plot non-zero coefficient paths to avoid clutter
    if np.any(np.abs(result.coef_path_[:, i]) > 1e-10):
        plt.plot(result.lambda_path_, result.coef_path_[:, i], label=f'Feature {i}')
plt.xscale('log')
plt.xlabel('Log(lambda)')
plt.ylabel('Coefficients')
plt.title('LASSO Coefficient Path using Homotopy Method')
plt.legend()
plt.show()
```

This visualization helps in understanding how features are selected as the regularization parameter changes.
