import csv
import os
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
from LassoHomotopy import LassoHomotopyModel

def load_data_from_csv(file_name):
    """
    Reads the CSV file and returns features (X) and target (y) arrays.
    Ensures that data is processed correctly even if column names vary.
    
    Parameters:
    - file_name: str, path to the CSV file.
    
    Returns:
    - X: numpy array of feature values.
    - y: numpy array of target values.
    """
    data = []
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, file_name)

    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append({k: float(v) for k, v in row.items()})

    # Extract features (any column name starting with 'x' or 'X')
    X = np.array([[value for key, value in entry.items() if key.lower().startswith('x')] for entry in data])

    # Ensure that there is a target column (either 'y' or 'target')
    if 'y' in data[0]:
        y = np.array([entry['y'] for entry in data])
    elif 'target' in data[0]:
        y = np.array([entry['target'] for entry in data])
    else:
        raise ValueError("Neither 'y' nor 'target' column found in the dataset")

    return X, y


def test_small_dataset_with_lasso():
    """
    Tests the LassoHomotopy model on a small dataset to ensure that
    the model is capable of learning non-zero coefficients.
    """
    # Load the dataset for testing
    X, y = load_data_from_csv("small_test.csv")

    # Initialize and train the model
    model = LassoHomotopyModel()
    result = model.fit(X, y)

    # Validate that the model learns meaningful coefficients
    assert np.any(result.coef_ != 0), "Expected non-zero coefficients in the model"

    # Validate that predictions have the same shape as the target values
    predictions = result.predict(X)
    assert predictions.shape == y.shape, "Predictions shape does not match the target shape"


def test_lasso_with_collinear_features():
    """
    Evaluates the model's ability to handle data with highly collinear features.
    Lasso should select a sparse set of coefficients in such cases.
    """
    # Load dataset that contains collinear features
    X, y = load_data_from_csv("collinear_data.csv")

    # Use a model with less regularization to allow for more coefficients to be learned
    model = LassoHomotopyModel(lambda_min_ratio=1e-5)
    result = model.fit(X, y)

    # Count the number of zero coefficients in the fitted model
    zero_coeffs = np.sum(np.abs(result.coef_) < 1e-10)
    total_coeffs = len(result.coef_)

    # Assert that there are non-zero coefficients (i.e., the solution should be sparse)
    assert zero_coeffs > 0, "Expected sparse solution with many zero coefficients in collinear data"

    # Print the sparsity of the solution for reference
    print(f"Coefficient sparsity: {zero_coeffs}/{total_coeffs} coefficients are zero")


def test_sparse_solution_on_synthetic_data():
    """
    Verifies that the LassoHomotopy model correctly identifies sparse solutions
    when trained on synthetic data.
    """
    X, y = load_data_from_csv("collinear_data.csv")

    # Standardize X for numerical stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    X_standardized = (X - X_mean) / X_std

    # Fit the model with very low regularization to encourage sparse solutions
    model = LassoHomotopyModel(lambda_min_ratio=1e-10, max_iter=5000, tol=1e-8)
    result = model.fit(X_standardized, y)

    # Scale coefficients back to the original scale
    original_scale_coeffs = result.coef_ / X_std

    # Identify indices with non-zero coefficients
    non_zero_indices = np.where(np.abs(original_scale_coeffs) > 0.01)[0]

    # Print information on non-zero coefficients for insight
    print(f"Number of features: {X.shape[1]}")
    print(f"Non-zero coefficients: {len(non_zero_indices)} out of {X.shape[1]}")
    print(f"Indices with non-zero coefficients: {non_zero_indices}")
    print(f"Non-zero coefficient values: {original_scale_coeffs[non_zero_indices]}")

    # Ensure that the solution is sparse
    assert len(non_zero_indices) < X.shape[1], "Model should produce a sparse solution with fewer non-zero coefficients"

    # Ensure that at least one coefficient is non-zero
    assert len(non_zero_indices) > 0, "Expected at least one non-zero coefficient"

    # Optionally, print mean squared error (MSE) for additional insight
    predictions = result.predict(X_standardized)
    mse = np.mean((y - predictions) ** 2)
    mean_mse = np.mean((y - np.mean(y)) ** 2)
    print(f"MSE of the model: {mse}")
    print(f"Mean predictor MSE (baseline): {mean_mse}")


def test_lambda_parameter_variation():
    """
    Tests how the LassoHomotopy model behaves with different values of the lambda regularization parameter.
    """
    # Load the dataset for testing
    X, y = load_data_from_csv("small_test.csv")

    # Fit with a higher lambda (stronger regularization)
    model_high_lambda = LassoHomotopyModel(lambda_min_ratio=0.5)
    result_high_lambda = model_high_lambda.fit(X, y)

    # Fit with a lower lambda (weaker regularization)
    model_low_lambda = LassoHomotopyModel(lambda_min_ratio=1e-6)
    result_low_lambda = model_low_lambda.fit(X, y)

    # Check how many coefficients are zero for each model
    high_lambda_zeros = np.sum(np.abs(result_high_lambda.coef_) < 1e-10)
    low_lambda_zeros = np.sum(np.abs(result_low_lambda.coef_) < 1e-10)

    print(f"High regularization: {high_lambda_zeros}/{len(result_high_lambda.coef_)} coefficients are zero")
    print(f"Low regularization: {low_lambda_zeros}/{len(result_low_lambda.coef_)} coefficients are zero")

    # Validate that predictions are correctly shaped
    predictions_high = result_high_lambda.predict(X)
    predictions_low = result_low_lambda.predict(X)

    assert predictions_high.shape == y.shape
    assert predictions_low.shape == y.shape
