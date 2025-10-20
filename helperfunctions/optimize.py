import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import root_mean_squared_error, r2_score

def optimise_pls_cv_test(X_train, y_train, X_test, y_test, n_comp):
    """
    Run PLS with multiple outputs, using cross-validation to compute MSE for each number of components.
    Also evaluate and plot test set performance.
    
    Parameters
    ----------
    X_train : array-like, shape (n_samples_train, n_features)
        Calibration (training) features.
    y_train : array-like, shape (n_samples_train, n_targets)
        Calibration (training) target(s).
    X_test : array-like, shape (n_samples_test, n_features)
        Test set features.
    y_test : array-like, shape (n_samples_test, n_targets)
        Test set target(s).
    n_comp : int
        Maximum number of PLS components to try.
    
    Returns
    -------
    n_comp: int of optimal number of PLS components
    """
    rmse_cv = []
    rmse_test = []
    rmse_train = []
    components = np.arange(1, n_comp + 1)

    for i in components:
        pls = PLSRegression(n_components=i)
        y_cv = cross_val_predict(pls, X_train, y_train, cv=10)
        rmse_cv.append(root_mean_squared_error(y_train, y_cv))
        pls = PLSRegression(n_components=i)
        pls.fit(X_train, y_train)
        y_train_pred = pls.predict(X_train)
        y_test_pred = pls.predict(X_test)
        rmse_train.append(root_mean_squared_error(y_train, y_train_pred))
        rmse_test.append(root_mean_squared_error(y_test, y_test_pred))
        sys.stdout.write(f"\r{i/n_comp*100:.1f}% completed")
        sys.stdout.flush()
    sys.stdout.write("\n")

    best_idx = int(np.argmin(rmse_cv))
    best_n = best_idx + 1
    print("Suggested number of components:", best_n)

    with plt.style.context("ggplot"):
        plt.figure(figsize=(8, 5))
        plt.plot(components, rmse_train, label='Training RMSE', marker='o', color='green')
        plt.plot(components, rmse_cv, label='CV RMSE', marker='o', color='red')
        plt.plot(components, rmse_test, marker='o', color='blue', label='Test RMSE')
        plt.xlabel('Number of PLS Components')
        plt.ylabel('Root Mean Squared Error')
        plt.title('Bias-Variance Trade-off in PLS')
        plt.legend()
        plt.grid(True)
        plt.show()
    return best_n