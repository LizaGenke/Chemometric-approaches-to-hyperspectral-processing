import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

class PLS:
    def __init__(self, ncomp, weights=None):
        self.ncomp = ncomp
        self.weights = weights

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        n, zp = X.shape
        q = Y.shape[1]

        if self.weights is None:
            self.weights = np.ones(n, dtype=np.float64) / n
        else:
            self.weights = np.asarray(self.weights, dtype=np.float64)
            self.weights = self.weights / np.sum(self.weights)

        self.xmeans = np.sum(self.weights[:, None] * X, axis=0)
        X = X - self.xmeans

        self.ymeans = np.sum(self.weights[:, None] * Y, axis=0)
        Y = Y - self.ymeans

        self.T = np.zeros((n, self.ncomp), dtype=np.float64)
        self.R = np.zeros((zp, self.ncomp), dtype=np.float64)
        self.W = np.zeros((zp, self.ncomp), dtype=np.float64)
        self.P = np.zeros((zp, self.ncomp), dtype=np.float64)
        self.C = np.zeros((q, self.ncomp), dtype=np.float64)
        self.TT = np.zeros(self.ncomp, dtype=np.float64)

        Xd = self.weights[:, None] * X
        tXY = Xd.T @ Y

        for a in range(self.ncomp):
            if q == 1:
                w = tXY[..., 0]
            else:
                # SVD for multi-target Y
                u, _, _ = np.linalg.svd(tXY.T, full_matrices=False)
                u = u[:, 0]
                w = tXY @ u

            w = w / np.sqrt(np.sum(w * w))

            r = w.copy()
            if a > 0:
                for j in range(a):
                    r = r - np.sum(self.P[:, j] * w) * self.R[:, j]

            t = X @ r
            tt = np.sum(self.weights * t * t)

            c = (tXY.T @ r) / tt
            p = (Xd.T @ t) / tt

            tXY = tXY - (p[:, None] @ c[None]) * tt

            self.T[:, a] = t
            self.P[:, a] = p
            self.W[:, a] = w
            self.R[:, a] = r
            self.C[:, a] = c
            self.TT[a] = tt

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = X - self.xmeans
        T_new = X @ self.R
        return T_new

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.T

    def get_params(self):
        return {
            "T": self.T,
            "P": self.P,
            "W": self.W,
            "C": self.C,
            "R": self.R,
            "TT": self.TT,
            "xmeans": self.xmeans,
            "ymeans": self.ymeans,
            "weights": self.weights,
            "T.ortorcho": True
        }

    def predict(self, X, nlv=None):
        X = np.asarray(X, dtype=np.float64)
        X = X - self.xmeans

        if nlv is None:
            nlv = self.ncomp
        else:
            nlv = min(nlv, self.ncomp)

        B = self.W[:, :nlv] @ np.linalg.inv(self.P[:, :nlv].T @ self.W[:, :nlv]) @ self.C[:, :nlv].T
        predictions = X @ B + self.ymeans
        return predictions
    
    
def rosa_pls(X_blocks, Y, ncomp_list):
    """
    ROSA-PLS for multiple X blocks using your NumPy PLS class.
    X_blocks: list of np.ndarray, each (n_samples, n_features_block)
    Y: np.ndarray, shape (n_samples, n_targets)
    ncomp_list: list of int, number of components for each block
    Returns: list of dicts with PLS models and predictions for each block
    """
    Y_res = Y.copy()
    X_res_blocks = [X.copy() for X in X_blocks]
    results = []

    for i, (X, ncomp) in enumerate(zip(X_blocks, ncomp_list)):
        pls = PLS(ncomp=ncomp)
        pls.fit(X_res_blocks[i], Y_res)
        Y_pred = pls.predict(X_res_blocks[i], ncomp)
        results.append({
            'pls': pls,
            'Y_pred': Y_pred,
            'block': i
        })
        # Update Y residual for next block
        Y_res = Y_res - Y_pred
        # Orthogonalize next X blocks with respect to current block's scores
        if i < len(X_blocks) - 1:
            T = pls.T[:, :ncomp]  # scores
            for j in range(i+1, len(X_blocks)):
                # Project and remove the part explained by T
                P = np.linalg.pinv(T) @ X_res_blocks[j]
                X_res_blocks[j] = X_res_blocks[j] - T @ P

    return results

def optimise_rosa_pls_cv_test(X_train_blocks, y_train, X_test_blocks, y_test, comp_combinations, n_splits=5):
    rmse_cv = np.zeros(len(comp_combinations))
    rmse_train = np.zeros(len(comp_combinations))
    rmse_test = np.zeros(len(comp_combinations))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for idx, comps_per_block in enumerate(comp_combinations):
        y_cv_pred_all = np.zeros_like(y_train, dtype=float)

        for train_idx, val_idx in kf.split(y_train):
            X_train_fold = [X_block[train_idx] for X_block in X_train_blocks]
            X_val_fold = [X_block[val_idx] for X_block in X_train_blocks]
            y_train_fold = y_train[train_idx]

            results = rosa_pls(X_train_fold, y_train_fold, comps_per_block)

            y_val_pred = np.sum([
                result['pls'].predict(X_val_fold[block_idx], comps_per_block[block_idx])
                for block_idx, result in enumerate(results)
            ], axis=0)

            y_cv_pred_all[val_idx] = y_val_pred

        # Fit on full training set to get train/test predictions
        results_full = rosa_pls(X_train_blocks, y_train, comps_per_block)
        y_train_pred = np.sum([r['Y_pred'] for r in results_full], axis=0)
        y_test_pred = np.sum([
            result['pls'].predict(X_test_blocks[k], comps_per_block[k])
            for k, result in enumerate(results_full)
        ], axis=0)

        # Store RMSEs
        rmse_cv[idx] = root_mean_squared_error(y_train, y_cv_pred_all)
        rmse_train[idx] = root_mean_squared_error(y_train, y_train_pred)
        rmse_test[idx] = root_mean_squared_error(y_test, y_test_pred)

    # Find best by CV RMSE
    best_idx = np.argmin(rmse_cv)
    best_components = comp_combinations[best_idx]
    print(f"Best component combination: {best_components}")

    return best_components