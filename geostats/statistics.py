import numpy as np
import torch
from scipy.stats import skew

from .smoothness_losses import SmoothnessLoss


def skewness(y_pred: np.ndarray, mask: np.ndarray) -> float:
    masked_pred = y_pred[mask == 1]
    return skew(masked_pred)


def set_nans_to_zero(mask: np.ndarray) -> np.ndarray:
    mask[np.isnan(mask)] = 0
    return mask


def compute_deriv_mask(mask: np.ndarray, axis: int):
    rolled = np.roll(mask, 1, axis=axis)
    deriv_mask = mask * rolled
    slices = [slice(None)] * len(mask.shape)
    slices[axis] = slice(1, None)
    deriv_mask = deriv_mask[tuple(slices)]
    return deriv_mask


def nugget_effect_c0(y_pred: np.ndarray, mask: np.ndarray) -> float:
    y_pred_tensor = torch.tensor(
        y_pred[None, None, ...], dtype=torch.float32
    )  # Shape (1, 1, height, width)
    mask_tensor = torch.tensor(
        mask[None, None, ...], dtype=torch.float32
    )  # Shape (1, 1, height, width)
    with torch.no_grad():
        sl = SmoothnessLoss(lambd=1, avg_batch_size=None, device=torch.device("cpu"))
        sl(mask_tensor, y_pred_tensor)
        sl.compute()
    # Divide by 2 to account for the fact that we only divided by the mask size once,
    # even though we summed variance for distance h=[1, 0] and h=[0, 1]. Therefore, we
    # have only counted half of the pairs used in the computation of the variogram. We
    # divide by 2 to account for this fact. We divide by an additional 2 to account for
    # the factor 1/2 in the definition of the variogram. Therefore, we divide by a
    # total of 4.
    c0 = sl.value.numpy() / 4

    return c0


def neighbor_covariance(y_pred: np.ndarray, mask: np.ndarray) -> float:
    # Compute the covariance between all neighbors
    valid_y_pred = y_pred[..., :-1, :-1]
    valid_mask = mask[..., :-1, :-1]
    mean_pred = np.mean(valid_y_pred[valid_mask == 1])
    centered_pred = y_pred - mean_pred
    # masked_centered_pred = np.where(mask, centered_pred, 0)
    mask_size = np.sum(mask)
    horizontal_mult = centered_pred[..., :-1, 1:] * centered_pred[..., :-1, :-1]
    masked_horizontal_mult = horizontal_mult * valid_mask
    horizontal_cov = np.sum(masked_horizontal_mult) / mask_size
    vertical_mult = centered_pred[..., 1:, :-1] * centered_pred[..., :-1, :-1]
    masked_vertical_mult = vertical_mult * valid_mask
    vertical_cov = np.sum(masked_vertical_mult) / mask_size
    # vertical_cov = (
    #     np.sum(masked_centered_pred[..., :-1, :-1] * masked_centered_pred[..., 1:, :-1])
    #     / mask_size
    # )
    neighbor_cov = (horizontal_cov + vertical_cov) / 2

    return neighbor_cov


def double_c0(y_pred: np.ndarray, mask: np.ndarray) -> float:
    mask = mask[..., :-1, :-1]
    # dx = y_pred[..., :-1, 1:] - y_pred[..., :-1, :-1]
    # dy = y_pred[..., 1:, :-1] - y_pred[..., :-1, :-1]
    dx = np.diff(y_pred, axis=-2)[..., :, :-1]
    dy = np.diff(y_pred, axis=-1)[..., :-1, :]
    dx = dx * mask
    dy = dy * mask
    dx_sq = dx**2
    dy_sq = dy**2
    mask_size = np.sum(mask)
    double_c0 = (np.sum(dx_sq) / mask_size + np.sum(dy_sq) / mask_size) / 2
    return double_c0


# def double_var(y_pred: np.ndarray, mask: np.ndarray) -> float:
#     mean = np.mean(valid_masked_pred)


def sill_variance(y_pred: np.ndarray, mask: np.ndarray) -> float:
    valid_y_pred = y_pred[..., :-1, :-1]
    valid_mask = mask[..., :-1, :-1]
    valid_masked_pred = valid_y_pred[valid_mask == 1]
    sill = np.var(valid_masked_pred)
    # masked_pred = y_pred[mask == 1]
    # sill = np.var(masked_pred)

    return sill


def partial_sill_s1(c0, sill):
    return sill - c0


def spatially_structured_variance_c1(c0: float, sill: float) -> float:
    c1 = sill - c0
    return c1


def cambardella_index(c0, s1):
    return c0 / (s1 + c0) * 100


def spd_index(c0, s1):
    return s1 / (s1 + c0) * 100


def signal_to_noise_ratio(c0, c1):
    return c1 / c0
