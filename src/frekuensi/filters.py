import numpy as np
from scipy.ndimage import gaussian_filter

KINDS = ("lowpass", "highpass")


def _as_image(image):
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("image must be a 2D grayscale array")
    return array


def _check_kind(kind):
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def _check_cutoff(cutoff):
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")


def _finish(lowpass, kind):
    _check_kind(kind)
    return lowpass if kind == "lowpass" else 1.0 - lowpass


def to_frequency(image):
    return np.fft.fftshift(np.fft.fft2(_as_image(image)))


def to_spatial(shifted):
    return np.fft.ifft2(np.fft.ifftshift(shifted))


def spectrum(image):
    return 20 * np.log1p(np.abs(to_frequency(image)))


def distance_grid(shape):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y = np.arange(-crow, rows - crow)[:, np.newaxis]
    x = np.arange(-ccol, cols - ccol)[np.newaxis, :]
    return np.sqrt(y**2 + x**2)


def ideal_mask(shape, cutoff, kind="lowpass"):
    _check_cutoff(cutoff)
    lowpass = (distance_grid(shape) < cutoff).astype(np.float64)
    return _finish(lowpass, kind)


def butterworth_mask(shape, cutoff, order=2, kind="lowpass"):
    _check_cutoff(cutoff)
    if order <= 0:
        raise ValueError("order must be positive")
    lowpass = 1.0 / (1.0 + (distance_grid(shape) / cutoff) ** (2 * order))
    return _finish(lowpass, kind)


def gaussian_mask(shape, cutoff, kind="lowpass"):
    _check_cutoff(cutoff)
    sigma = cutoff / np.sqrt(2 * np.log(2))
    lowpass = np.exp(-(distance_grid(shape) ** 2) / (2 * sigma**2))
    return _finish(lowpass, kind)


def apply_mask(image, mask):
    array = _as_image(image)
    mask = np.asarray(mask, dtype=np.float64)
    if mask.shape != array.shape:
        raise ValueError(f"mask shape {mask.shape} does not match image shape {array.shape}")
    return to_spatial(to_frequency(array) * mask).real


def unsharp_mask(image, sigma=1.0, amount=1.0):
    array = _as_image(image)
    blurred = gaussian_filter(array, sigma=sigma)
    return array + amount * (array - blurred)
