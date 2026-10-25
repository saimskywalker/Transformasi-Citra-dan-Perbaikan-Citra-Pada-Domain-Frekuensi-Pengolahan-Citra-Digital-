import itertools

import numpy as np
import pytest

from frekuensi import (
    apply_mask,
    butterworth_mask,
    distance_grid,
    gaussian_mask,
    ideal_mask,
    spectrum,
    to_frequency,
    to_spatial,
    unsharp_mask,
)

SHAPES = [(64, 64), (63, 63), (48, 80), (51, 40)]


def _mask_builders(cutoff=10):
    return {
        "ideal": lambda shape, kind: ideal_mask(shape, cutoff, kind=kind),
        "butterworth": lambda shape, kind: butterworth_mask(shape, cutoff, order=2, kind=kind),
        "gaussian": lambda shape, kind: gaussian_mask(shape, cutoff, kind=kind),
    }


MASK_NAMES = ["ideal", "butterworth", "gaussian"]
KINDS = ["lowpass", "highpass"]


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 80)).astype(np.uint8)


def test_fft_ifft_roundtrip(image):
    restored = to_spatial(to_frequency(image))
    assert np.allclose(restored.imag, 0, atol=1e-9)
    assert np.allclose(restored.real, image, atol=1e-9)


def test_spectrum_shape_and_center(image):
    result = spectrum(image)
    assert result.shape == image.shape
    assert np.all(np.isfinite(result))
    rows, cols = image.shape
    assert np.unravel_index(np.argmax(result), result.shape) == (rows // 2, cols // 2)


@pytest.mark.parametrize("shape", SHAPES)
def test_distance_grid_center_is_dc(shape):
    grid = distance_grid(shape)
    assert grid.shape == shape
    assert grid[shape[0] // 2, shape[1] // 2] == 0
    dc = np.zeros(shape)
    dc[shape[0] // 2, shape[1] // 2] = 1
    assert np.argmax(np.abs(np.fft.ifftshift(dc))) == 0


@pytest.mark.parametrize("name", MASK_NAMES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_mask_range_shape_dtype(name, kind, shape):
    mask = _mask_builders()[name](shape, kind)
    assert mask.shape == shape
    assert mask.dtype == np.float64
    assert mask.min() >= 0
    assert mask.max() <= 1


@pytest.mark.parametrize("name", MASK_NAMES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_mask_is_point_symmetric(name, kind, shape):
    unshifted = np.fft.ifftshift(_mask_builders()[name](shape, kind))
    negated = np.roll(unshifted[::-1, ::-1], 1, axis=(0, 1))
    assert np.allclose(unshifted, negated)


@pytest.mark.parametrize("name", MASK_NAMES)
@pytest.mark.parametrize("shape", [(63, 63), (51, 41)])
def test_mask_is_mirror_symmetric_for_odd_shapes(name, shape):
    mask = _mask_builders()[name](shape, "lowpass")
    assert np.allclose(mask, mask[::-1, :])
    assert np.allclose(mask, mask[:, ::-1])


@pytest.mark.parametrize("name", MASK_NAMES)
@pytest.mark.parametrize("shape", SHAPES)
def test_highpass_is_one_minus_lowpass(name, shape):
    build = _mask_builders()[name]
    assert np.allclose(build(shape, "highpass"), 1 - build(shape, "lowpass"))


@pytest.mark.parametrize("name", MASK_NAMES)
def test_lowpass_preserves_mean(name, image):
    mask = _mask_builders()[name](image.shape, "lowpass")
    result = apply_mask(image, mask)
    assert result.mean() == pytest.approx(image.mean(), abs=1e-9)


@pytest.mark.parametrize("name", MASK_NAMES)
def test_highpass_removes_mean(name, image):
    mask = _mask_builders()[name](image.shape, "highpass")
    result = apply_mask(image, mask)
    assert result.mean() == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("name", MASK_NAMES)
@pytest.mark.parametrize("kind", KINDS)
def test_apply_mask_returns_real_float_image(name, kind, image):
    result = apply_mask(image, _mask_builders()[name](image.shape, kind))
    assert result.shape == image.shape
    assert result.dtype == np.float64
    assert not np.iscomplexobj(result)


def test_apply_mask_rejects_wrong_shape(image):
    with pytest.raises(ValueError):
        apply_mask(image, np.ones((3, 3)))


def test_lowpass_smooths_and_highpass_keeps_detail(image):
    low = apply_mask(image, gaussian_mask(image.shape, 8, kind="lowpass"))
    high = apply_mask(image, gaussian_mask(image.shape, 8, kind="highpass"))
    assert low.std() < image.std()
    assert np.allclose(low + high, image)


@pytest.mark.parametrize("kind", KINDS)
def test_butterworth_approaches_ideal_with_higher_order(kind):
    shape = (64, 64)
    ideal = ideal_mask(shape, 12, kind=kind)
    errors = [
        np.abs(butterworth_mask(shape, 12, order=order, kind=kind) - ideal).mean()
        for order in (1, 2, 4, 8, 16)
    ]
    assert all(later < earlier for earlier, later in itertools.pairwise(errors))


def test_gaussian_is_half_at_cutoff():
    shape = (65, 65)
    mask = gaussian_mask(shape, 10)
    assert mask[32, 42] == pytest.approx(0.5)
    assert butterworth_mask(shape, 10, order=3)[32, 42] == pytest.approx(0.5)


def test_ideal_cutoff_matches_notebook_convention():
    shape = (65, 65)
    lowpass = ideal_mask(shape, 10)
    assert lowpass[32, 41] == 1
    assert lowpass[32, 42] == 0


def test_unsharp_increases_edge_contrast():
    step = np.zeros((32, 32))
    step[:, 16:] = 1.0
    sharpened = unsharp_mask(step, sigma=1.5, amount=1.0)
    assert sharpened.shape == step.shape
    assert sharpened.dtype == np.float64
    before = step[:, 16] - step[:, 15]
    after = sharpened[:, 16] - sharpened[:, 15]
    assert np.all(after > before)
    assert sharpened.max() > 1.0
    assert sharpened.min() < 0.0


def test_unsharp_with_zero_amount_is_identity(image):
    assert np.allclose(unsharp_mask(image, sigma=2.0, amount=0.0), image)


@pytest.mark.parametrize(
    "call",
    [
        lambda: ideal_mask((8, 8), 0),
        lambda: butterworth_mask((8, 8), 4, order=0),
        lambda: gaussian_mask((8, 8), -1),
        lambda: ideal_mask((8, 8), 4, kind="bandpass"),
        lambda: spectrum(np.zeros((2, 2, 3))),
    ],
)
def test_invalid_arguments_raise(call):
    with pytest.raises(ValueError):
        call()
