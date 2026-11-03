"""Prediction on new photos: preparation as for the training images, fit / predict, save / load."""
import numpy as np
import torch
from PIL import Image
from torch import nn

from artguide.inference import Predictor, load_photo
from artguide.models.baselines import logistic_regression

EXIF_ORIENTATION = 0x0112  # tag 274; value 6 = "turn 90 degrees clockwise to display"


def fake_backbone() -> nn.Module:
    """Stands in for the 45 MB ResNet: 3 x 4 x 4 average colours = 48 features."""
    return nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten())


def test_a_photo_comes_out_square_and_uint8(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (300, 200), (10, 20, 30)).save(path)
    x = load_photo(path, size=64)
    assert x.shape == (3, 64, 64) and x.dtype == torch.uint8
    assert x[:, 32, 32].tolist() == [10, 20, 30]


def test_the_exif_rotation_of_a_phone_photo_is_applied(tmp_path):
    im = Image.new("RGB", (300, 200), (0, 0, 255))
    im.paste((255, 0, 0), (0, 0, 150, 200))  # left half red, right half blue
    exif = Image.Exif()
    exif[EXIF_ORIENTATION] = 6
    path = tmp_path / "phone.jpg"
    im.save(path, exif=exif, quality=95)
    x = load_photo(path, size=64)
    # turned clockwise, the left half becomes the top half
    assert x[0, 8, 32] > 200 and x[2, 8, 32] < 50  # top: red
    assert x[2, 56, 32] > 200 and x[0, 56, 32] < 50  # bottom: blue


def toy_images() -> tuple[torch.Tensor, np.ndarray]:
    """Class 0 = dark images, class 1 = bright images."""
    gen = torch.Generator().manual_seed(0)
    dark = torch.randint(0, 80, (10, 3, 32, 32), dtype=torch.uint8, generator=gen)
    bright = torch.randint(170, 256, (10, 3, 32, 32), dtype=torch.uint8, generator=gen)
    return torch.cat([dark, bright]), np.array([0] * 10 + [1] * 10)


def test_fit_then_predict():
    images, labels = toy_images()
    predictor = Predictor(logistic_regression(1.0), ["dark", "bright"], backbone=fake_backbone())
    probs = predictor.fit(images, labels).probabilities(images)
    assert probs.shape == (20, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert (predictor.predict(images) == labels).all()


def test_save_and_load_give_the_same_answers(tmp_path):
    images, labels = toy_images()
    predictor = Predictor(logistic_regression(1.0), ["dark", "bright"], {"C": 1.0}, backbone=fake_backbone())
    predictor.fit(images, labels).save(tmp_path / "model.pkl")
    loaded = Predictor.load(tmp_path / "model.pkl", backbone=fake_backbone())
    assert loaded.classes == ["dark", "bright"] and loaded.info["C"] == 1.0 and "weights" in loaded.info
    np.testing.assert_array_equal(loaded.probabilities(images), predictor.probabilities(images))
