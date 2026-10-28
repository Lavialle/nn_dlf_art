"""The MLP transforms: fixed without augmentation, random but same shape with it."""
import torch

from artguide.data.transforms import mlp_transform

MEAN, STD = [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]


def fake_image() -> torch.Tensor:
    return torch.randint(0, 256, (3, 128, 128), dtype=torch.uint8)


def test_without_augmentation_the_output_is_fixed():
    image, transform = fake_image(), mlp_transform(MEAN, STD, size=64)
    out = transform(image)
    assert out.shape == (3, 64, 64) and out.dtype == torch.float32
    assert torch.equal(out, transform(image))


def test_augmentation_changes_the_image_but_not_the_shape():
    torch.manual_seed(0)
    image, transform = fake_image(), mlp_transform(MEAN, STD, size=64, augment=True)
    outs = [transform(image) for _ in range(5)]
    assert all(o.shape == (3, 64, 64) for o in outs)
    assert not all(torch.equal(outs[0], o) for o in outs[1:])  # a new random crop / flip each time


def test_a_flip_alone_keeps_the_colours():
    image = fake_image()
    flipped = torch.flip(image, dims=[2])  # what RandomHorizontalFlip does
    assert torch.equal(image.float().mean(dim=(1, 2)), flipped.float().mean(dim=(1, 2)))
