import pytest

torch = pytest.importorskip("torch")

from utils.mask import num_patches, sample_mask


def test_sample_mask_shape_ratio():
    batch = 2
    img_size = 64
    patch_size = 8
    patches = num_patches(img_size, patch_size)
    mask = sample_mask(batch, patches, mask_ratio=0.75, device=torch.device("cpu"))
    assert mask.shape == (batch, patches)
    visible = (~mask).sum(dim=1)
    assert torch.all(visible == torch.tensor(int(patches * 0.25)))


def test_sample_mask_visible_tokens():
    batch = 1
    patches = 196
    mask = sample_mask(batch, patches, mask_ratio=0.5, device=torch.device("cpu"))
    assert mask.sum().item() == patches * 0.5
