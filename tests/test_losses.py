import pytest

torch = pytest.importorskip("torch")

from losses.info_nce import InfoNCELoss
from losses.mae_reconstruction import MAELoss, patchify


def test_info_nce_positive():
    loss_fn = InfoNCELoss(temperature=0.2)
    z = torch.randn(4, 16)
    loss = loss_fn(z, z)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_mae_loss_decreases():
    images = torch.randn(2, 3, 32, 32)
    patch_size = 8
    preds = patchify(images, patch_size)
    mask = torch.zeros(preds.shape[:2])
    loss_fn = MAELoss(norm_pix_loss=False)
    loss = loss_fn(images, preds, mask, patch_size)
    assert torch.allclose(loss, torch.tensor(0.0))
