import pytest

torch = pytest.importorskip("torch")

from losses.mae_reconstruction import MAELoss, patchify
from models.mae_decoder import MAEDecoder
from utils.mask import random_masking


@pytest.mark.parametrize("mask_ratio", [0.5, 0.75])
def test_random_masking_shapes(mask_ratio: float) -> None:
    tokens = torch.randn(2, 64, 768)
    visible, mask, ids_restore = random_masking(tokens, mask_ratio)
    len_keep = int(64 * (1 - mask_ratio))
    assert visible.shape == (2, len_keep, 768)
    assert mask.shape == (2, 64)
    assert ids_restore.shape == (2, 64)
    masked_tokens = mask.sum(dim=1)
    expected_masked = torch.full_like(masked_tokens, fill_value=64 * mask_ratio)
    assert torch.allclose(masked_tokens, expected_masked, atol=1e-5)


def test_decoder_output_shape() -> None:
    decoder = MAEDecoder(
        encoder_dim=768,
        patch_size=8,
        img_size=64,
        decoder_dim=256,
        depth=2,
        num_heads=4,
    )
    tokens = torch.randn(2, 64, 768)
    visible, mask, ids_restore = random_masking(tokens, 0.75)
    preds = decoder(visible, mask, ids_restore)
    assert preds.shape == (2, 64, 8 * 8 * 3)


def test_mae_loss_mask_zero() -> None:
    images = torch.randn(2, 3, 32, 32)
    patch_size = 8
    preds = patchify(images, patch_size)
    mask = torch.zeros(preds.shape[:2])
    loss_fn = MAELoss(norm_pix_loss=False)
    loss = loss_fn(images, preds, mask, patch_size)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_decoder_training_reduces_loss() -> None:
    torch.manual_seed(0)
    images = torch.randn(1, 3, 64, 64)
    patch_size = 8
    patch_tokens = patchify(images, patch_size)
    encoder_dim = patch_tokens.size(-1)
    decoder = MAEDecoder(
        encoder_dim=encoder_dim,
        patch_size=patch_size,
        img_size=64,
        decoder_dim=256,
        depth=2,
        num_heads=4,
    )
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    loss_fn = MAELoss(norm_pix_loss=False)

    losses = []
    for _ in range(20):
        torch.manual_seed(42)  # ensure consistent masking for comparison
        visible, mask, ids_restore = random_masking(patch_tokens, 0.75)
        preds = decoder(visible, mask, ids_restore)
        loss = loss_fn(images, preds, mask, patch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
