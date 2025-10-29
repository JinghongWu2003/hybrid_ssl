import pytest

torch = pytest.importorskip("torch")

from models.hybrid_model import HybridConfig, HybridModel


def test_hybrid_forward_shapes():
    cfg = HybridConfig(
        encoder="vit_small",
        img_size=64,
        patch_size=8,
        mask_ratio=0.75,
        projector_dim=128,
        projector_layers=2,
        temp=0.2,
    )
    model = HybridModel(cfg)
    batch = {
        "view1": torch.randn(2, 3, 64, 64),
        "view2": torch.randn(2, 3, 64, 64),
        "image": torch.randn(2, 3, 64, 64),
    }
    outputs = model(batch, alpha=0.5)
    assert outputs["z1"].shape == (2, model.projector.net[0].in_features)
    assert outputs["h1"].shape == (2, cfg.projector_dim)
    assert outputs["recon"].shape[0] == 2
    assert outputs["mask"].shape[1] == (64 // cfg.patch_size) ** 2


def test_resnet_tokens():
    cfg = HybridConfig(
        encoder="resnet18",
        img_size=224,
        patch_size=16,
        mask_ratio=0.5,
        projector_dim=128,
        projector_layers=2,
        temp=0.2,
    )
    model = HybridModel(cfg)
    batch = {
        "view1": torch.randn(1, 3, 224, 224),
        "view2": torch.randn(1, 3, 224, 224),
        "image": torch.randn(1, 3, 224, 224),
    }
    outputs = model(batch, alpha=0.5)
    assert outputs["h1"].shape[-1] == cfg.projector_dim
    assert outputs["mask"].shape[1] == outputs["recon"].shape[1]
