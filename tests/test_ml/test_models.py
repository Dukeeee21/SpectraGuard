"""Tests de arquitectura: shapes de salida y flujo de gradientes de las
ramas espacial, de frecuencia y del modelo híbrido.

`pretrained=False` en todos los tests: verifican la arquitectura, no los
pesos descargados, así corren rápido y sin depender de red.
"""

from __future__ import annotations

import torch

from ml.models.frequency_branch import FrequencyBranch
from ml.models.hybrid_model import HybridDetector
from ml.models.spatial_branch import SpatialBranch

BATCH_SIZE = 2
IMAGE_SIZE = 224


class TestSpatialBranch:
    def test_output_shape_matches_declared_output_dim(self) -> None:
        branch = SpatialBranch(pretrained=False)
        face = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

        features = branch(face)

        assert features.shape == (BATCH_SIZE, branch.output_dim)

    def test_freeze_backbone_disables_gradients(self) -> None:
        branch = SpatialBranch(pretrained=False, freeze_backbone=True)

        assert all(not p.requires_grad for p in branch.backbone.parameters())


class TestFrequencyBranch:
    def test_output_shape_matches_declared_output_dim(self) -> None:
        branch = FrequencyBranch(output_dim=256)
        spectrum = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)

        features = branch(spectrum)

        assert features.shape == (BATCH_SIZE, 256)

    def test_accepts_input_sizes_other_than_224(self) -> None:
        branch = FrequencyBranch(output_dim=256)
        spectrum = torch.rand(BATCH_SIZE, 1, 96, 96)

        features = branch(spectrum)

        assert features.shape == (BATCH_SIZE, 256)


class TestHybridDetector:
    def test_forward_returns_one_logit_per_sample(self) -> None:
        model = HybridDetector(pretrained_spatial=False)
        face = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        spectrum = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)

        logits = model(face, spectrum)

        assert logits.shape == (BATCH_SIZE, 1)

    def test_gradients_flow_through_both_branches(self) -> None:
        model = HybridDetector(pretrained_spatial=False)
        face = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        spectrum = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)
        target = torch.randint(0, 2, (BATCH_SIZE, 1)).float()

        logits = model(face, spectrum)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()

        spatial_param = next(model.spatial_branch.backbone.parameters())
        frequency_param = next(model.frequency_branch.parameters())
        assert spatial_param.grad is not None
        assert frequency_param.grad is not None
