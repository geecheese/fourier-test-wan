#!/usr/bin/env python3
"""Unit tests for :mod:`fourier_test_space`."""

from __future__ import annotations

import math
import unittest

import torch

from src.methods.fourier_test_space import FourierDualLoss


class FourierDualLossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)

    def test_dirichlet_transform_is_truncated_and_weighted(self) -> None:
        loss = FourierDualLoss(modes=4, spatial_dim=1, boundary="dirichlet", domain=math.pi)
        grid = 32
        # A smooth sampled residual whose first sine coefficient is nonzero.
        x = torch.arange(1, grid + 1, dtype=torch.float64) * math.pi / (grid + 1)
        residual = torch.sin(2.0 * x)
        coefficients = loss.transform(residual)
        self.assertEqual(tuple(coefficients.shape), (4,))
        self.assertGreater(float(coefficients[1].abs()), 1.0)
        self.assertLess(float(coefficients[[0, 2, 3]].abs().max()), 1.0e-12)
        expected = coefficients.square() * (1.0 / 5.0)
        self.assertTrue(torch.allclose(loss.energy(residual), expected))
        self.assertAlmostEqual(float(loss(residual)), float(expected.sum()), places=12)

    def test_neumann_dct_includes_constant_mode(self) -> None:
        loss = FourierDualLoss(modes=3, spatial_dim=1, boundary="neumann", domain=1.0)
        residual = torch.ones(20, dtype=torch.float64)
        coefficients = loss.transform(residual)
        self.assertGreater(float(coefficients[0].abs()), 0.0)
        self.assertLess(float(coefficients[1:].abs().max()), 1.0e-12)
        self.assertTrue(torch.isfinite(loss(residual)))

    def test_two_dimensional_mixed_boundaries_and_batch(self) -> None:
        loss = FourierDualLoss(
            modes=(3, 4), spatial_dim=2,
            boundary=("dirichlet", "neumann"), domain=(math.pi, 2.0),
        )
        residual = torch.randn(2, 16, 18, dtype=torch.float64, requires_grad=True)
        coefficients = loss.transform(residual)
        self.assertEqual(tuple(coefficients.shape), (2, 3, 4))
        value = loss(residual, reduction="none")
        self.assertEqual(tuple(value.shape), (2,))
        value.sum().backward()
        self.assertIsNotNone(residual.grad)
        self.assertTrue(torch.isfinite(residual.grad).all())

    def test_validation_and_reductions(self) -> None:
        with self.assertRaises(ValueError):
            FourierDualLoss(modes=2, spatial_dim=3)
        with self.assertRaises(ValueError):
            FourierDualLoss(modes=(2, 3), spatial_dim=1)
        loss = FourierDualLoss(modes=2)
        residual = torch.zeros(8)
        self.assertEqual(float(loss(residual)), 0.0)
        self.assertEqual(float(loss(residual, reduction="mean")), 0.0)
        with self.assertRaises(ValueError):
            loss(residual, reduction="bad")


if __name__ == "__main__":
    unittest.main()
