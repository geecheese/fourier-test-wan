"""Geometry-aware PB weak moments for one global H1 trial; validation only."""
import importlib.util
from pathlib import Path
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
import pb_benchmark as pb

_path = Path(__file__).resolve().parents[3]/'wan-dfr-fourier-release/src/methods/fourier_test_space.py'
_spec = importlib.util.spec_from_file_location('pb_existing_fourier', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def spectral_data(K):
    if K < 1:
        raise ValueError('K must be positive')
    dual = _module.FourierDualLoss(K, spatial_dim=2, domain=(2., 2.), norm='hminus1')
    args = dict(dtype=torch.float64, device=torch.device('cpu'))
    return dual.eigenvalues(**args).numpy(), dual.weights(**args).numpy()


def quadrature(n):
    """8 angular Gauss panels; radial Gauss split exactly at the circle.

    Square radius is 1/max(|cos(theta)|, |sin(theta)|); panels isolate corners.
    Returns regional volume and independent circle line rules (float64).
    """
    z, w = leggauss(n)
    theta = np.concatenate([(j+(z+1)/2)*np.pi/4 for j in range(8)])
    wt = np.tile(w*np.pi/8, 8)
    unit = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    outer = 1/np.max(np.abs(unit), axis=-1)
    points, weights = [], []
    for lo, hi in [(np.zeros_like(outer), np.full_like(outer, pb.R)),
                   (np.full_like(outer, pb.R), outer)]:
        radius = lo[:, None]+(hi-lo)[:, None]*(z+1)/2
        points.append((radius[..., None]*unit[:, None, :]).reshape(-1, 2))
        weights.append((wt[:, None]*w[None, :]*(hi-lo)[:, None]/2*radius).ravel())
    return np.concatenate(points), np.concatenate(weights), pb.R*unit, pb.R*wt


def basis(x, K):
    waves = np.arange(1, K+1)*np.pi/2
    sx, sy = [np.sin((x[:, i, None]+1)*waves) for i in range(2)]
    dx, dy = [np.cos((x[:, i, None]+1)*waves)*waves for i in range(2)]
    return sx, sy, dx, dy


def moments(value, gradient, K, rule, source_form='volume'):
    """Accept ONE global scalar trial and its gradient; H1 is caller's contract.

    r = integral eps grad(u).grad(phi) + kappa² sinh(u+us) phi
        - (F_singular + f_manufactured) phi - integral_Gamma J_s phi.
    No trial Laplacian, penalties, or strong-residual evaluation is used.
    """
    x, w, circle, wc = rule
    sx, sy, dx, dy = basis(x, K)
    def project(a, b, f):
        return a.T @ ((w*f)[:, None]*b)
    g = gradient(x)
    eps = pb.epsilon(x)
    result = project(dx, sy, eps*g[:, 0])+project(sx, dy, eps*g[:, 1])
    result += project(sx, sy, pb.kappa(x)**2*np.sinh(value(x)+pb.u_s(x)))
    result -= project(sx, sy, pb.manufactured_remainder(x))
    cx, cy, _, _ = basis(circle, K)
    result -= cx.T @ ((wc*pb.singular_flux_jump(circle))[:, None]*cy)
    if source_form == 'volume':
        result -= project(sx, sy, pb.singular_source(x))
    elif source_form == 'integrated':
        field = (eps-pb.EPS_MINUS)[:, None]*pb.grad_s(x)
        result += project(dx, sy, field[:, 0])+project(sx, dy, field[:, 1])
        result += cx.T @ ((wc*pb.source_flux_jump(circle))[:, None]*cy)
    else:
        raise ValueError('source_form must be volume or integrated')
    return result


def metrics(residual):
    _, weights = spectral_data(residual.shape[0])
    return dict(max_abs=float(np.max(np.abs(residual))),
                rms=float(np.sqrt(np.mean(residual**2))),
                L_DFR=float(np.sum(weights*residual**2)),
                L_unweighted=float(np.sum(residual**2)))
