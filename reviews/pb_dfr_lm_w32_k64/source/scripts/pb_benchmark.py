"""Provisional implementation benchmark; no advisor physical data or Dirac terms."""
import numpy as np
from numpy.polynomial import Polynomial

R, EPS_MINUS, EPS_PLUS, KAPPA, SIGMA, Q, A = .5, 2., 80., 1., .08, 1., .2
DOMAIN = ((-1., 1.), (-1., 1.))
P = A * Polynomial([-R**2, 1])**2 * Polynomial([1, -1])**2


def epsilon(x):
    return np.where(np.sum(x*x, axis=-1) < R*R, EPS_MINUS, EPS_PLUS)


def kappa(x):
    return np.full(x.shape[:-1], KAPPA)


def u_r_exact(x):
    return P(np.sum(x*x, axis=-1))


def grad_exact(x):
    return 2*x*P.deriv()(np.sum(x*x, axis=-1))[..., None]


def lap_exact(x):
    t = np.sum(x*x, axis=-1)
    return 4*(P.deriv()(t) + t*P.deriv(2)(t))


def u_s(x, eps=None):
    eps = epsilon(x) if eps is None else eps
    return Q/(4*np.pi*eps*np.sqrt(np.sum(x*x, axis=-1)+SIGMA**2))


def grad_s(x, eps=None):
    eps = epsilon(x) if eps is None else eps
    c = -Q/(4*np.pi*eps)*(np.sum(x*x, axis=-1)+SIGMA**2)**(-1.5)
    return x*c[..., None]


def singular_source(x):
    t, eps = np.sum(x*x, axis=-1), epsilon(x)
    lap = Q/(4*np.pi*eps)*(t-2*SIGMA**2)/(t+SIGMA**2)**2.5
    return (eps-EPS_MINUS)*lap


def forcing(x):
    return -epsilon(x)*lap_exact(x)+kappa(x)**2*np.sinh(u_r_exact(x)+u_s(x))


def manufactured_remainder(x):
    return forcing(x)-singular_source(x)


def u_exact(x):
    return u_r_exact(x)+u_s(x)


def boundary_data(x):
    """Total g on the square; regular trace is g-u_s, generally nonzero."""
    return u_exact(x)


def singular_flux_jump(x):
    n = x/R
    return np.sum((EPS_PLUS*grad_s(x, EPS_PLUS)-EPS_MINUS*grad_s(x, EPS_MINUS))*n, axis=-1)


def source_flux_jump(x):
    return np.sum((EPS_PLUS-EPS_MINUS)*grad_s(x, EPS_PLUS)*x/R, axis=-1)


def perturbation(x):
    s = np.sin(np.pi*(x+1)/2)
    return np.prod((1-x*x)*s, axis=-1)


def grad_perturbation(x):
    s, c = np.sin(np.pi*(x+1)/2), np.cos(np.pi*(x+1)/2)
    f = (1-x*x)*s
    d = -2*x*s+(1-x*x)*np.pi/2*c
    return np.stack((d[:, 0]*f[:, 1], f[:, 0]*d[:, 1]), axis=-1)


def trial(delta=0.):
    """One global scalar callable and its analytic gradient, never region trials."""
    return (lambda x: u_r_exact(x)+delta*perturbation(x),
            lambda x: grad_exact(x)+delta*grad_perturbation(x))
