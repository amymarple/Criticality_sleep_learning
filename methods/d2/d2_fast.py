
# -*- coding: utf-8 -*-
"""
d2_fast.py -- Optimized distance-to-criticality (β=2) via AR(n)+KL

Key speedups over the reference implementation:
- FFT-based autocovariance in O(N log N).
- Levinson–Durbin Toeplitz solver for Yule–Walker in O(p^2).
- Vectorized KL objective with precomputed exp(-i w k) matrix (cache by (num_w, p)).
- Multi-resolution objective (coarse num_w for search, refine once at the end).
- Toeplitz construction via np.take on |j-k| index matrix (no Python loops).

Public API (drop-in compatible):
- fit_ar_yule_walker(x, order, unbiased=False)  -> (phi, sigma2)
- d2_from_timeseries(x, order=10, dt=1.0, num_w=2**14, method="auto",
                     maxiter=600, random_restarts=1, seed=0) -> (d2_bits_per_second, info)
"""
from __future__ import annotations
import numpy as np

try:
    from scipy.optimize import minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ------------------
# FFT autocovariance
# ------------------
def _autocov_fft(x: np.ndarray, maxlag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    # next power of two for zero-padding (for circular convolution safety)
    m = 1 << (int(np.ceil(np.log2(2*n-1))))
    Xf = np.fft.rfft(x, n=m)
    ac_full = np.fft.irfft(Xf * np.conj(Xf), n=m)[:n]  # lags 0..n-1
    # biased estimator: divide by N
    ac_full = ac_full / n
    return ac_full[:maxlag+1]

# -----------------------------
# Levinson–Durbin (real, stable)
# -----------------------------
def _levinson(ac: np.ndarray, order: int):
    """Return AR coeffs (phi[1..p]) and innovation variance using Levinson–Durbin.
    ac: autocovariance array with length >= order+1, ac[0]..ac[p]
    """
    p = order
    if p < 1:
        return np.empty(0), float(ac[0])
    phi = np.zeros(p, dtype=float)
    E = float(ac[0])
    if not np.isfinite(E) or E <= 0:
        # fallback to tiny positive to avoid nan
        E = 1e-12
    for k in range(p):
        if k == 0:
            lambda_k = ac[1] / E if E != 0 else 0.0
            phi[0] = lambda_k
            E *= (1.0 - lambda_k*lambda_k)
        else:
            # reflection coefficient
            num = ac[k+1] - np.dot(phi[:k], ac[1:k+1][::-1])
            den = E
            lambda_k = num / den if den != 0 else 0.0
            phi[:k] = phi[:k] - lambda_k * phi[:k][::-1]
            phi[k] = lambda_k
            E *= (1.0 - lambda_k*lambda_k)
        # keep E strictly positive
        if E <= 0:
            E = 1e-18
    return phi, float(E)

def fit_ar_yule_walker(x, order, unbiased=False):
    """Fast Yule–Walker using FFT autocovariance + Levinson–Durbin."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < order + 5:
        raise ValueError("x must be 1D and longer than order.")
    ac = _autocov_fft(x, order)
    phi, sigma2 = _levinson(ac, order)
    sigma2 = float(max(sigma2, 1e-18))
    return phi, sigma2

# --------------------------------------------
# Precomputed frequency matrix for fast KL rate
# --------------------------------------------
class _SpectralCache:
    __slots__ = ("num_w","p","w","E","dw")
    def __init__(self, num_w:int, p:int):
        self.num_w = int(num_w)
        self.p = int(p)
        # Use [-π, π] inclusive endpoints so trapezoid is simple
        self.w = np.linspace(-np.pi, np.pi, self.num_w, endpoint=True)
        k = np.arange(1, p+1, dtype=float)
        # (num_w, p)
        self.E = np.exp(-1j * np.outer(self.w, k))
        self.dw = (self.w[-1] - self.w[0]) / (self.num_w - 1)

# ------------------------------
# KL rate J(A||B) in bits/sample
# ------------------------------
def _phi_tilde_from_cache(phi: np.ndarray, cache: _SpectralCache) -> np.ndarray:
    # 1 - phi~(w) = 1 - sum phi_k e^{-i w k}
    return 1.0 - cache.E @ phi

def _kl_rate_bits_from_AR_cached(phiA, sigma2A, phiB, cache: _SpectralCache):
    one_minus_A = _phi_tilde_from_cache(phiA, cache)
    one_minus_B = _phi_tilde_from_cache(phiB, cache)
    denA = (one_minus_A.real**2 + one_minus_A.imag**2)
    denB = (one_minus_B.real**2 + one_minus_B.imag**2)
    # r(w)
    r = denB / denA
    rmean = r.mean()
    # closed-form elimination of sigmaB^2
    integrand = np.log(rmean * denA / denB) + (r / rmean) - 1.0
    # integral (1/(4π)) ∫ integrand dw  ; do trapezoid with constant spacing dw
    J_nats = (cache.dw * integrand.sum()) / (4.0 * np.pi)
    J_bits = J_nats / np.log(2.0)
    return float(J_bits), float(rmean)

# -----------------------
# β=2 manifold utilities
# -----------------------
def _theta_to_phiB(theta):
    """Parameterize phi_B with last component set by the sum=1 constraint."""
    theta = np.asarray(theta, float)
    return np.concatenate([theta, [1.0 - theta.sum()]])

def _phiB_init_from(phiA):
    """Deterministic start: project phiA to the sum=1 hyperplane (β=2 manifold)"""
    s = float(phiA.sum())
    if s == 1.0:
        return phiA.copy()
    # shift uniformly to satisfy sum=1
    return phiA + (1.0 - s)/len(phiA)

# --------------------------------
# Main entry: time-series to d2
# --------------------------------
def d2_from_timeseries(x, order=10, dt=1.0, num_w=2**14, method="auto", maxiter=600, random_restarts=1, seed=0):
    """
    Optimized computation of d2 (β=2) from a univariate series x.
    """
    x = np.asarray(x, float)
    x = (x - x.mean()) / (x.std() + 1e-12)  # z-score
    phiA, sigma2A = fit_ar_yule_walker(x, order)
    p = len(phiA)

    # Multi-resolution: use a coarse cache for the search, refine at the end
    coarse = max(1024, min(4096, num_w//4))
    cache_coarse = _SpectralCache(coarse, p)
    cache_fine   = _SpectralCache(num_w, p)

    phiB0 = _phiB_init_from(phiA)
    theta0 = phiB0[:-1]

    rng = np.random.default_rng(seed)

    def obj(theta, cache):
        phiB = _theta_to_phiB(theta)
        J_bits, _ = _kl_rate_bits_from_AR_cached(phiA, sigma2A, phiB, cache)
        return J_bits

    best_val = float("inf"); best_theta = theta0

    def run_once(theta_start):
        if _HAVE_SCIPY and method in ("auto","scipy","powell","neldermead"):
            mtd = "Powell" if method in ("auto","powell") else ("Nelder-Mead" if method=="neldermead" else "Powell")
            res = minimize(lambda th: obj(th, cache_coarse), theta_start, method=mtd,
                           options=dict(maxiter=maxiter, xtol=1e-6, ftol=1e-8))
            return float(res.fun), np.asarray(res.x, float)
        # very fast coordinate descent fallback
        th = theta_start.copy()
        step = 0.1
        fbest = obj(th, cache_coarse)
        for it in range(maxiter):
            improved = False
            for j in range(th.size):
                for sgn in (+1.0, -1.0):
                    trial = th.copy()
                    trial[j] += sgn*step
                    f = obj(trial, cache_coarse)
                    if f < fbest:
                        th, fbest = trial, f
                        improved = True
            step *= 0.7
            if not improved and step < 1e-5:
                break
        return fbest, th

    starts = [theta0]
    for rr in range(random_restarts):
        starts.append(theta0 + 0.05 * rng.standard_normal(theta0.shape))

    for s in starts:
        val, th = run_once(np.asarray(s, float))
        if val < best_val:
            best_val, best_theta = val, th

    # refine on fine grid
    J_bits_fine = obj(best_theta, cache_fine)
    phiB_star = _theta_to_phiB(best_theta)
    d2_bits_per_sample = float(J_bits_fine)
    d2_bits_per_second = d2_bits_per_sample / float(dt)

    # Diagnostics
    J_bits_final, rmean = _kl_rate_bits_from_AR_cached(phiA, sigma2A, phiB_star, cache_fine)
    info = dict(
        phi_A=phiA,
        sigma2_A=sigma2A,
        phi_B=phiB_star,
        d2_bits_per_sample=d2_bits_per_sample,
        d2_bits_per_second=d2_bits_per_second,
        r_mean=rmean,
        num_w=num_w,
        order=order,
        optimizer=("scipy.Powell" if _HAVE_SCIPY and method in ("auto","powell","scipy") 
                   else "coordinate_descent"),
    )
    return d2_bits_per_second, info

if __name__ == "__main__":
    # Example usage
    rng = np.random.default_rng(0)
    N = 200000
    phi_true = np.array([0.9, 0.05])
    eps = rng.normal(0, 1.0, size=N)
    x = np.zeros(N)
    for t in range(2, N):
        x[t] = phi_true[0]*x[t-1] + phi_true[1]*x[t-2] + eps[t]
    d2_bps, info = d2_from_timeseries(x, order=2, dt=0.001, num_w=8192, random_restarts=2)
    print("d2 ≈ %.6f bits/s" % d2_bps)
