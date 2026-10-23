"""Free-Form Deformation (FFD) 'morphing box' for the car-surface -> target-surface morph.
A lattice of cubic-B-spline control points surrounds the car. The displacement of ANY point x is  d(x) = sum_ijk B_i(u)B_j(v)B_k(w) dP_ijk .
Because d is one smooth global field (C2) the surface and the volume mesh move together: boundary layers keep their thickness and cells do not fold.
The three outermost control layers on every side of the box are fixed to zero -> d = 0 (with zero slope) on the box surface, so the far field
and the ground plane (z = 0 is the lower box face) stay untouched.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree


def _bspline_w(s):
    """uniform cubic B-spline weights for local coordinate s in [0,1): (n,4)"""
    s2 = s * s; s3 = s2 * s
    return np.stack([(1 - s) ** 3 / 6.0, (3 * s3 - 6 * s2 + 4) / 6.0, (-3 * s3 + 3 * s2 + 3 * s + 1) / 6.0, s3 / 6.0], 1)


def _bspline_dw(s):
    """derivative of the weights w.r.t. s: (n,4)"""
    s2 = s * s
    return np.stack([-(1 - s) ** 2 / 2.0, (9 * s2 - 12 * s) / 6.0, (-9 * s2 + 6 * s + 3) / 6.0, s2 / 2.0], 1)


class Lattice:
    def __init__(self, lo, hi, h):
        lo = np.asarray(lo, float); hi = np.asarray(hi, float); h = np.broadcast_to(np.asarray(h, float), (3,)).copy()
        self.n_cells = np.maximum(np.ceil((hi - lo) / h).astype(int), 4); self.h = (hi - lo) / self.n_cells; self.lo = lo; self.hi = lo + self.h * self.n_cells
        self.n = self.n_cells + 3                              # control points per axis (valid domain = n_cells cells)
        self.nc = int(np.prod(self.n))
        idx = np.indices(self.n).reshape(3, -1).T              # (nc,3)
        self.pos = self.lo + (idx - 1) * self.h                # control point positions
        edge = (idx < 3) | (idx > self.n - 4)                  # 3 outer layers each side -> d = 0 and slope 0 at the box surface
        self.free = ~edge.any(1)
        self.fid = -np.ones(self.nc, int); self.fid[self.free] = np.arange(self.free.sum()); self.nfree = int(self.free.sum())
        self._flat = lambda i, j, k: (i * self.n[1] + j) * self.n[2] + k

    def _local(self, X, deriv=None):
        s = (X - self.lo) / self.h
        inside = np.all((s >= 0) & (s < self.n_cells), axis=1)
        i0 = np.clip(np.floor(s).astype(int), 0, self.n_cells - 1); u = s - i0
        W = [_bspline_w(u[:, a]) if deriv != a else _bspline_dw(u[:, a]) / self.h[a] for a in range(3)]
        return inside, i0, W

    def basis_matrix(self, X):
        """sparse (len(X) x nfree) matrix of the basis weights (only free control points are unknowns)"""
        inside, i0, (Wx, Wy, Wz) = self._local(X)
        rows = []; cols = []; vals = []
        ar = np.arange(len(X))
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    cid = self._flat(i0[:, 0] + a, i0[:, 1] + b, i0[:, 2] + c); f = self.fid[cid]
                    w = Wx[:, a] * Wy[:, b] * Wz[:, c] * inside; m = (f >= 0) & (w != 0)
                    rows.append(ar[m]); cols.append(f[m]); vals.append(w[m])
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(X), self.nfree))

    def evaluate(self, X, dP, chunk=100000):
        """displacement at points X given control displacements dP (nfree,3) (zero outside the box)"""
        D = np.zeros((len(X), 3)); full = np.zeros((self.nc, 3)); full[self.free] = dP
        for s0 in range(0, len(X), chunk):
            Xc = X[s0:s0 + chunk]; inside, i0, (Wx, Wy, Wz) = self._local(Xc)
            if not inside.any(): continue
            m = np.where(inside)[0]; acc = np.zeros((len(m), 3))
            for a in range(4):
                for b in range(4):
                    for c in range(4):
                        cid = self._flat(i0[m, 0] + a, i0[m, 1] + b, i0[m, 2] + c)
                        acc += (Wx[m, a] * Wy[m, b] * Wz[m, c])[:, None] * full[cid]
            D[s0 + m] = acc
        return D

    def jacobian_det(self, X, dP):
        """det(I + grad d) at points X (finite folding indicator: must stay > 0)"""
        inside, i0, _ = self._local(X); full = np.zeros((self.nc, 3)); full[self.free] = dP
        G = np.zeros((len(X), 3, 3))
        for ax in range(3):
            _, _, (Wx, Wy, Wz) = self._local(X, deriv=ax)
            for a in range(4):
                for b in range(4):
                    for c in range(4):
                        cid = self._flat(np.minimum(i0[:, 0] + a, self.n[0] - 1), np.minimum(i0[:, 1] + b, self.n[1] - 1), np.minimum(i0[:, 2] + c, self.n[2] - 1))
                        G[:, :, ax] += (Wx[:, a] * Wy[:, b] * Wz[:, c] * inside)[:, None] * full[cid]
        return np.linalg.det(np.eye(3)[None] + G)

    def laplacian(self):
        """sparse smoothness operator on the free control displacements (second differences along each axis)"""
        n = self.n; rows = []; cols = []; vals = []; r = 0
        idx3 = np.indices(n).reshape(3, -1).T
        for ax in range(3):
            ok = (idx3[:, ax] >= 1) & (idx3[:, ax] <= n[ax] - 2)
            for p in idx3[ok]:
                c = self._flat(*p)
                for off, w in ((-1, 1.0), (0, -2.0), (1, 1.0)):
                    q = p.copy(); q[ax] += off; f = self.fid[self._flat(*q)]
                    if f >= 0: rows.append(r); cols.append(f); vals.append(w)
                r += 1
        return sparse.csr_matrix((vals, (rows, cols)), shape=(r, self.nfree))


def fit_ffd(P, S0, S1, T1, refine_fn, lat, iters=14, lam=0.05, nsamp=120000, verbose=True):
    """ICP-style fit: control displacements dP such that the deformed base surface points P + d(P) cover the target surface (forward: each point to its
    nearest target point, normal-aware; reverse: target samples pull their nearest deformed point so no part of the target is left uncovered)."""
    B = lat.basis_matrix(P); L = lat.laplacian()
    R1, _ = refine_fn(T1, 0.03); Y = R1.mean(1)
    if len(Y) > nsamp: Y = Y[np.random.default_rng(0).choice(len(Y), nsamp, replace=False)]
    _, tri0, _, _ = S0.closest(P)
    Q0, _, _, _ = S1.closest(P, normals=S0.tn[tri0])
    dP = np.zeros((lat.nfree, 3)); T = Q0.copy()                  # first target: normal-aware projection
    A = sparse.vstack([B, np.sqrt(lam) * L]).tocsr()
    for it in range(iters):
        rhs = np.vstack([T - P, np.zeros((L.shape[0], 3))])
        for a in range(3):
            dP[:, a] = lsqr(A, rhs[:, a], x0=dP[:, a], atol=1e-9, btol=1e-9, iter_lim=300)[0]
        X = P + B @ dP
        Qf, _, _, _ = S1.closest(X)
        _, j = cKDTree(X).query(Y); cnt = np.bincount(j, minlength=len(P)); rr = np.zeros_like(P)
        for a in range(3): rr[:, a] = np.bincount(j, weights=Y[:, a] - X[j, a], minlength=len(P))
        rr /= np.maximum(cnt, 1)[:, None]
        T = X + 0.6 * (Qf - X) + 0.6 * rr                          # attraction to the target surface from both directions
        if verbose: print("  ffd iter %2d: forward mean %.2f mm, p99 %.1f mm" % (it + 1, np.linalg.norm(Qf - X, axis=1).mean() * 1e3, np.percentile(np.linalg.norm(Qf - X, axis=1), 99) * 1e3))
    return dP


class HierFFD:
    """two (or more) FFD stages applied one after the other: x -> x + d1(x) -> + d2(x + d1(x)).  A coarse, very smooth first box removes the large
    shape difference without folding; a finer second box removes the small remaining error. Each stage is fold-free, so the composition is too."""
    def __init__(self, stages):
        self.stages = stages                                   # list of (Lattice, dP)

    def displacement(self, X, chunk=100000):
        Y = X.copy()
        for lat, dP in self.stages: Y = Y + lat.evaluate(Y, dP, chunk)
        return Y - X

    def jacobian_det(self, X):
        """det of the composed map: product of the stage Jacobians, each evaluated at the point where that stage acts"""
        Y = X.copy(); J = np.ones(len(X))
        for lat, dP in self.stages:
            J *= lat.jacobian_det(Y, dP); Y = Y + lat.evaluate(Y, dP)
        return J


def fit_hier(P, S0, S1, T1, refine_fn, lat1, lat2, lam1=3.0, lam2=0.5, iters1=10, iters2=8, verbose=True):
    dP1 = fit_ffd(P, S0, S1, T1, refine_fn, lat1, iters=iters1, lam=lam1, verbose=False)
    X1 = P + lat1.evaluate(P, dP1)
    if verbose: print("  stage 1 done")
    # stage 2: the base surface is now X1; S0 is only used for the normal-aware first guess, so pass a surface built from the deformed points' own triangles is not needed:
    # use plain closest-point targets (X1 is already close to the target, so the nearest point is the right one)
    class _Id:                                                  # minimal stand-in for S0: returns triangle 0 with the normal of the closest target triangle
        pass
    dP2 = fit_ffd_close(X1, S1, T1, refine_fn, lat2, iters=iters2, lam=lam2, verbose=False)
    return HierFFD([(lat1, dP1), (lat2, dP2)])


def fit_ffd_close(P, S1, T1, refine_fn, lat, iters=8, lam=0.5, nsamp=120000, verbose=False):
    """second-stage fit: P is already near the target; targets are plain closest points + reverse pull"""
    B = lat.basis_matrix(P); L = lat.laplacian()
    R1, _ = refine_fn(T1, 0.03); Y = R1.mean(1)
    if len(Y) > nsamp: Y = Y[np.random.default_rng(1).choice(len(Y), nsamp, replace=False)]
    dP = np.zeros((lat.nfree, 3)); A = sparse.vstack([B, np.sqrt(lam) * L]).tocsr()
    X = P.copy()
    for it in range(iters):
        Qf, _, _, _ = S1.closest(X)
        _, j = cKDTree(X).query(Y); cnt = np.bincount(j, minlength=len(P)); rr = np.zeros_like(P)
        for a in range(3): rr[:, a] = np.bincount(j, weights=Y[:, a] - X[j, a], minlength=len(P))
        rr /= np.maximum(cnt, 1)[:, None]
        T = X + 0.6 * (Qf - X) + 0.6 * rr
        rhs = np.vstack([T - P, np.zeros((L.shape[0], 3))])
        for a in range(3): dP[:, a] = lsqr(A, rhs[:, a], x0=dP[:, a], atol=1e-9, btol=1e-9, iter_lim=200)[0]
        X = P + B @ dP
    return dP
