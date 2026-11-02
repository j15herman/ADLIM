"""Per-postsynaptic-neuron dendritic geometry for `plasticity_mlp`.

Each output neuron `j` of a Linear layer (weight W, shape (out, in)) owns a 1-D
dendritic line. Its `in` incoming synapses sit at random positions
`x_{j,i} ~ U[0,1]` along that line. Barriers (compartment boundaries) warp the
line into a coordinate `u_{j,i}` such that the heterosynaptic spillover kernel
between two synapses of the SAME neuron is a plain 1-D exponential

    K_j(i,k) = exp( -|u_{j,i} - u_{j,k}| / sigma ).                    (sigma>0)

`|du|=0 -> K=1` (complete spillover, as specified); a barrier between i and k
inflates `|du|` and throttles `K` even when their raw positions `x` are close
-> compartmentalization. Spillover is therefore heterosynaptic among the inputs
that share a postsynaptic neuron, gated by distance along that neuron's dendrite
and the barriers between them.

Two barrier models, unified by `u` (selected upstream by `--plast_geometry`):

  conductance  (resistor line):
      resistivity  rho_j(x) = 1 + sum_b a_{j,b} * exp(-(x - c_{j,b})^2 / (2 w^2))
      from `n` random Gaussian bumps (centres c ~ U[0,1], amplitudes
      a ~ U[0, height], width w = barrier_width). The warped coordinate is the
      cumulative resistance
          u_{j,i} = integral_0^{x_{j,i}} rho_j(t) dt,
      evaluated on a uniform grid of `grid` points (trapezoidal cumulative
      integral) and linearly interpolated at each synapse's `x`.

  travel_cost (discrete barriers):
      barriers at random positions p_{j,b} ~ U[0,1] each of height h = height,
          u_{j,i} = x_{j,i} + sum_{b : p_{j,b} < x_{j,i}} h,
      i.e. raw distance plus the summed heights of the barriers crossed between
      two synapses. Computed exactly (the cumulative step-sum).

All buffers are fixed: rolled once from a seeded generator at construction and
re-rolled only for neurons that die. Not built for `--plast_geometry none`.
"""

import math

import torch

# sqrt(2*pi)/2: prefactor of the analytic Gaussian cumulative integral (used in
# the conductance model's grid; the verify script reproduces it via erf).
_GAUSS_HALF_NORM = math.sqrt(2.0 * math.pi) / 2.0
_EPS = 1e-12


class DendriteGeometry:
    """Fixed per-neuron dendritic lines + warped coordinates for one Linear layer.

    Args:
        out_features, in_features: the layer's weight shape (out, in).
        geometry: 'conductance' | 'travel_cost' (never 'none' -- the caller
            skips construction for 'none').
        n_barriers: bumps (conductance) / discrete barriers (travel_cost) per
            neuron.
        barrier_height: max Gaussian amplitude (conductance) / per-barrier height
            (travel_cost).
        barrier_width: Gaussian bump width `w` (conductance only).
        seed: base RNG seed (per-layer offset added by the caller).
        device, dtype: storage for the buffers.
        grid: number of grid points for the conductance cumulative integral.

    Buffers (all fixed; (out, in) unless noted):
        X  - synapse positions x_{j,i} in [0,1].
        U  - warped coordinates u_{j,i}.
        plus the per-neuron barrier parameters needed to re-roll on death.
    """

    def __init__(self, out_features, in_features, *, geometry, n_barriers,
                 barrier_height, barrier_width, seed, device, dtype, grid=1000):
        if geometry not in ('conductance', 'travel_cost'):
            raise ValueError(f"DendriteGeometry: unexpected geometry {geometry!r}")
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.geometry = geometry
        self.n_barriers = int(n_barriers)
        self.barrier_height = float(barrier_height)
        self.barrier_width = float(barrier_width)
        self.grid = int(grid)
        self.device = device
        self.dtype = dtype
        # Dedicated generator so geometry draws never perturb the global RNG
        # stream (and so re-rolls on death are reproducible).
        self._gen = torch.Generator(device='cpu').manual_seed(int(seed))

        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
        self.X = z(self.out_features, self.in_features)
        self.U = z(self.out_features, self.in_features)
        # conductance bump params / travel_cost barrier params (per neuron).
        self.C = z(self.out_features, self.n_barriers)   # centres / positions
        self.A = z(self.out_features, self.n_barriers)   # amplitudes (conductance)

        self.roll(None)   # roll all neurons once

    # ------------------------------------------------------------------
    def _rand(self, *shape):
        """U[0,1) of `shape` on the geometry generator, cast to the buffer dtype/device."""
        r = torch.rand(*shape, generator=self._gen, dtype=torch.float32)
        return r.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def roll(self, rows):
        """(Re)draw positions + barriers + warped coordinates for `rows` (a 1-D
        LongTensor / list of neuron indices), or ALL neurons if `rows is None`.
        Used at construction (all) and on neuron death (the dead rows)."""
        if rows is None:
            idx = torch.arange(self.out_features)
        else:
            idx = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
        if idx.numel() == 0:
            return
        n = idx.numel()

        X = self._rand(n, self.in_features)                       # positions
        if self.geometry == 'travel_cost':
            P = self._rand(n, self.n_barriers)                    # barrier positions
            A = torch.zeros_like(P)                               # unused; height is fixed
            U = self._travel_cost_u(X, P)
        else:                                                     # conductance
            C = self._rand(n, self.n_barriers)                    # bump centres
            A = self._rand(n, self.n_barriers) * self.barrier_height  # amplitudes
            P = C
            U = self._conductance_u(X, C, A)

        self.X[idx] = X
        self.C[idx] = P
        self.A[idx] = A
        self.U[idx] = U

    # ------------------------------------------------------------------
    # Warped coordinate builders
    # ------------------------------------------------------------------
    def _travel_cost_u(self, X, P):
        """u = x + h * (#barriers strictly left of x). Exact step-sum.
        X (n, in), P (n, n_barriers) -> U (n, in)."""
        h = self.barrier_height
        # count barriers whose position is strictly left of each synapse
        left = (P.unsqueeze(1) < X.unsqueeze(2)).sum(dim=2).to(self.dtype)  # (n, in)
        return X + h * left

    def _conductance_u(self, X, C, A):
        """u = integral_0^x rho(t) dt with rho(t)=1+sum_b a_b exp(-(t-c_b)^2/(2w^2)),
        via a uniform-grid trapezoidal cumulative integral + linear interp at X.
        X (n, in), C/A (n, n_barriers) -> U (n, in)."""
        G = self.grid
        w = max(self.barrier_width, _EPS)
        t = torch.linspace(0.0, 1.0, G, device=self.device, dtype=self.dtype)  # (G,)
        # rho on the grid: (n, G)
        diff = t.view(1, 1, G) - C.unsqueeze(2)                  # (n, n_barriers, G)
        bumps = A.unsqueeze(2) * torch.exp(-(diff * diff) / (2.0 * w * w))
        rho = 1.0 + bumps.sum(dim=1)                             # (n, G)
        # trapezoidal cumulative integral along the grid -> u on the grid
        dt = 1.0 / (G - 1)
        seg = 0.5 * (rho[:, :-1] + rho[:, 1:]) * dt              # (n, G-1)
        u_grid = torch.zeros_like(rho)
        u_grid[:, 1:] = torch.cumsum(seg, dim=1)
        # linear interpolation of u_grid at each synapse position x in [0,1]
        return self._interp_uniform(X, u_grid)

    @staticmethod
    def _interp_uniform(x, fp):
        """Linear interpolation of per-row grid values `fp` (n, G) over the uniform
        grid [0,1] at query positions `x` (n, in). Returns (n, in)."""
        G = fp.shape[1]
        pos = x.clamp(0.0, 1.0) * (G - 1)
        lo = pos.floor().long().clamp(0, G - 2)                  # (n, in)
        frac = (pos - lo.to(fp.dtype))
        f_lo = torch.gather(fp, 1, lo)
        f_hi = torch.gather(fp, 1, lo + 1)
        return f_lo + (f_hi - f_lo) * frac

    # ------------------------------------------------------------------
    # Spillover kernel and kernel averages
    # ------------------------------------------------------------------
    def kernel(self, j, sigma):
        """The exponential spillover kernel K_j(i,k)=exp(-|u_ji-u_jk|/sigma) for
        neuron `j`: an (in, in) matrix. Reference helper for tests; training uses
        `spread_binned`."""
        u = self.U[j]                                            # (in,)
        d = (u.unsqueeze(0) - u.unsqueeze(1)).abs()              # (in, in)
        return torch.exp(-d / max(float(sigma), _EPS))

    @torch.no_grad()
    def spread_exact(self, Omega, sigma):
        """Reference per-neuron normalized kernel average,

            Omega_tilde_{j,i} = sum_k K_j(i,k) Omega_{j,k} / sum_k K_j(i,k),

        computed exactly by looping neurons (O(out*in^2); for small layers /
        verification). sigma->0 => K=delta => Omega_tilde = Omega (per-synapse);
        sigma->inf, no barriers => Omega_tilde = the per-neuron row mean (uniform
        spread). Omega (out, in) -> (out, in)."""
        sigma = max(float(sigma), _EPS)
        out = self.U.shape[0]
        res = torch.empty_like(Omega)
        for j in range(out):
            u = self.U[j]
            K = torch.exp(-(u.unsqueeze(0) - u.unsqueeze(1)).abs() / sigma)
            res[j] = (K @ Omega[j]) / K.sum(dim=1).clamp_min(_EPS)
        return res

    @torch.no_grad()
    def spread_binned(self, Omega, sigma, n_bins):
        """Production O(out*n_bins) version of `spread_exact`: bin each neuron's
        synapses into `n_bins` uniform bins over its own [u_min, u_max], then
        apply the two-sided geometric (exponential) filter over the bins and
        gather the smoothed numerator/denominator back to each synapse. The
        kernel between adjacent bin centres is exp(-(bin_width)/sigma), so the
        binned smoothing reproduces the per-neuron exponential kernel up to the
        bin resolution (-> spread_exact as n_bins grows). Omega (out, in) ->
        (out, in)."""
        U = self.U
        out, inp = U.shape
        G = max(int(n_bins), 1)
        sigma = max(float(sigma), _EPS)
        umin = U.min(dim=1, keepdim=True).values                  # (out,1)
        rng = (U.max(dim=1, keepdim=True).values - umin).clamp_min(_EPS)
        b = ((U - umin) / rng * G).long().clamp(0, G - 1)         # (out,in) bin idx
        num = torch.zeros(out, G, dtype=Omega.dtype, device=Omega.device)
        cnt = torch.zeros(out, G, dtype=Omega.dtype, device=Omega.device)
        num.scatter_add_(1, b, Omega)
        cnt.scatter_add_(1, b, torch.ones_like(Omega))
        r = torch.exp(-(rng / G) / sigma)                         # (out,1) decay/bin
        snum = self._two_sided_geometric(num, r)
        scnt = self._two_sided_geometric(cnt, r).clamp_min(_EPS)
        return torch.gather(snum, 1, b) / torch.gather(scnt, 1, b)

    @staticmethod
    def _two_sided_geometric(s, r):
        """out[:, b] = sum_b' r^|b-b'| s[:, b'] (the symmetric geometric filter),
        as a per-row (out, G, G) kernel matvec. s (out, G); r (out, 1)."""
        out, G = s.shape
        idx = torch.arange(G, device=s.device, dtype=s.dtype)
        D = (idx.view(G, 1) - idx.view(1, G)).abs()               # (G, G) |b-b'|
        logr = torch.log(r.clamp_min(_EPS)).view(out, 1, 1)       # (out,1,1)
        M = torch.exp(D.unsqueeze(0) * logr)                      # (out, G, G)
        return torch.bmm(M, s.unsqueeze(2)).squeeze(2)
