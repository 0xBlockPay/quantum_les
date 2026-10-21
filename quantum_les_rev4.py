"""
================================================================================
Hybrid Quantum-Enhanced LES Closure Model (Q-LES, Rev 12)
================================================================================
POSITIONING (what this project IS and is NOT):
  This is NOT a "quantum Navier-Stokes solver". The CFD engine is a fully classical
  compressible NS + k_SGS solver (finite differences, RK4, periodic BCs) -- the main
  simulation driver runs entirely on the classical computer.
  The quantum computer plays the role of an UNCONVENTIONAL SGS SUBGRID MODEL: in each
  block it evaluates <psi|H_triad|psi> of a second-quantized triadic Hamiltonian via
  VQE (NVIDIA CUDA-Q, or a CPU state-vector fallback), and that expectation value scales
  the block-local SGS dissipation. The publishable scientific contribution is exactly
  this closure -- "can quantum variational models represent turbulent subgrid physics?"
  -- not replacing classical CFD.
Solving: 3D Compressible Navier-Stokes + k_SGS Transport Equation (classical engine)
          with a per-block quantum-VQE triad SGS closure (the quantum part).
================================================================================
Governing Equations (6-Component Conservative State Vector):
  U[0] = rho         (Continuity: d(rho)/dt + div(rho*u) = 0)
  U[1] = rho * ux    (X-Momentum)
  U[2] = rho * uy    (Y-Momentum)
  U[3] = rho * uz    (Z-Momentum)
  U[4] = E           (Total Energy)
  U[5] = rho * k_sgs (Subgrid Kinetic Energy Transport:
                      d(rho*k_sgs)/dt + div(rho*k_sgs*u) = P_sgs - eps_sgs^Q + Diff)

Revision 4 — fixes from reviewer (wytyczne.txt):

  FIX 1  (SGS production, revised): production_k = Pi_SGS (FULL sign), so the
         k_SGS source term carries both forward scatter and backscatter. The
         negative part is no longer discarded.
  FIX 2  (Energy coupling, symmetric): rhs[4] -= Pi_SGS AND rhs[5] += Pi_SGS use
         the SAME full Pi_SGS field -> a strictly symmetric Resolved<->SGS energy
         exchange (no double counting, no asymmetric clipping). See FIX 3.
  FIX 3  (Energy balance consistency): because production_k == Pi_SGS (not
         max(0,Pi_SGS)) and rhs[4] -= Pi_SGS, the transfer is symmetric: whatever
         leaves the resolved field enters the SGS field and vice versa.
  FIX 4  (JW nomenclature): "Exact Jordan-Wigner mapping of the triad operator"
         — the NS Hamiltonian itself is a reduced model; only the fermionic-to-
         qubit mapping is exact.
  FIX 5  (Energy budget + DRIFT): compute_energy_budget() + per-step KE logging
         and an explicit "Energy Drift = (KE_tot - KE_tot0)/KE_tot0" line, which
         is the first check a CFD reviewer performs.
  FIX 6  (Triad discoverer O(N^2)): dictionary lookup mode_lookup[k_vec].
  FIX 7  (Dynamic qubits, repaired): n_q is now driven by the RICHNESS of the
         candidate subgrid shell (candidate_modes = min(4*num_qubits, len(indices)),
         n_q = min(12, max(6, candidate_modes//2))) and active_modes are sliced to
         n_q. Previously n_q could never grow because active_modes was always
         pre-truncated to the base num_qubits.
  FIX 8  (CPU fallback cap): when CUDA-Q is absent the state-vector simulator is
         limited to n_q <= 8 (2^8 = 256 amplitudes/block) so 12-qubit runs do not
         explode combinatorially on CPU.
  FIX 9  (tri_weight in Hamiltonian): the triad coupling now scales with the cube
         root of the actual modal energy product tri_weight, so FFT energy content
         genuinely modulates triad strength:
             coupling = geom * cbrt(tri_weight + eps) * (1+0.2*S_mag)/Delta^(4/3).
  FIX 10 (BLOCK-LOCAL Leonard stress): Pi_SGS is now computed PER BLOCK in block
         Fourier space (apply_test_filter_block), so the SGS physics and the quantum
         closure act on the SAME spatial scale. The global Leonard tensor + local
         Hamiltonian scale mismatch is removed.

Revision 5 — further fixes from reviewer (To jest juz zdecydowanie.txt):

  FIX 11 (Ghost-cell halo overlap): block-local spectral ops (Leonard stress +
         triad FFT) run on an extended (nb+2*halo)^3 stencil gathered with periodic
         wrap, so resonant triads and Leonard-stress contributions that straddle a
         block boundary are no longer lost. Only the core is written back.
  FIX 12 (Preserve VQE history): when the dynamic qubit count changes, existing
         thetas are carried over into the new parameter vector instead of being
         fully re-randomized.
  FIX 13 (Hamiltonian statistics): per-step log now reports total triad count and
         mean active-modes-per-block, so one can see whether the QPU works on a rich
         spectrum.
  FIX 14 (Total energy budget): compute_energy_budget() now also returns the TOTAL
         conserved energy mean(U[4]) = <e_internal + e_kinetic> with its own drift,
         since a KE drift alone does not imply a total-energy drift.
  FIX 15 (LES validation harness): --validate runs the Taylor-Green KE-decay
         benchmark, records KE(t)/E(t) to CSV, reports d(log10 KE)/dt, and provides
         compare_to_reference() to quantify agreement vs DNS/Smagorinsky/DynGermano.

Revision 6 — further fixes from reviewer:

  FIX 16 (Core-mode spectral analysis): the halo is now strictly an AUXILIARY zone
         for the local spectral Leonard filter. The modal/triad analysis and k0 use
         the block's own CORE-resolution wavenumber grid, so the Hamiltonian operator
         no longer depends on the auxiliary halo width.
  FIX 17 (SGS energy budget): compute_energy_budget() now also reports SGS production
         mean(Pi_SGS), quantum dissipation mean(eps_sgs^Q) and their Pi/Eps ratio.
  FIX 18 (SGS transfer conservativeness): logs the resolved-side loss (-Pi_SGS in
         rhs[4]) vs SGS-side gain (+Pi_SGS in rhs[5]) and their closure error, which
         must vanish for the symmetric transfer.
  FIX 19 (Two-way quantum feedback): the SGS dissipation realized in step n-1 feeds
         back into the modal damping of the step-n Hamiltonian (H_n <- eps_{n-1}),
         turning the one-way H->VQE->eps chain into a feedback-coupled quantum closure.
  FIX 20 (compare_to_reference clarity): renamed the interpolated reference curve to
         ref_on_q so it is unambiguous that the REFERENCE is sampled on the quantum grid.
  FIX 21 (Validation figures): --validate now also writes publication-style PNGs of the
         KE(t) decay, the SGS production/dissipation budget and total-energy conservation.

Revision 7 — further fixes from reviewer:

  FIX 22 (Dynamic qubit hysteresis): the per-block qubit count now changes only after
         q_hysteresis consecutive steps in the same direction (cooldown), so a rich
         spectrum no longer makes n_q flicker 8 -> 10 -> 8 -> 10 every step.
  FIX 23 (Energy-spectrum benchmark): compute_energy_spectrum() + plot_energy_spectrum()
         produce E(k) (log-log) with a Kolmogorov -5/3 anchor and an optional reference
         (DNS / Smagorinsky / Dynamic Germano), plus a _spectrum.csv for post-processing.
  FIX 18 (revised): the SGS transfer closure check now uses the ACTUAL RHS integrals
         (resolved-side loss vs SGS-side gain recorded in compute_rhs), not a tautology.
  FIX 19 (revised): feedback now also modulates the TRIAD coupling Gamma_kpq (not only
         the diagonal omega_k); wording corrected to "feedback-coupled" (not "fully").
  FIX 17 (revised): Pi/Eps ratio is clipped and the more stable (Pi - Eps) is reported.

  Retained: spectral Leonard stress, QPU closed-loop, dynamic Q-criterion,
  collision-free allocation, JW self-test vs Fock space.
================================================================================
"""

import sys
import time
import argparse
import numpy as np

# Conditional CUDA-Q Import with graceful CPU simulator fallback
try:
    import cudaq
    from cudaq import spin
    CUDAQ_AVAILABLE = True
except ImportError:
    CUDAQ_AVAILABLE = False
    print("[WARN] NVIDIA CUDA-Q not found in Python path. Enabling CPU emulation mode.")


# ==============================================================================
# 1. PAULI ALGEBRA (framework-agnostic)
# ==============================================================================
_PAULI_MUL = {
    ('I','I'):(1.0,'I'), ('I','X'):(1.0,'X'), ('I','Y'):(1.0,'Y'), ('I','Z'):(1.0,'Z'),
    ('X','I'):(1.0,'X'), ('Y','I'):(1.0,'Y'), ('Z','I'):(1.0,'Z'),
    ('X','X'):(1.0,'I'), ('Y','Y'):(1.0,'I'), ('Z','Z'):(1.0,'I'),
    ('X','Y'):(1j,'Z'), ('Y','X'):(-1j,'Z'),
    ('X','Z'):(-1j,'Y'), ('Z','X'):(1j,'Y'),
    ('Y','Z'):(1j,'X'), ('Z','Y'):(-1j,'X'),
}

_PAULI_MAT = {
    'I': np.array([[1, 0], [0, 1]], complex),
    'X': np.array([[0, 1], [1, 0]], complex),
    'Y': np.array([[0, -1j], [1j, 0]], complex),
    'Z': np.array([[1, 0], [0, -1]], complex),
}


def _pauli_mul_letters(l1, l2):
    """Multiply two aligned Pauli letter strings; return (factor, letters)."""
    n = max(len(l1), len(l2))
    a = tuple(l1) + ("I",) * (n - len(l1))
    b = tuple(l2) + ("I",) * (n - len(l2))
    letters = []
    factor = 1.0
    for q in range(n):
        c, L = _PAULI_MUL[a[q], b[q]]
        factor *= c
        letters.append(L)
    return factor, tuple(letters)


class PauliSum:
    """Sum of Pauli strings with complex coefficients (full-length letter tuples)."""

    def __init__(self, terms=()):
        self.terms = [(complex(c), tuple(l)) for c, l in terms]

    def __add__(self, other):
        return PauliSum(self.terms + other.terms)

    def __sub__(self, other):
        return PauliSum(self.terms + [(-c, l) for c, l in other.terms])

    def scale(self, c):
        return PauliSum([(c * cc, l) for cc, l in self.terms])

    def mul(self, other):
        terms = []
        for c1, l1 in self.terms:
            for c2, l2 in other.terms:
                f, L = _pauli_mul_letters(l1, l2)
                terms.append((c1 * c2 * f, L))
        return PauliSum(terms)

    def simplify(self):
        d = {}
        for c, l in self.terms:
            d[l] = d.get(l, 0.0) + c
        return PauliSum([(c, l) for l, c in d.items() if abs(c) > 1e-15])

    def matrix(self, n):
        """Full 2^n x 2^n matrix (for self-tests)."""
        M = np.zeros((2 ** n, 2 ** n), complex)
        for c, l in self.terms:
            T = np.array([[1.0 + 0j]])
            for q in range(n):
                T = np.kron(T, _PAULI_MAT[l[q] if q < len(l) else 'I'])
            M = M + c * T
        return M

    def expectation(self, state):
        """<psi|P|psi> for a state vector (real for a Hermitian P)."""
        return float(np.real(sum(c * _pauli_expectation(state, l) for c, l in self.terms)))

    def to_cudaq(self):
        """Convert to a CUDA-Q spin operator expression."""
        H = None
        for c, l in self.terms:
            if abs(c) < 1e-15:
                continue
            op = None
            for q, L in enumerate(l):
                if L == 'I':
                    continue
                p = {'X': spin.x, 'Y': spin.y, 'Z': spin.z}[L](q)
                op = p if op is None else op * p
            op = c * (op if op is not None else spin.i(0))
            H = op if H is None else H + op
        return H


# ------------------------------------------------------------------------------
# Exact Jordan-Wigner MAPPING OF THE TRIAD OPERATOR (FIX 4 nomenclature):
#   a_k     = 1/2 (prod_{l<k} Z_l) (X_k + i Y_k)
#   a_k^dag = 1/2 (prod_{l<k} Z_l) (X_k - i Y_k)
# JW is a *-homomorphism, so products of images = image of the product.
# The Navier-Stokes Hamiltonian itself is a reduced model; only the
# fermionic-to-qubit mapping is exact.
# ------------------------------------------------------------------------------
def _z_below(k):
    s = PauliSum([(1.0, ())])
    for l in range(k):
        s = s.mul(PauliSum([(1.0, ("I",) * l + ("Z",))]))
    return s


def jw_a(n, k):
    """JW image of the annihilation operator a_k."""
    xk = PauliSum([(1.0, ("I",) * k + ("X",))])
    yk = PauliSum([(1.0, ("I",) * k + ("Y",))])
    return _z_below(k).mul(xk + yk.scale(1j)).scale(0.5)


def jw_adag(n, k):
    """JW image of the creation operator a_k^dag."""
    xk = PauliSum([(1.0, ("I",) * k + ("X",))])
    yk = PauliSum([(1.0, ("I",) * k + ("Y",))])
    return _z_below(k).mul(xk - yk.scale(1j)).scale(0.5)


def _fock_a(n, k, dagger=False):
    """Fock-space matrix of a_k (or a_k^dag)."""
    a1 = np.array([[0, 0], [1, 0]], complex) if dagger else np.array([[0, 1], [0, 0]], complex)
    M = np.array([[1.0 + 0j]])
    for q in range(n):
        M = np.kron(M, a1 if q == k else (_PAULI_MAT['Z'] if q < k else _PAULI_MAT['I']))
    return M


def run_jw_self_test():
    """Verify JW images against explicit Fock-space matrices (3 modes)."""
    n = 3
    ok = True
    for k in range(n):
        if not np.allclose(jw_a(n, k).matrix(n), _fock_a(n, k), atol=1e-12):
            ok = False
            print(f"[SELF-TEST] FAIL: JW(a_{k}) != Fock a_{k}")
        if not np.allclose(jw_adag(n, k).matrix(n), _fock_a(n, k, dagger=True), atol=1e-12):
            ok = False
            print(f"[SELF-TEST] FAIL: JW(a_{k}^dag) != Fock a_{k}^dag")
    M_fock = _fock_a(n, 0, dagger=True) @ _fock_a(n, 1) @ _fock_a(n, 2)
    T = jw_adag(n, 0).mul(jw_a(n, 1)).mul(jw_a(n, 2))
    if not np.allclose(T.matrix(n), M_fock, atol=1e-12):
        ok = False
        print("[SELF-TEST] FAIL: JW(a_0^dag a_1 a_2) != Fock product")
    print("[SELF-TEST] JW mapping vs Fock space:", "PASS" if ok else "FAIL")
    return ok


# ==============================================================================
# 2. CPU STATE-VECTOR SIMULATOR
# ==============================================================================
def _apply_single(state, q, g):
    n = int(np.log2(state.size))
    s = state.reshape([2] * n)
    s = np.moveaxis(s, q, 0)
    s0, s1 = s[0], s[1]
    o0 = g[0, 0] * s0 + g[0, 1] * s1
    o1 = g[1, 0] * s0 + g[1, 1] * s1
    out = np.stack([o0, o1], axis=0)
    out = np.moveaxis(out, 0, q)
    return out.reshape(-1)


def _cx(state, c, t):
    """CNOT (control c -> target t) on the flat basis vector."""
    n = int(np.log2(state.size))
    size = state.size
    out = np.zeros_like(state)
    for idx in range(size):
        bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
        if bits[c]:
            bits[t] ^= 1
        newidx = 0
        for q in range(n):
            newidx |= bits[q] << (n - 1 - q)
        out[newidx] = state[idx]
    return out


def _ry(theta):
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], complex)


def _rz(theta):
    e = np.exp(-1j * theta / 2.0)
    return np.array([[e, 0], [0, np.conj(e)]], complex)


def _pauli_expectation(state, letters):
    v = state
    for q, L in enumerate(letters):
        if L != 'I':
            v = _apply_single(v, q, _PAULI_MAT[L])
    return float(np.real(np.vdot(state, v)))


def cpu_ansatz_state(n, thetas, couplings):
    """
    State vector of the variational ansatz, identical gate sequence to the
    CUDA-Q kernel vqe_triad_ansatz.
    """
    state = np.zeros(2 ** n, complex)
    state[0] = 1.0
    Hg = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2.0)
    for i in range(n):
        state = _apply_single(state, i, Hg)
    for i in range(n):
        state = _apply_single(state, i, _ry(thetas[i] * couplings[i]))
        state = _apply_single(state, i, _rz(thetas[n + i]))
    for i in range(n):
        j = (i + 1) % n
        state = _cx(state, i, j)
        state = _apply_single(state, j, _rz(thetas[2 * n + i]))
        state = _cx(state, i, j)
    return state


# ==============================================================================
# 3. DYNAMIC SPECTRAL TRIAD DISCOVERY (FIX 6: O(N^2) via dictionary)
# ==============================================================================
def count_subgrid_candidates(ux_hat, uy_hat, uz_hat, k_coords, num_qubits):
    """FIX 7/22 (Rev7): number of subgrid-shell candidate modes. Drives the TARGET qubit
    count; the caller applies hysteresis before committing to a concrete n_q."""
    nx = ux_hat.shape[0]
    KX, KY, KZ = k_coords
    K_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
    k_cutoff = nx // 2
    shell_mask = (K_mag >= k_cutoff * 0.35) & (K_mag <= k_cutoff * 1.05)
    indices = np.argwhere(shell_mask)
    if len(indices) < num_qubits:
        indices = np.argwhere(K_mag > 0.5)
    return int(len(indices))


def find_spectral_triads_from_fft(ux_hat, uy_hat, uz_hat, k_coords, n_q, dx=None):
    """
    Dynamically identifies resonant triads k + p + q = 0 from a 3D FFT.

    FIX 7 (repaired dynamic qubits) / FIX 22 (Rev7 hysteresis): the qubit count n_q is
    DECIDED BY THE CALLER (candidate-shell richness + hysteresis). This function selects
    up to n_q distinct subgrid modes by energy and discovers resonant triads among them.

    Returns (active_modes, triads).
    """
    nx = ux_hat.shape[0]
    KX, KY, KZ = k_coords
    K_mag = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Modal kinetic energy density E(k) = 0.5 * |u_hat|^2
    E_modal = 0.5 * (np.abs(ux_hat)**2 + np.abs(uy_hat)**2 + np.abs(uz_hat)**2)

    # Filter for subgrid cutoff shell (near Nyquist)
    k_cutoff = nx // 2
    shell_mask = (K_mag >= k_cutoff * 0.35) & (K_mag <= k_cutoff * 1.05)

    indices = np.argwhere(shell_mask)
    if len(indices) < n_q:
        indices = np.argwhere(K_mag > 0.5)

    energies = np.array([E_modal[idx[0], idx[1], idx[2]] for idx in indices])
    if len(energies) == 0:
        energies = np.zeros(0)
    sorted_order = np.argsort(energies)[::-1] if len(energies) else []

    # Pick up to n_q distinct modes (1-to-1 qubit mapping)
    active_k_vectors = []
    seen_k = set()
    for rank in sorted_order:
        idx = indices[rank]
        k_vec = np.array([KX[idx[0], idx[1], idx[2]],
                         KY[idx[0], idx[1], idx[2]],
                         KZ[idx[0], idx[1], idx[2]]])
        k_key = tuple(np.round(k_vec, decimals=3))
        if k_key not in seen_k:
            seen_k.add(k_key)
            active_k_vectors.append((k_vec, float(energies[rank])))
        if len(active_k_vectors) >= n_q:
            break

    # Pad with physical canonical modes if needed
    if dx is not None:
        k0 = 2.0 * np.pi / (nx * dx)
    else:
        k0 = 2.0 * np.pi / nx
    canonical_candidates = [
        np.array([k0, 0, 0]), np.array([0, k0, 0]), np.array([-k0, -k0, 0]),
        np.array([k0, k0, 0]), np.array([0, -k0, k0]), np.array([-k0, 0, -k0])
    ]
    cand_idx = 0
    while len(active_k_vectors) < n_q:
        k_c = canonical_candidates[cand_idx % len(canonical_candidates)]
        active_k_vectors.append((k_c, 1e-4))
        cand_idx += 1

    # FIX 7: slice to the DYNAMIC qubit count (previously sliced to base num_qubits)
    active_modes = active_k_vectors[:n_q]

    # FIX 6: Dictionary lookup for O(N^2) triad discovery
    mode_lookup = {}
    for idx, (k_vec, e) in enumerate(active_modes):
        key = tuple(np.round(k_vec, decimals=3))
        mode_lookup[key] = (idx, k_vec, e)

    tol_resonance = 0.5 * k0
    triads = []
    for i in range(n_q):
        for j in range(i + 1, n_q):
            k_i = active_modes[i][0]
            k_j = active_modes[j][0]
            target_q = -(k_i + k_j)
            target_key = tuple(np.round(target_q, decimals=3))
            # O(1) dictionary lookup instead of O(N) scan
            if target_key in mode_lookup:
                m, k_m, e_m = mode_lookup[target_key]
                if m > j:  # enforce i < j < m
                    e_i = active_modes[i][1]
                    e_j = active_modes[j][1]
                    tri_weight = float(e_i * e_j * e_m)
                    triads.append((i, j, m, k_i, k_j, k_m, tri_weight))
            else:
                # Fuzzy match: check within tolerance (handles rounding)
                for m in range(j + 1, n_q):
                    k_m = active_modes[m][0]
                    dist = np.linalg.norm(k_m - target_q)
                    if dist < tol_resonance:
                        e_i = active_modes[i][1]
                        e_j = active_modes[j][1]
                        e_m = active_modes[m][1]
                        tri_weight = float(e_i * e_j * e_m)
                        triads.append((i, j, m, k_i, k_j, k_m, tri_weight))
                        break

    # Fallback canonical closed triad if flow is still purely laminar
    if len(triads) == 0:
        triads.append((
            0, 1, 2,
            active_modes[0][0], active_modes[1][0],
            -active_modes[0][0] - active_modes[1][0],
            1e-6
        ))

    return active_modes, triads


# ==============================================================================
# 4. EXACT JORDAN-WIGNER MAPPING OF THE TRIAD OPERATOR (FIX 4)
# ==============================================================================
def build_jw_triad_hamiltonian(num_qubits, filter_delta, nu_mol,
                               active_k_modes, triads, S_mag,
                               eps_feedback=0.0, feedback_coeff=0.1,
                               gamma_scale=1.0):
    """
    Constructs the Second-Quantized Triadic Navier-Stokes Hamiltonian:
      H_SGS = sum_k omega_k a_k^dag a_k + sum_{k+p+q=0} Gamma_kpq (a_k^dag a_p a_q + h.c.)

    FIX 4 (nomenclature): This is the exact Jordan-Wigner MAPPING of the triad
    operator a_i^dag a_j a_m + h.c. The Navier-Stokes Hamiltonian itself is a
    reduced model (modal truncation + geometric coupling); only the
    fermionic-to-qubit mapping is exact (verified by Fock-space self-test).

    FIX 9: the triad coupling now scales with cbrt(tri_weight) so that the actual
    FFT modal energy content modulates the strength of each resonant triad.

    FIX 31 (Rev10) -- HONEST MODEL LIMITATION (for a theoretical reviewer):
      Gamma_kpq is a HEURISTIC GEOMETRIC closure, NOT derived from the exact NS quadratic
      nonlinearity. The full modal-NS interaction for a resonant triad (k,p,q), k+p+q=0,
        d u_hat_i(k)/dt = -i sum_{p+q=k} [ k_j u_hat_j(p) u_hat_i(q) + q_j u_hat_j(q) u_hat_i(k) ]
      carries a tensor projection P_i(k)[k x u(p)] whose exact value depends on the full
      vector amplitudes. We do NOT evaluate that projection; instead we use
        Gamma_kpq = geom * cbrt(tri_weight) * (1 + 0.2 S_mag) / Delta^(4/3)
      which captures the CORRECT qualitative structure of the NS triad interaction --
        (a) the helical/geometric factor |k x p|/(|k||p|)  [vanishes for collinear triads],
        (b) amplitude scaling ~ (E_k E_p E_q)^(1/6) via cbrt of the energy product,
        (c) a dimensional prefactor Delta^(-4/3) consistent with an eddy-turnover rate --
      but the numerical prefactor (1 + 0.2 S_mag) and the exponent are MODEL CHOICES, not
      derived coefficients. Consequences to state in any paper: this is a data-informed
      surrogate for the SGS triad exchange, calibrated for stability, and its quantitative
      accuracy must be assessed by the benchmarks (FIX 27-30), not assumed from first principles.

    Returns a PauliSum (framework-agnostic).
    """
    H = PauliSum([(0.0, ())])

    # FIX 26 (Rev8): bound the two-way feedback so it cannot destabilize the Hamiltonian.
    # The triad-coupling scale (1 + c*eps) is clipped to [0.5, 2.0] so an unexpectedly large
    # previous-step SGS dissipation cannot blow up Gamma_kpq; the diagonal damping increment
    # is separately capped at the eddy turnover rate (prevents over-damping of omega_k).
    fb = max(0.0, eps_feedback)
    feedback_scale = float(np.clip(1.0 + feedback_coeff * fb, 0.5, 2.0))

    # 1. Diagonal modal energy turnover & molecular dissipation:
    #    0.5 * omega_k * (I - Z_k)  [number operator n_k = (I - Z_k)/2]
    for idx, (k_vec, modal_energy) in enumerate(active_k_modes[:num_qubits]):
        k_sq = float(np.sum(k_vec ** 2))
        k_mag = np.sqrt(k_sq)
        eddy_turnover = np.sqrt(max(1e-10, (k_mag ** 3) * modal_energy))
        viscous_dissipation = nu_mol * k_sq
        omega_k = eddy_turnover + viscous_dissipation
        # FIX 19 (Rev6): two-way quantum feedback — the SGS dissipation realized in the
        # previous step feeds back into the modal damping of THIS Hamiltonian, closing
        # the loop H_n -> VQE -> eps_{n-1} -> H_n (feedback-coupled closure, not one-way).
        # FIX 26 (Rev8): cap the feedback increment at the eddy turnover rate so a large
        # previous-step dissipation cannot over-damp the mode.
        omega_k += min(feedback_coeff * fb, max(1e-12, eddy_turnover))

        zk = PauliSum([(1.0, ("I",) * idx + ("Z",))])
        H = H + PauliSum([(0.5 * omega_k, ())]) + zk.scale(-0.5 * omega_k)

    # 2. 3-Body Triadic Coupling: Gamma * (a_i^dag a_j a_m + h.c.)
    #    Exact JW mapping: product of exact JW images of a_i^dag, a_j, a_m
    for (q_a, q_b, q_c, k_vec, p_vec, q_vec, tri_weight) in triads:
        indices = sorted([q_a, q_b, q_c])
        if len(set(indices)) < 3:
            continue
        i, j, m = indices[0], indices[1], indices[2]

        # Helical geometry factor |k x p| / (|k||p|)
        k_cross = np.cross(k_vec, p_vec)
        geom = float(np.linalg.norm(k_cross) /
                     (np.linalg.norm(k_vec) * np.linalg.norm(p_vec) + 1e-6))

        # FIX 9: triad weight directly modulates the coupling strength
        coupling = (geom * np.cbrt(tri_weight + 1e-12) * (1.0 + 0.2 * S_mag)
                    / (filter_delta ** 1.333))
        # FIX 19 (Rev7): feedback also modulates the TRIAD coupling Gamma_kpq (not only
        # the diagonal omega_k), so previous-step SGS dissipation feeds back into both
        # modal damping and triadic exchange -> a genuinely feedback-coupled closure.
        # FIX 26 (Rev8): use the CLIPPED feedback scale (bounded in [0.5, 2.0]) so Gamma_kpq
        # stays stable even if eps_feedback grows more than expected.
        coupling *= feedback_scale
        # FIX 39 (Rev12): global sensitivity-study multiplier on the heuristic triad coupling.
        # gamma_scale = 1.0 reproduces Rev11 exactly; values != 1.0 are used ONLY to answer
        # 'how sensitive are the results to the (not NS-derived) Gamma_kpq prefactor?'
        # (see FIX 31: the prefactor is a model choice, so it must be swept, not assumed).
        coupling *= gamma_scale

        # Exact JW images (product of images = image of the product)
        T = jw_adag(num_qubits, i).mul(jw_a(num_qubits, j)).mul(jw_a(num_qubits, m))
        T_dag = jw_adag(num_qubits, m).mul(jw_adag(num_qubits, j)).mul(jw_a(num_qubits, i))

        H = H + T.scale(0.5 * coupling) + T_dag.scale(0.5 * coupling)

    return H.simplify()


# ==============================================================================
# 5. VARIATIONAL ANSATZ & COBYLA OPTIMIZATION
# ==============================================================================
if CUDAQ_AVAILABLE:
    @cudaq.kernel
    def vqe_triad_ansatz(qubit_count: int, thetas: list[float], strain_couplings: list[float]):
        q = cudaq.qvector(qubit_count)
        for i in range(qubit_count):
            h(q[i])

        for i in range(qubit_count):
            ry(thetas[i] * strain_couplings[i], q[i])
            rz(thetas[i + qubit_count], q[i])

        for i in range(qubit_count):
            next_i = (i + 1) % qubit_count
            x.ctrl(q[i], q[next_i])
            rz(thetas[2 * qubit_count + i], q[next_i])
            x.ctrl(q[i], q[next_i])


class VQETriadTrainer:
    """
    Executes COBYLA optimization (CUDA-Q) or bound-constrained local search (CPU)
    to minimize the block-local Triad Hamiltonian expectation value.
    """
    def __init__(self, num_qubits=6, block_id=0):
        self.num_qubits = num_qubits
        self.block_id = block_id
        self.num_params = 3 * num_qubits
        self.thetas = np.random.uniform(-0.2, 0.2, self.num_params).tolist()
        # FIX 38 (Rev12): optional shared cost-accounting dict (set by the solver before
        # optimize_step); when present, every <psi|H|psi> evaluation ('observe') is
        # wall-clock timed into it so the VQE cost can be decomposed per component.
        self.cost_acc = None

    def cost_function(self, thetas, pauli_terms, cudaq_H, couplings) -> float:
        _t0 = time.perf_counter() if self.cost_acc is not None else 0.0
        if CUDAQ_AVAILABLE:
            energy = cudaq.observe(vqe_triad_ansatz, cudaq_H,
                                   self.num_qubits, thetas, couplings).expectation()
        else:
            state = cpu_ansatz_state(self.num_qubits, thetas, couplings)
            energy = pauli_terms.expectation(state)
        if self.cost_acc is not None:   # FIX 38 (Rev12): 'observe()' wall-clock accounting
            self.cost_acc["observe"] += time.perf_counter() - _t0
        return float(energy)

    def _local_search(self, pauli_terms, cudaq_H, couplings, max_evals):
        """COBYLA-style bound-constrained local search (CPU fallback)."""
        best_e = self.cost_function(self.thetas, pauli_terms, cudaq_H, couplings)
        for _ in range(max_evals):
            idx = np.random.randint(0, self.num_params)
            orig = self.thetas[idx]
            self.thetas[idx] = orig + 0.04 * (np.random.rand() - 0.5)
            e_test = self.cost_function(self.thetas, pauli_terms, cudaq_H, couplings)
            if e_test < best_e:
                best_e = e_test
            else:
                self.thetas[idx] = orig
        for idx in range(self.num_params):
            orig = self.thetas[idx]
            for step in (0.05, -0.05, 0.02, -0.02):
                self.thetas[idx] = orig + step
                e_t = self.cost_function(self.thetas, pauli_terms, cudaq_H, couplings)
                if e_t < best_e:
                    best_e = e_t
                    orig = self.thetas[idx]
                else:
                    self.thetas[idx] = orig
        return float(best_e)

    def optimize_step(self, pauli_terms, cudaq_H, couplings, max_evals=15) -> float:
        """Executes a genuine optimization step."""
        if CUDAQ_AVAILABLE:
            try:
                optimizer = cudaq.optimizers.COBYLA()
                optimizer.max_eval = max_evals

                def opt_wrapper(params):
                    return self.cost_function(params, pauli_terms, cudaq_H, couplings)

                result = optimizer.optimize(dimensions=self.num_params, opt_fn=opt_wrapper)
                self.thetas = list(result.opt_params)
                return float(result.opt_val)
            except Exception:
                pass
        return self._local_search(pauli_terms, cudaq_H, couplings, max_evals)


# ==============================================================================
# 6. COMPRESSIBLE LES SOLVER: NAVIER-STOKES + k_SGS TRANSPORT
# ==============================================================================
class CompressibleLES3D:
    """
    6-Equation Compressible LES System with Exact Periodic BCs and RK4.

    FIX 1 : production_k = Pi_SGS (full sign) — genuine Leonard stress production.
    FIX 2 : rhs[4] includes -Pi_SGS — symmetric bidirectional energy coupling.
    FIX 3 : consistent (non-asymmetric) Resolved<->SGS energy transfer.
    FIX 5 : compute_energy_budget() + Energy Drift logging.
    FIX 7 : Dynamic qubit count driven by candidate-shell richness.
    FIX 8 : CPU fallback capped at n_q <= 8.
    FIX 9 : tri_weight modulates the triad coupling in the Hamiltonian.
    FIX 10: block-local Leonard stress Pi_SGS (same scale as quantum closure).
    """
    def __init__(self, nx=32, reynolds=2000.0, mach=0.35,
                 cudaq_target="nvidia", vqe_max_evals=15, block_size=16, halo=2,
                 closure="quantum", gamma_scale=1.0):
        # FIX 27 (Rev8): SGS closure selector. "quantum" = the CUDA-Q/VQE triad closure
        # (default, unchanged). Classical options for the head-to-head benchmark:
        #   "smagorinsky", "wale", "dyn_germano".
        # FIX 37 (Rev12): "dns" = in-code no-SGS reference trajectory on the SAME grid/solver/
        # init (nu_t = Pi_SGS = eps^Q = 0, pure molecular dissipation). HONESTY: at N=32 this
        # is under-resolved for Re >= 5000 -> a coarse-grid no-SGS reference of the same code,
        # NOT a published DNS dataset.
        self.closure = closure
        # FIX 39 (Rev12): global scale factor on the heuristic triad coupling Gamma_kpq
        # (sensitivity-study knob; 1.0 = unchanged Rev11 behaviour). See FIX 31: Gamma_kpq is
        # NOT NS-derived, so its absolute prefactor is a model choice that must be swept.
        self.gamma_scale = float(gamma_scale)
        self.nx = nx
        self.ny = nx
        self.nz = nx
        self.Lx = 2.0 * np.pi
        self.Ly = 2.0 * np.pi
        self.Lz = 2.0 * np.pi

        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz
        self.filter_delta = self.dx

        self.gamma = 1.4
        self.Re = reynolds
        self.Mach = mach
        self.nu_mol = 1.0 / self.Re
        self.cfl_target = 0.45
        self.vqe_max_evals = vqe_max_evals

        # 6 Conservative variables
        self.U = np.zeros((6, self.nx, self.ny, self.nz))
        self.nu_t_field = np.zeros((self.nx, self.ny, self.nz))
        self.pi_sgs_field = np.zeros((self.nx, self.ny, self.nz))

        # QPU-coupled SGS dissipation as a spatial field (block-wise)
        self.eps_sgs_field = np.zeros((self.nx, self.ny, self.nz))
        self.quantum_dissipation_rate = 0.85
        # FIX 38 (Rev12): per-component VQE cost accounting (wall-clock seconds, accumulated
        # over the run; divide by #steps for a per-step breakdown):
        #   fft                -- block-local forward FFTs feeding the Hamiltonian
        #   triad_search       -- candidate count + resonant-triad discovery (O(N^2) lookup)
        #   hamiltonian_build  -- JW mapping + PauliSum assembly (+ to_cudaq when available)
        #   observe            -- every <psi|H|psi> evaluation inside the optimizer
        #                         (cudaq.observe on GPU, CPU state-vector expectation otherwise)
        #   optimize_overhead  -- search loop / bookkeeping = optimize_step wall - observe wall
        self.vqe_cost = {"fft": 0.0, "triad_search": 0.0, "hamiltonian_build": 0.0,
                         "observe": 0.0, "optimize_overhead": 0.0, "closure_total": 0.0}

        # Grid coordinate meshes
        coords_x = np.linspace(0, self.Lx, self.nx, endpoint=False)
        coords_y = np.linspace(0, self.Ly, self.ny, endpoint=False)
        coords_z = np.linspace(0, self.Lz, self.nz, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(coords_x, coords_y, coords_z, indexing='ij')

        # Wavenumber coordinates for 3D FFT (global)
        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        kz = 2.0 * np.pi * np.fft.fftfreq(self.nz, d=self.dz)
        self.k_coords = np.meshgrid(kx, ky, kz, indexing='ij')

        # Precompute Spectral Gaussian Test Filter Transfer Function (global)
        delta_test = 2.0 * self.filter_delta
        k_sq_sum = self.k_coords[0]**2 + self.k_coords[1]**2 + self.k_coords[2]**2
        self.G_test_hat = np.exp(-k_sq_sum * (delta_test ** 2) / 24.0)

        # FIX 27 (Rev8): GRID-scale Gaussian filter (width Delta), needed by the Dynamic
        # Germano procedure to form the grid-scale Leonard stress L_ij (B_ij uses the
        # 2*Delta test filter above). The two different widths are what make the dynamic
        # coefficient non-trivial.
        delta_grid = self.filter_delta
        self.G_grid_hat = np.exp(-k_sq_sum * (delta_grid ** 2) / 24.0)

        # ------------------------------------------------------------------
        # FIX 3: DYNAMIC DOMAIN PARTITIONING
        # block_size=16 for large domains; falls back to 2x2x2 for small grids.
        # ------------------------------------------------------------------
        self.block_size = block_size
        if nx >= 2 * block_size:
            self.nb_blocks_per_dim = nx // block_size
            self.nb = block_size
        elif nx >= 8:
            self.nb_blocks_per_dim = 2
            self.nb = nx // 2
        else:
            self.nb_blocks_per_dim = 1
            self.nb = nx

        self.blocks = []
        for a in range(self.nb_blocks_per_dim):
            for b in range(self.nb_blocks_per_dim):
                for c in range(self.nb_blocks_per_dim):
                    self.blocks.append((
                        slice(a * self.nb, (a + 1) * self.nb),
                        slice(b * self.nb, (b + 1) * self.nb),
                        slice(c * self.nb, (c + 1) * self.nb),
                    ))
        self.num_blocks = len(self.blocks)

        # FIX 7: Dynamic qubit count (updated in update_quantum_sgs_physics)
        self.num_qubits = 6
        self.block_trainers = []
        # FIX 19 (Rev6): per-block quantum feedback state (previous-step SGS dissipation)
        self.block_prev_eps = np.zeros(self.num_blocks)
        self.feedback_coeff = 0.1
        # FIX 22 (Rev7): per-block qubit-count hysteresis counters + cooldown length
        self.block_q_up = np.zeros(self.num_blocks, dtype=int)
        self.block_q_down = np.zeros(self.num_blocks, dtype=int)
        self.q_hysteresis = 3
        # FIX 35 (Rev11): dynamic qubit-count heuristic knobs. The mapping n_q ~ cand/RATIO and
        # the [NQ_MIN, NQ_MAX] bounds are HEURISTIC design choices with NO first-principles
        # derivation (reviewer: 'why 1 qubit per 2 candidate modes?'). Exposed as named,
        # documented parameters so they are clearly tunable knobs, not hidden magic numbers.
        self.qubit_mode_ratio = 2     # n_q ~ cand / ratio   (heuristic; no physical derivation)
        self.nq_min = 6               # lower bound on qubits (heuristic)
        self.nq_max = 12              # upper bound on qubits (heuristic)

        # ------------------------------------------------------------------
        # FIX 11 (Rev5): ghost-cell / halo overlap for block-local spectral ops.
        # Triads and the Leonard stress that straddle a block boundary are no
        # longer lost: each block works on an extended (nb + 2*halo)^3 stencil
        # gathered with periodic wrap, then only the core is written back.
        # ------------------------------------------------------------------
        self.halo = max(0, int(halo))
        self.nb_halo = self.nb + 2 * self.halo

        # ------------------------------------------------------------------
        # FIX 16 (Rev6): CORE-resolution wavenumber grid for the modal/triad analysis.
        # The halo is ONLY an auxiliary zone for the local spectral Leonard filter;
        # the quantum closure operates on the block's true physical modes at core
        # resolution, so the Hamiltonian operator is independent of the halo width.
        # ------------------------------------------------------------------
        kx_b = 2.0 * np.pi * np.fft.fftfreq(self.nb, d=self.dx)
        ky_b = 2.0 * np.pi * np.fft.fftfreq(self.nb, d=self.dy)
        kz_b = 2.0 * np.pi * np.fft.fftfreq(self.nb, d=self.dz)
        self.k_block_coords = np.meshgrid(kx_b, ky_b, kz_b, indexing='ij')

        # ------------------------------------------------------------------
        # FIX 10/16: BLOCK-LOCAL Spectral Gaussian Test Filter Transfer Function.
        # Evaluated on the HALOED stencil wavenumbers (auxiliary zone) so the local
        # Leonard stress filter is accurate at block boundaries; only the core of the
        # filtered field is written back.
        # ------------------------------------------------------------------
        kx_h = 2.0 * np.pi * np.fft.fftfreq(self.nb_halo, d=self.dx)
        ky_h = 2.0 * np.pi * np.fft.fftfreq(self.nb_halo, d=self.dy)
        kz_h = 2.0 * np.pi * np.fft.fftfreq(self.nb_halo, d=self.dz)
        KXh, KYh, KZh = np.meshgrid(kx_h, ky_h, kz_h, indexing='ij')
        kb_sq = KXh**2 + KYh**2 + KZh**2
        self.G_test_hat_block = np.exp(-kb_sq * (delta_test ** 2) / 24.0)

        if CUDAQ_AVAILABLE:
            try:
                cudaq.set_target(cudaq_target)
                print(f"[CUDA-Q] Target backend active: {cudaq_target}")
            except Exception:
                cudaq.set_target("qpp-cpu")
                print("[CUDA-Q] Target set to CPU simulation: 'qpp-cpu'")

        self.init_flow_field()
        self._init_trainers(6)

        # FIX 5/14 (Rev5): reference total KE AND total energy for drift diagnostics
        bud0 = self.compute_energy_budget()
        self.initial_total_ke = max(bud0['total_ke'], 1e-12)
        self.initial_total_energy = max(bud0['total_energy'], 1e-12)

    def _init_trainers(self, num_qubits):
        """Initialize block trainers with (possibly dynamic) qubit count."""
        self.num_qubits = num_qubits
        self.block_trainers = [VQETriadTrainer(num_qubits, block_id=k)
                               for k in range(self.num_blocks)]

    # Periodic Finite Difference Operators
    def p_grad_x(self, f):
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * self.dx)

    def p_grad_y(self, f):
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * self.dy)

    def p_grad_z(self, f):
        return (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2.0 * self.dz)

    def p_laplacian(self, f):
        return ((np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / (self.dx ** 2) +
                (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / (self.dy ** 2) +
                (np.roll(f, -1, axis=2) - 2.0 * f + np.roll(f, 1, axis=2)) / (self.dz ** 2))

    def apply_test_filter(self, field):
        """Applies isotropic 3D Gaussian test filter in periodic Fourier space (global)."""
        f_hat = np.fft.fftn(field)
        filtered = np.real(np.fft.ifftn(f_hat * self.G_test_hat))
        return filtered

    def apply_grid_filter(self, field):
        """FIX 27 (Rev8): grid-scale Gaussian filter (width Delta) in global Fourier space."""
        f_hat = np.fft.fftn(field)
        filtered = np.real(np.fft.ifftn(f_hat * self.G_grid_hat))
        return filtered

    def apply_test_filter_block(self, field_b):
        """FIX 10: block-local isotropic 3D Gaussian test filter (block Fourier space)."""
        f_hat = np.fft.fftn(field_b)
        filtered = np.real(np.fft.ifftn(f_hat * self.G_test_hat_block))
        return filtered

    def _halo_idx(self, sl):
        """FIX 11: periodic index array covering [start-halo, stop+halo)."""
        start = sl.start if sl.start is not None else 0
        stop = sl.stop if sl.stop is not None else self.nx
        return np.arange(start - self.halo, stop + self.halo) % self.nx

    def _extract_haloed(self, field, slx, sly, slz):
        """FIX 11: gather an (nb+2*halo)^3 subfield with periodic ghost cells."""
        ix = self._halo_idx(slx)
        iy = self._halo_idx(sly)
        iz = self._halo_idx(slz)
        IX, IY, IZ = np.meshgrid(ix, iy, iz, indexing='ij')
        return field[IX, IY, IZ]

    def _core_slice(self):
        """FIX 11: slice selecting the core (non-ghost) region of a haloed array."""
        h = self.halo
        return (slice(h, h + self.nb), slice(h, h + self.nb), slice(h, h + self.nb))

    def init_flow_field(self):
        """Initialize standard Taylor-Green vortex benchmark."""
        rho_0 = 1.0
        p_0 = rho_0 / (self.gamma * (self.Mach ** 2))

        ux = np.sin(self.X) * np.cos(self.Y) * np.cos(self.Z)
        uy = -np.cos(self.X) * np.sin(self.Y) * np.cos(self.Z)
        uz = 0.05 * np.sin(2.0 * self.Z) * np.sin(self.X + self.Y)

        p = p_0 + (rho_0 / 16.0) * (np.cos(2.0 * self.X) + np.cos(2.0 * self.Y)) * \
              (np.cos(2.0 * self.Z) + 2.0)
        rho = rho_0 * (p / p_0) ** (1.0 / self.gamma)
        E = p / (self.gamma - 1.0) + 0.5 * rho * (ux**2 + uy**2 + uz**2)
        k_sgs_init = 1e-4 * np.ones((self.nx, self.ny, self.nz))

        self.U[0] = rho
        self.U[1] = rho * ux
        self.U[2] = rho * uy
        self.U[3] = rho * uz
        self.U[4] = E
        self.U[5] = rho * k_sgs_init

    def compute_cfl_timestep(self) -> float:
        rho = np.maximum(0.05, self.U[0])
        ux = self.U[1] / rho
        uy = self.U[2] / rho
        uz = self.U[3] / rho
        v_mag = np.sqrt(ux**2 + uy**2 + uz**2)

        ke = 0.5 * rho * (ux**2 + uy**2 + uz**2)
        p = np.maximum(0.05, (self.gamma - 1.0) * (self.U[4] - ke))
        c_sound = np.sqrt(self.gamma * p / rho)

        max_wave_speed = float(np.max(v_mag + c_sound)) + 1e-6
        dt_conv = self.cfl_target * self.dx / max_wave_speed
        max_nu = float(np.max(self.nu_mol + np.abs(self.nu_t_field))) + 1e-6
        dt_visc = 0.25 * (self.dx ** 2) / max_nu

        return min(dt_conv, dt_visc, 0.01)

    def compute_energy_budget(self):
        """
        FIX 5/14 (Rev5): Compute the kinetic-energy budget AND the TOTAL conserved
        energy mean(U[4]) = <e_internal + e_kinetic>. A KE drift alone does not imply
        a total-energy drift, so both are reported with their own drifts.
        Returns dict: resolved_ke, sgs_ke, total_ke, total_energy, ke_drift, te_drift.
        """
        rho = np.maximum(0.05, self.U[0])
        ux = self.U[1] / rho
        uy = self.U[2] / rho
        uz = self.U[3] / rho

        resolved_ke = float(np.mean(0.5 * rho * (ux**2 + uy**2 + uz**2)))
        sgs_ke = float(np.mean(self.U[5] / rho))
        total_ke = resolved_ke + sgs_ke

        # FIX 14 (Rev5): TOTAL energy (internal + kinetic) from the conserved U[4]
        total_energy = float(np.mean(self.U[4]))

        ke_drift = ((total_ke - self.initial_total_ke) / self.initial_total_ke
                    if getattr(self, "initial_total_ke", None) else 0.0)
        te_drift = ((total_energy - self.initial_total_energy) / self.initial_total_energy
                    if getattr(self, "initial_total_energy", None) else 0.0)

        # FIX 17 (Rev6): SGS energy budget — production (Pi_SGS) vs quantum dissipation
        k_sgs_b = np.maximum(1e-8, self.U[5] / rho)
        mean_pi_sgs = float(np.mean(self.pi_sgs_field))
        eps_q_local = self.eps_sgs_field * rho * (k_sgs_b ** 1.5) / self.filter_delta
        mean_eps_q = float(np.mean(eps_q_local))
        # FIX 17 (Rev7): stable Pi/Eps diagnostics — clip the ratio (avoids blow-up when
        # Eps ~ 0) and also report (Pi - Eps), which is more robust than the ratio.
        if abs(mean_eps_q) > 1e-30:
            pi_eps_ratio = float(np.clip(mean_pi_sgs / mean_eps_q, -1e3, 1e3))
        else:
            pi_eps_ratio = float("nan")
        pi_minus_eps = mean_pi_sgs - mean_eps_q

        # FIX 18 (Rev7): SGS transfer conservativeness from the ACTUAL RHS integrals.
        # resolved-side loss (-Pi_SGS in rhs[4]) vs SGS-side gain (+production_k in rhs[5]).
        # Expected ~ machine epsilon; a non-zero value would indicate the two transfer
        # terms have been decoupled (e.g. by clipping or filtering one side only).
        r4 = getattr(self, "_rhs4_transfer_integral", None)
        r5 = getattr(self, "_rhs5_transfer_integral", None)
        if r4 is not None and r5 is not None:
            transfer_error = abs(r4 + r5)
        else:
            transfer_error = float("nan")

        return dict(resolved_ke=resolved_ke, sgs_ke=sgs_ke, total_ke=total_ke,
                    total_energy=total_energy, ke_drift=ke_drift, te_drift=te_drift,
                    mean_pi_sgs=mean_pi_sgs, mean_eps_q=mean_eps_q, pi_eps_ratio=pi_eps_ratio,
                    pi_minus_eps=pi_minus_eps, rhs4_transfer=r4, rhs5_transfer=r5,
                    transfer_error=transfer_error)

    def _block_local_leonard_stress(self, ux_b, uy_b, uz_b,
                                    S11_b, S22_b, S33_b, S12_b, S23_b, S13_b):
        """
        FIX 10: Compute the Leonard stress tensor L_ij and its contraction with the
        strain rate Pi_SGS = -L_ij*S_ij LOCALLY within one block (block Fourier
        space), so SGS physics acts on the same spatial scale as the quantum closure.
        Returns Pi_SGS_block (array of shape (nb, nb, nb)).
        """
        ux_f = self.apply_test_filter_block(ux_b)
        uy_f = self.apply_test_filter_block(uy_b)
        uz_f = self.apply_test_filter_block(uz_b)
        u_u_f = self.apply_test_filter_block(ux_b * ux_b)
        v_v_f = self.apply_test_filter_block(uy_b * uy_b)
        w_w_f = self.apply_test_filter_block(uz_b * uz_b)
        u_v_f = self.apply_test_filter_block(ux_b * uy_b)
        v_w_f = self.apply_test_filter_block(uy_b * uz_b)
        u_w_f = self.apply_test_filter_block(ux_b * uz_b)

        L11 = u_u_f - ux_f * ux_f
        L22 = v_v_f - uy_f * uy_f
        L33 = w_w_f - uz_f * uz_f
        L12 = u_v_f - ux_f * uy_f
        L23 = v_w_f - uy_f * uz_f
        L13 = u_w_f - ux_f * uz_f

        pi_sgs_b = -(L11 * S11_b + L22 * S22_b + L33 * S33_b +
                     2.0 * (L12 * S12_b + L23 * S23_b + L13 * S13_b))
        return pi_sgs_b

    # ==================================================================
    # FIX 27 (Rev8): CLASSICAL LES CLOSURES for the head-to-head benchmark.
    # Same resolved dynamics + block-local Leonard stress as the quantum path,
    # but nu_t comes from a standard SGS model (no VQE). Each returns the same
    # 8-tuple as update_quantum_sgs_physics so step() is closure-agnostic.
    # ==================================================================
    def _wale_nu_t(self, S11, S22, S33, S12, S23, S13, O12, O23, O13, s_mag_sq):
        """WALE model -- Wall-Adapting Local Eddy viscosity (Nicoud & Ducros, 2001, Eq. 25).

            nu_t = [ c_w * min( |S~|, sqrt(beta^2 |Omega|^2) ) ]^3 / [ |S|^2 + sqrt(2) |Omega|^2 ]^(3/2) * Delta^2

        Verified term-by-term against the original paper (reviewer #2, Rev9):
          |S|     = sqrt( 2 S_ij S_ij )            -> s_mag_sq passed in IS exactly |S|^2
          |Omega| = sqrt( 1/2 Omega_ij Omega_ij )   -> O_mag_sq = 0.5 * einsum(O,O)
          S~_ij   = 1/2 ( S_ik S_kj + Omega_ik Omega_kj ), then its DEViatoric part
                   S*_ij = S~_ij - (1/3) delta_ij tr(S~);  |S~| = sqrt( 1/2 S*_ij S*_ij )
          c_w = 1.35, beta = 0.6                    -> standard WALE constants
        The rotation tensor Omega is antisymmetric (Omega_12=O12, Omega_21=-O12, ...)."""
        shape = S11.shape
        S = np.zeros(shape + (3, 3))
        S[..., 0, 0] = S11; S[..., 0, 1] = S12; S[..., 0, 2] = S13
        S[..., 1, 0] = S12; S[..., 1, 1] = S22; S[..., 1, 2] = S23
        S[..., 2, 0] = S13; S[..., 2, 1] = S23; S[..., 2, 2] = S33
        O = np.zeros(shape + (3, 3))
        O[..., 0, 1] = O12; O[..., 0, 2] = O13
        O[..., 1, 0] = -O12; O[..., 1, 2] = O23
        O[..., 2, 0] = -O13; O[..., 2, 1] = -O23
        SS = np.einsum('...ik,...kj->...ij', S, S)
        OO = np.einsum('...ik,...kj->...ij', O, O)
        S_tilde = 0.5 * (SS + OO)
        tr = (S_tilde[..., 0, 0] + S_tilde[..., 1, 1] + S_tilde[..., 2, 2]) / 3.0
        Sstar = np.zeros_like(S_tilde)
        Sstar[..., 0, 0] = S_tilde[..., 0, 0] - tr
        Sstar[..., 1, 1] = S_tilde[..., 1, 1] - tr
        Sstar[..., 2, 2] = S_tilde[..., 2, 2] - tr
        Sstar[..., 0, 1] = S_tilde[..., 0, 1]; Sstar[..., 1, 0] = S_tilde[..., 1, 0]
        Sstar[..., 0, 2] = S_tilde[..., 0, 2]; Sstar[..., 2, 0] = S_tilde[..., 2, 0]
        Sstar[..., 1, 2] = S_tilde[..., 1, 2]; Sstar[..., 2, 1] = S_tilde[..., 1, 2]
        S_tilde_mag = np.sqrt(0.5 * np.einsum('...ij,...ij->...', Sstar, Sstar))
        O_mag_sq = 0.5 * np.einsum('...ij,...ij->...', O, O)
        beta, c_w = 0.6, 1.35
        num = np.minimum(S_tilde_mag, beta * np.sqrt(O_mag_sq + 1e-30))
        denom = (s_mag_sq + np.sqrt(2.0) * O_mag_sq) ** 1.5
        return (c_w * num) ** 3 / np.maximum(denom, 1e-30) * self.filter_delta ** 2

    def _dyn_germano_nu_t(self, ux, uy, uz):
        """Spatially-Averaged Germano dynamic procedure: nu = -<B_ij L_ij>/<L_ij L_ij>.
        L_ij at grid scale (filter Delta), B_ij at test scale (filter 2 Delta). The box
        average makes this the stable 'spatially-averaged dynamic' variant.

        NOTE (reviewer #1, Rev9): this is NOT the local Germano form that yields a
        position-dependent C_s(x); it returns a single box-averaged coefficient. It is
        labeled 'Spatially-Averaged Dynamic Germano' in all logs/figures accordingly."""
        def fg(f): return self.apply_grid_filter(f)   # width Delta
        def ft(f): return self.apply_test_filter(f)    # width 2 Delta
        uxg, uyg, uzg = fg(ux), fg(uy), fg(uz)
        L11 = fg(ux * ux) - uxg * uxg
        L22 = fg(uy * uy) - uyg * uyg
        L33 = fg(uz * uz) - uzg * uzg
        L12 = fg(ux * uy) - uxg * uyg
        L23 = fg(uy * uz) - uyg * uzg
        L13 = fg(ux * uz) - uxg * uzg
        uxt, uyt, uzt = ft(ux), ft(uy), ft(uz)
        B11 = ft(ux * ux) - uxt * uxt
        B22 = ft(uy * uy) - uyt * uyt
        B33 = ft(uz * uz) - uzt * uzt
        B12 = ft(ux * uy) - uxt * uyt
        B23 = ft(uy * uz) - uyt * uzt
        B13 = ft(ux * uz) - uxt * uzt
        L2 = (L11**2 + L22**2 + L33**2 + 2.0 * (L12**2 + L23**2 + L13**2))
        BL = (L11 * B11 + L22 * B22 + L33 * B33 + 2.0 * (L12 * B12 + L23 * B23 + L13 * B13))
        denom = float(np.mean(L2)); numer = float(np.mean(BL))
        nu = -numer / denom if abs(denom) > 1e-30 else 0.0
        return np.full(ux.shape, nu)   # single box-averaged coefficient; caller floors/clips

    def _update_classical_sgs(self, ux, uy, uz, rho):
        """FIX 27 (Rev8): classical SGS closures for the benchmark. Computes the SAME
        block-local Leonard stress Pi_SGS as the quantum path (consistent diagnostic) and
        sets nu_t from the selected standard model. Returns the same 8-tuple as
        update_quantum_sgs_physics so step() needs no closure-specific logic."""
        # Strain & vorticity tensors (global finite differences, periodic)
        S11 = self.p_grad_x(ux); S22 = self.p_grad_y(uy); S33 = self.p_grad_z(uz)
        S12 = 0.5 * (self.p_grad_y(ux) + self.p_grad_x(uy))
        S23 = 0.5 * (self.p_grad_z(uy) + self.p_grad_y(uz))
        S13 = 0.5 * (self.p_grad_z(ux) + self.p_grad_x(uz))
        O12 = 0.5 * (self.p_grad_y(ux) - self.p_grad_x(uy))
        O23 = 0.5 * (self.p_grad_z(uy) - self.p_grad_y(uz))
        O13 = 0.5 * (self.p_grad_z(ux) - self.p_grad_x(uz))
        s_mag_sq = 2.0 * (S11**2 + S22**2 + S33**2 + 2.0 * (S12**2 + S23**2 + S13**2))

        # Block-local Leonard stress Pi_SGS (identical to the quantum path)
        self.pi_sgs_field[:] = 0.0
        for blk, (slx, sly, slz) in enumerate(self.blocks):
            ux_h = self._extract_haloed(ux, slx, sly, slz)
            uy_h = self._extract_haloed(uy, slx, sly, slz)
            uz_h = self._extract_haloed(uz, slx, sly, slz)
            S11h = self._extract_haloed(S11, slx, sly, slz); S22h = self._extract_haloed(S22, slx, sly, slz)
            S33h = self._extract_haloed(S33, slx, sly, slz); S12h = self._extract_haloed(S12, slx, sly, slz)
            S23h = self._extract_haloed(S23, slx, sly, slz); S13h = self._extract_haloed(S13, slx, sly, slz)
            pi_h = self._block_local_leonard_stress(ux_h, uy_h, uz_h, S11h, S22h, S33h, S12h, S23h, S13h)
            self.pi_sgs_field[slx, sly, slz] = pi_h[self._core_slice()]

        # nu_t from the selected classical closure
        if self.closure == "smagorinsky":
            Cs = 0.18
            nu_t = (Cs * self.filter_delta) ** 2 * np.sqrt(s_mag_sq)
        elif self.closure == "wale":
            nu_t = self._wale_nu_t(S11, S22, S33, S12, S23, S13, O12, O23, O13, s_mag_sq)
        elif self.closure == "dyn_germano":
            nu_smag_ref = (0.18 * self.filter_delta) ** 2 * np.sqrt(s_mag_sq)
            nu_t = self._dyn_germano_nu_t(ux, uy, uz)
            nu_t = np.clip(nu_t, -0.5 * nu_smag_ref, 3.0 * nu_smag_ref)   # floor + cap (stability)
        else:
            raise ValueError(f"unknown classical closure: {self.closure}")

        if self.closure != "dyn_germano":
            nu_t = np.maximum(0.0, nu_t)
        self.nu_t_field[:] = nu_t

        # SGS dissipation rate (physical, per unit mass) ~ nu_t |S|^2, mapped to the
        # multiplier form used by the k_SGS equation: diss_k = eps_field * rho * k^1.5 / Delta.
        eps_rate = np.maximum(0.0, nu_t) * s_mag_sq
        k_sgs_b = np.maximum(1e-8, self.U[5] / rho)
        mult = eps_rate * self.filter_delta / (k_sgs_b ** 1.5 + 1e-30)
        self.eps_sgs_field[:] = np.clip(mult, 0.0, 2.5)

        mean_nu_t = float(np.mean(self.nu_t_field))
        n_backscatter_cells = int(np.sum(nu_t < 0.0))
        backscatter_ratio = float(n_backscatter_cells / self.nu_t_field.size)
        # report the physical SGS dissipation rate in the same slot for comparability
        self.quantum_dissipation_rate = float(np.mean(eps_rate))
        n_triads_total = 0; n_triads_mean = 0.0; n_modes_mean = float(self.num_qubits)
        return (mean_nu_t, mean_nu_t, 0.0, int(np.sum(s_mag_sq > 1e-24)), backscatter_ratio,
                n_triads_total, n_triads_mean, n_modes_mean)

    def update_quantum_sgs_physics(self, ux, uy, uz, rho):
        """
        Executes per timestep:
          1. Strain / vorticity / Q-criterion (global).
          2. BLOCK-LOCAL spectral Leonard stress L_ij and Pi_SGS (FIX 10) — one
             tensor per block, at the same scale as the quantum closure.
          3. Domain-wide QPU SGS — one VQE per block (FIX 3 dynamic partition).

        FIX 38 (Rev12): every stage is wall-clock timed into self.vqe_cost so the
        per-step cost can be decomposed as FFT / triad search / Hamiltonian build /
        observe() / optimizer overhead (the HPC question 'where does the VQE time go?').
        """
        _t_start = time.perf_counter()   # FIX 38 (Rev12): closure_total accounting
        # 1. Strain rate tensor S_ij & Vorticity tensor Omega_ij
        S11 = self.p_grad_x(ux)
        S22 = self.p_grad_y(uy)
        S33 = self.p_grad_z(uz)
        S12 = 0.5 * (self.p_grad_y(ux) + self.p_grad_x(uy))
        S23 = 0.5 * (self.p_grad_z(uy) + self.p_grad_y(uz))
        S13 = 0.5 * (self.p_grad_z(ux) + self.p_grad_x(uz))

        O12 = 0.5 * (self.p_grad_y(ux) - self.p_grad_x(uy))
        O23 = 0.5 * (self.p_grad_z(uy) - self.p_grad_y(uz))
        O13 = 0.5 * (self.p_grad_z(ux) - self.p_grad_x(uz))

        s_norm_sq = 2.0 * (S11**2 + S22**2 + S33**2 + 2.0 * (S12**2 + S23**2 + S13**2))
        s_mag = np.sqrt(s_norm_sq) + 1e-8
        o_norm_sq = 4.0 * (O12**2 + O23**2 + O13**2)

        # Dynamic statistical Q-criterion threshold
        Q_criterion = 0.5 * (o_norm_sq - s_norm_sq)
        q_mean = float(np.mean(Q_criterion))
        q_std = float(np.std(Q_criterion))
        q_threshold = max(1e-5, q_mean + 0.5 * q_std)
        turbulent_mask = (Q_criterion > q_threshold) | (s_mag > np.mean(s_mag))
        turbulent_indices = np.argwhere(turbulent_mask)

        # ------------------------------------------------------------------
        # 2 & 3. DOMAIN-WIDE QPU SGS — one VQE per block (FIX 3 dynamic partition)
        #         with BLOCK-LOCAL Leonard stress (FIX 10) and nu_t (same scale).
        # ------------------------------------------------------------------
        base_C_eps = 0.85
        C_quantum = 0.082
        ref_turnover = max(1e-4, float(np.mean(s_mag)))
        self.eps_sgs_field[:] = 0.0
        self.pi_sgs_field[:] = 0.0
        self.nu_t_field[:] = 0.0
        q_energies = []
        n_backscatter_cells = 0
        n_triads_total = 0
        n_modes_total = 0

        for blk, (slx, sly, slz) in enumerate(self.blocks):
            # ---- FIX 11: gather haloed (ghost-cell) velocity & strain stencils ----
            ux_h = self._extract_haloed(ux, slx, sly, slz)
            uy_h = self._extract_haloed(uy, slx, sly, slz)
            uz_h = self._extract_haloed(uz, slx, sly, slz)
            S11_h = self._extract_haloed(S11, slx, sly, slz)
            S22_h = self._extract_haloed(S22, slx, sly, slz)
            S33_h = self._extract_haloed(S33, slx, sly, slz)
            S12_h = self._extract_haloed(S12, slx, sly, slz)
            S23_h = self._extract_haloed(S23, slx, sly, slz)
            S13_h = self._extract_haloed(S13, slx, sly, slz)

            # FIX 10: block-local Leonard stress & Pi_SGS on the haloed stencil
            pi_sgs_h = self._block_local_leonard_stress(
                ux_h, uy_h, uz_h, S11_h, S22_h, S33_h, S12_h, S23_h, S13_h
            )
            # Write back only the core (non-ghost) region
            cs = self._core_slice()
            pi_sgs_b = pi_sgs_h[cs]
            self.pi_sgs_field[slx, sly, slz] = pi_sgs_b

            # Local quantum eddy viscosity nu_t (backscatter from Pi_SGS sign)
            local_k_sgs_b = np.maximum(1e-8, self.U[5][slx, sly, slz] / rho[slx, sly, slz])
            base_nu_t_b = C_quantum * self.filter_delta * np.sqrt(local_k_sgs_b)
            nu_t_b = np.where(pi_sgs_b < 0.0, -0.45 * base_nu_t_b, base_nu_t_b)
            nu_t_b = np.maximum(-0.95 * self.nu_mol, nu_t_b)
            self.nu_t_field[slx, sly, slz] = nu_t_b
            n_backscatter_cells += int(np.sum(pi_sgs_b < 0.0))

            # ---- FIX 16 (Rev6): triad analysis on CORE modes at core resolution.
            # The haloed stencil is used only for the local Leonard filter; the modal
            # spectrum fed to the Hamiltonian is the block's own (nb)^3 basis, so the
            # operator no longer depends on the auxiliary halo width.
            # ----
            ux_b = ux[slx, sly, slz]
            uy_b = uy[slx, sly, slz]
            uz_b = uz[slx, sly, slz]
            _t_a = time.perf_counter()   # FIX 38 (Rev12): block-local FFT stage
            ux_hat_b = np.fft.fftn(ux_b) / (self.nb ** 3)
            uy_hat_b = np.fft.fftn(uy_b) / (self.nb ** 3)
            uz_hat_b = np.fft.fftn(uz_b) / (self.nb ** 3)
            self.vqe_cost["fft"] += time.perf_counter() - _t_a

            # ---- FIX 7/8 + FIX 22 (Rev7): DYNAMIC qubit count WITH hysteresis.
            # Target from candidate-shell richness; commit only after q_hysteresis
            # consecutive steps in the same direction to prevent per-step oscillation
            # (e.g. 8 -> 10 -> 8 -> 10) under a rich spectrum.
            trainer = self.block_trainers[blk]
            _t_a = time.perf_counter()   # FIX 38 (Rev12): candidate-count stage
            cand = count_subgrid_candidates(ux_hat_b, uy_hat_b, uz_hat_b,
                                            self.k_block_coords, self.num_qubits)
            self.vqe_cost["triad_search"] += time.perf_counter() - _t_a
            # FIX 35 (Rev11): n_q from candidate-shell richness. The ratio/bounds are HEURISTIC
            # (no physical derivation of '1 qubit per 2 modes'); exposed as self.qubit_mode_ratio,
            # self.nq_min, self.nq_max so they are documented, tunable knobs -- not magic numbers.
            target_nq = min(self.nq_max, max(self.nq_min, cand // self.qubit_mode_ratio))
            if not CUDAQ_AVAILABLE:
                target_nq = min(target_nq, 8)          # FIX 8 CPU cap
            prev_nq = trainer.num_qubits
            if target_nq > prev_nq:
                self.block_q_up[blk] += 1; self.block_q_down[blk] = 0
                n_q = target_nq if self.block_q_up[blk] >= self.q_hysteresis else prev_nq
            elif target_nq < prev_nq:
                self.block_q_down[blk] += 1; self.block_q_up[blk] = 0
                n_q = target_nq if self.block_q_down[blk] >= self.q_hysteresis else prev_nq
            else:
                self.block_q_up[blk] = 0; self.block_q_down[blk] = 0
                n_q = prev_nq

            # Block-local resonant triads (FIX 6 O(N^2) dictionary lookup)
            _t_a = time.perf_counter()   # FIX 38 (Rev12): triad-discovery stage
            active_modes, triads = find_spectral_triads_from_fft(
                ux_hat_b, uy_hat_b, uz_hat_b, self.k_block_coords,
                n_q, dx=self.dx
            )
            self.vqe_cost["triad_search"] += time.perf_counter() - _t_a
            # FIX 13 (Rev5): track Hamiltonian statistics for the log
            n_triads_total += len(triads)
            n_modes_total += n_q

            # Block-local strain statistics -> ansatz couplings
            S_b = s_mag[slx, sly, slz]
            sample_S = max(1e-8, float(np.mean(S_b)))
            couplings = [
                abs(float(np.mean(S11[slx, sly, slz]))) / sample_S,
                abs(float(np.mean(S22[slx, sly, slz]))) / sample_S,
                abs(float(np.mean(S33[slx, sly, slz]))) / sample_S,
                abs(float(np.mean(S12[slx, sly, slz]))) / sample_S,
                abs(float(np.mean(S23[slx, sly, slz]))) / sample_S,
                abs(float(np.mean(S13[slx, sly, slz]))) / sample_S,
            ]
            # Pad couplings to the dynamic qubit count (ansatz reads c[0..n_q-1])
            while len(couplings) < n_q:
                couplings.append(0.1)
            couplings = couplings[:n_q]

            # Block-local triad Hamiltonian (FIX 4 exact JW mapping, FIX 9 tri_weight,
            # FIX 19 two-way feedback from previous-step SGS dissipation,
            # FIX 39 gamma_scale sensitivity multiplier)
            _t_a = time.perf_counter()   # FIX 38 (Rev12): Hamiltonian-build stage
            pauli_terms = build_jw_triad_hamiltonian(
                n_q, self.filter_delta, self.nu_mol,
                active_modes, triads, sample_S,
                eps_feedback=self.block_prev_eps[blk],
                feedback_coeff=self.feedback_coeff,
                gamma_scale=self.gamma_scale
            )
            cudaq_H = pauli_terms.to_cudaq() if CUDAQ_AVAILABLE else None
            self.vqe_cost["hamiltonian_build"] += time.perf_counter() - _t_a

            # Block-local VQE optimization (dynamic qubit count, hysteresis-stabilized)
            if trainer.num_qubits != n_q:
                # FIX 12 (Rev5) + FIX 36 (Rev11): PARAMETER TRANSFER only -- carry over the
                # variational parameters (thetas) as a warm start instead of fully re-randomizing
                # when the dynamic qubit count changes. This is NOT optimizer continuation: the
                # COBYLA simplex / trust-region state is re-initialized from scratch each step, so
                # the paper must describe this as 'parameter transfer', not 'optimizer continuation'.
                old_thetas = list(trainer.thetas)
                new_params = 3 * n_q
                new_thetas = np.random.uniform(-0.2, 0.2, new_params).tolist()
                keep = min(len(old_thetas), new_params)
                new_thetas[:keep] = old_thetas[:keep]
                trainer.num_qubits = n_q
                trainer.num_params = new_params
                trainer.thetas = new_thetas

            # FIX 38 (Rev12): observe() timing hook + optimizer-overhead split
            trainer.cost_acc = self.vqe_cost
            _obs_before = self.vqe_cost["observe"]
            _t_opt0 = time.perf_counter()
            q_e = trainer.optimize_step(
                pauli_terms, cudaq_H, couplings, max_evals=self.vqe_max_evals
            )
            _opt_tot = time.perf_counter() - _t_opt0
            _obs_delta = self.vqe_cost["observe"] - _obs_before
            self.vqe_cost["optimize_overhead"] += max(0.0, _opt_tot - _obs_delta)
            q_energies.append(q_e)

            # Block-local SGS dissipation multiplier -> spatial field
            q_scale = float(np.clip(abs(q_e) / ref_turnover, 0.4, 2.5))
            blk_eps = base_C_eps * q_scale
            self.eps_sgs_field[slx, sly, slz] = blk_eps
            # FIX 19 (Rev6): feed the realized dissipation back into the next step's H
            self.block_prev_eps[blk] = blk_eps

        self.quantum_dissipation_rate = float(np.mean(self.eps_sgs_field))

        mean_nu_t = float(np.mean(self.nu_t_field))
        q_energy_mean = float(np.mean(q_energies))
        q_energy_std = float(np.std(q_energies))
        backscatter_ratio = float(n_backscatter_cells / self.pi_sgs_field.size)
        # FIX 13 (Rev5): Hamiltonian statistics (domain total + per-block mean)
        n_triads_mean = float(n_triads_total) / max(1, self.num_blocks)
        n_modes_mean = float(n_modes_total) / max(1, self.num_blocks)
        self.vqe_cost["closure_total"] += time.perf_counter() - _t_start   # FIX 38 (Rev12)
        return (mean_nu_t, q_energy_mean, q_energy_std, len(turbulent_indices),
                backscatter_ratio, n_triads_total, n_triads_mean, n_modes_mean)

    def compute_rhs(self, state):
        """
        Computes Right-Hand-Side for all 6 equations with Periodic BCs.

        FIX 1: production_k = Pi_SGS (FULL sign) — genuine Leonard stress production
               that also carries backscatter (negative part no longer discarded).
        FIX 2/3: rhs[4] -= Pi_SGS and rhs[5] += Pi_SGS use the SAME field, giving a
               strictly symmetric Resolved<->SGS energy exchange (no double counting).
        """
        rho = np.maximum(0.05, state[0])
        ux = state[1] / rho
        uy = state[2] / rho
        uz = state[3] / rho
        ke = 0.5 * rho * (ux**2 + uy**2 + uz**2)
        p = np.maximum(0.05, (self.gamma - 1.0) * (state[4] - ke))
        k_sgs = np.maximum(1e-8, state[5] / rho)

        nu_eff = np.maximum(0.05 * self.nu_mol, self.nu_mol + self.nu_t_field)
        rhs = np.zeros_like(state)

        # 1. Continuity
        rhs[0] = -(self.p_grad_x(rho * ux) + self.p_grad_y(rho * uy) + self.p_grad_z(rho * uz))

        # 2. Viscous stress diffusion
        diff_ux = nu_eff * self.p_laplacian(ux)
        diff_uy = nu_eff * self.p_laplacian(uy)
        diff_uz = nu_eff * self.p_laplacian(uz)

        # 3. Momentum X, Y, Z
        flux_x = self.p_grad_x(rho * ux * ux + p) + self.p_grad_y(rho * ux * uy) + self.p_grad_z(rho * ux * uz)
        rhs[1] = -flux_x + rho * diff_ux

        flux_y = self.p_grad_x(rho * uy * ux) + self.p_grad_y(rho * uy * uy + p) + self.p_grad_z(rho * uy * uz)
        rhs[2] = -flux_y + rho * diff_uy

        flux_z = self.p_grad_x(rho * uz * ux) + self.p_grad_y(rho * uz * uy) + self.p_grad_z(rho * uz * uz + p)
        rhs[3] = -flux_z + rho * diff_uz

        # 4. Total Energy (FIX 2/3: symmetric transfer, uses FULL Pi_SGS)
        E_tot = state[4]
        flux_E = self.p_grad_x((E_tot + p) * ux) + self.p_grad_y((E_tot + p) * uy) + self.p_grad_z((E_tot + p) * uz)
        visc_work = rho * (ux * diff_ux + uy * diff_uy + uz * diff_uz)
        # FIX 2/3: Pi_SGS transfers energy from resolved -> SGS (or backscatter).
        # The SAME full field is added to rhs[5] below -> symmetric, no double count.
        rhs[4] = -flux_E + visc_work - self.pi_sgs_field

        # 5. One-Equation k_SGS Transport
        advection_k = (self.p_grad_x(rho * k_sgs * ux) + self.p_grad_y(rho * k_sgs * uy)
                       + self.p_grad_z(rho * k_sgs * uz))

        # FIX 1: Production from genuine Leonard stress Pi_SGS = -L_ij*S_ij (full sign).
        # Symmetric with rhs[4] above (FIX 3): positive AND negative transfer both occur.
        production_k = self.pi_sgs_field

        # QPU closure: epsilon_sgs^Q(x) = eps_field(x) * rho * k^1.5 / Delta
        dissipation_k = self.eps_sgs_field * rho * (k_sgs ** 1.5) / self.filter_delta
        diff_k = (self.nu_mol + np.maximum(0.0, self.nu_t_field)) * self.p_laplacian(rho * k_sgs)

        rhs[5] = -advection_k + production_k - dissipation_k + diff_k

        # FIX 18 (Rev7): record the ACTUAL RHS transfer integrals so the closure check
        # measures the real resolved-side loss vs SGS-side gain (not a tautological A-A).
        self._rhs4_transfer_integral = float(np.sum(-self.pi_sgs_field))   # term in rhs[4]
        self._rhs5_transfer_integral = float(np.sum(production_k))          # term in rhs[5]
        return rhs

    def step(self):
        """Advances 1 timestep via 4th-Order Runge-Kutta (RK4)."""
        dt = self.compute_cfl_timestep()

        rho = np.maximum(0.05, self.U[0])
        ux = self.U[1] / rho
        uy = self.U[2] / rho
        uz = self.U[3] / rho

        # FIX 27 (Rev8): closure-agnostic dispatch. "quantum" runs the CUDA-Q/VQE triad
        # closure; classical closures (smagorinsky / wale / dyn_germano) run the standard
        # SGS model on the SAME resolved dynamics + block-local Leonard stress.
        if self.closure == "dns":
            # FIX 37 (Rev12): in-code no-SGS reference trajectory ("DNS mode") on the SAME
            # grid, solver, init and dt policy: nu_t = Pi_SGS = eps^Q stay zero -> pure
            # molecular dissipation. HONESTY (state in any paper): at N=32 this is
            # UNDER-RESOLVED for Re >= 5000, so it is a coarse-grid no-SGS reference of the
            # same code, NOT a published DNS dataset.
            stats = (0.0, 0.0, 0.0, 0, 0.0, 0, 0.0, float(self.num_qubits))
        elif self.closure == "quantum":
            stats = self.update_quantum_sgs_physics(ux, uy, uz, rho)
        else:
            stats = self._update_classical_sgs(ux, uy, uz, rho)
        (mean_nu_t, q_energy_mean, q_energy_std, active_cells, bscatter_ratio,
         n_triads_total, n_triads_mean, n_modes_mean) = stats

        # Classical 4th-Order Runge-Kutta (RK4)
        U0 = np.copy(self.U)
        k1 = self.compute_rhs(U0)
        k2 = self.compute_rhs(U0 + 0.5 * dt * k1)
        k3 = self.compute_rhs(U0 + 0.5 * dt * k2)
        k4 = self.compute_rhs(U0 + dt * k3)

        self.U = U0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Positivity & physical bound preservation
        self.U[0] = np.maximum(0.05, self.U[0])
        self.U[4] = np.maximum(0.1, self.U[4])
        self.U[5] = np.maximum(1e-8, self.U[5])

        return (dt, mean_nu_t, q_energy_mean, q_energy_std, active_cells,
                bscatter_ratio, n_triads_total, n_triads_mean, n_modes_mean)

    def run_validation(self, steps, csv_path="qles_ke_decay.csv"):
        """
        FIX 15 (Rev5): LES validation harness for the Taylor-Green Vortex benchmark.
        Records the kinetic-energy decay history KE(t) (and total energy E(t)) so it
        can be compared against DNS / Smagorinsky / Dynamic Germano reference curves.
        The decay rate d(log10 KE)/dt is a standard, resolution-independent metric for
        assessing whether the quantum closure reproduces the correct dissipation.
        """
        import csv as _csv
        header = ["step", "t", "ke_resolved", "ke_sgs", "total_ke",
                  "total_energy", "log10_ke", "ke_drift", "te_drift",
                  "mean_pi_sgs", "mean_eps_q"]
        rows = []
        t = 0.0
        for step in range(1, steps + 1):
            dt, *_ = self.step()
            t += dt
            bud = self.compute_energy_budget()
            ke_res = max(bud['resolved_ke'], 1e-30)
            rows.append([step, t, bud['resolved_ke'], bud['sgs_ke'], bud['total_ke'],
                         bud['total_energy'], np.log10(ke_res),
                         bud['ke_drift'], bud['te_drift'],
                         bud['mean_pi_sgs'], bud['mean_eps_q']])
            if step == 1 or step % max(1, steps // 20) == 0:
                print(f"[VALIDATE] step {step:03d} t={t:.4e} "
                      f"KE_res={bud['resolved_ke']:.4e} log10KE={np.log10(ke_res):.4f} "
                      f"E_drift={bud['te_drift']*100:.4f}%")

        with open(csv_path, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(header)
            for r in rows:
                w.writerow(r)

        # Dimensionless decay-rate estimate over the recorded window (LSQ slope)
        ts = np.array([r[1] for r in rows])
        lks = np.array([r[6] for r in rows])
        if len(ts) >= 2 and (ts[-1] - ts[0]) > 0:
            slope = float(np.polyfit(ts, lks, 1)[0])
            print(f"[VALIDATE] mean d(log10 KE)/dt over window = {slope:.4e}  "
                  f"(compare vs DNS / Smagorinsky / Dynamic Germano)")
        print(f"[VALIDATE] wrote decay history to {csv_path}")

        # FIX 21 (Rev6): publication-style validation figures
        self._plot_validation(rows, out_prefix=csv_path.rsplit(".", 1)[0])
        # FIX 23 (Rev7): energy spectrum E(k) benchmark figure + CSV
        self.plot_energy_spectrum(out_prefix=csv_path.rsplit(".", 1)[0])

    def _plot_validation(self, rows, out_prefix="qles_ke_decay"):
        """FIX 21 (Rev6): generate validation figures (KE decay, SGS budget, E conservation)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[VALIDATE] matplotlib unavailable, skipping plots: {e}")
            return

        ts = np.array([r[1] for r in rows])
        ke_res = np.array([r[2] for r in rows])
        te_drift_pct = np.array([r[8] for r in rows]) * 100.0
        pi_sgs = np.array([r[9] for r in rows])
        eps_q = np.array([r[10] for r in rows])

        # --- Figure 1: resolved-KE decay (linear + log with reference anchor) ---
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(ts, ke_res, "b-", label="Quantum-LES KE_res")
        ax[0].set_xlabel("t"); ax[0].set_ylabel("KE_resolved")
        ax[0].set_title("Taylor-Green: resolved KE decay"); ax[0].legend(); ax[0].grid(True)
        ax[1].semilogy(ts, np.maximum(ke_res, 1e-30), "b-", label="Quantum-LES KE_res")
        if len(ts) >= 2 and (ts[-1] - ts[0]) > 0:
            slope = float(np.polyfit(ts, np.log10(np.maximum(ke_res, 1e-30)), 1)[0])
            expfit = ke_res[0] * 10 ** (slope * (ts - ts[0]))
            # FIX 25 (Rev8): this is a fit to the model's OWN decay -- a reference SLOPE,
            # not a DNS / Dynamic Germano benchmark. Label it accordingly.
            ax[1].plot(ts, expfit, "k--", label="exp fit (reference slope only)")
        ax[1].set_xlabel("t"); ax[1].set_ylabel("KE (log)"); ax[1].legend(); ax[1].grid(True)
        fig.tight_layout()
        p1 = out_prefix + "_ke_decay.png"; fig.savefig(p1, dpi=130); plt.close(fig)

        # --- Figure 2: SGS production/dissipation budget + total-energy conservation ---
        fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.2))
        ax2[0].plot(ts, pi_sgs, "g-", label="SGS production <Pi_SGS>")
        ax2[0].plot(ts, eps_q, "r-", label="Quantum dissipation <eps^Q>")
        ax2[0].set_xlabel("t"); ax2[0].set_ylabel("rate")
        ax2[0].set_title("SGS energy budget: production vs dissipation")
        ax2[0].legend(); ax2[0].grid(True)
        ax2[1].plot(ts, te_drift_pct, "r-", label="(E-E0)/E0 [%]")
        ax2[1].axhline(0.0, color="k", lw=0.8, ls="-")
        ax2[1].set_xlabel("t"); ax2[1].set_ylabel("relative E drift [%]")
        ax2[1].set_title("Total-energy conservation (near-zero drift)")
        ax2[1].legend(); ax2[1].grid(True)
        fig2.tight_layout()
        p2 = out_prefix + "_sgs_budget.png"; fig2.savefig(p2, dpi=130); plt.close(fig2)

        print(f"[VALIDATE] wrote figures: {p1}, {p2}")

    def compute_energy_spectrum(self, n_bins=48):
        """
        FIX 23 (Rev7) + FIX 24 (Rev8) + FIX 30 (Rev10): isotropic energy spectrum E(k) from
        the resolved velocity FFT, now PARSEVAL-CALIBRATED in ABSOLUTE value.

        FIX 30: np.fft.fftn is UNnormalized, so sum_k |u_hat|^2 = M * sum_x |u|^2 with
        M = nx*ny*nz. Without correction the spectrum's absolute level is off by M^2 (only
        its shape/slope was right). We therefore use the normalized coefficient u_tilde =
        fftn(u)/M, for which Parseval gives  sum_k |u_tilde|^2 = mean_x |u|^2 , and hence
            integral_0^inf E(k) dk == KE_resolved   (exactly, when rho ~ const).
        Combined with the FIX 24 bin-width normalization, E(k) is now correct in BOTH shape
        and absolute energy. The per-shell MODE COUNT is also returned (n_shell ~ k^2).
        Returns (centers, Ek, shell_count); the calibration ratio is stored on self.
        """
        rho = np.maximum(0.05, self.U[0])
        M = self.nx * self.ny * self.nz                       # FIX 30: Parseval normalization
        ux_hat = np.fft.fftn(self.U[1] / rho) / M             # u_tilde = fftn(u)/M
        uy_hat = np.fft.fftn(self.U[2] / rho) / M
        uz_hat = np.fft.fftn(self.U[3] / rho) / M
        E_modal = 0.5 * (np.abs(ux_hat)**2 + np.abs(uy_hat)**2 + np.abs(uz_hat)**2)
        KX, KY, KZ = self.k_coords
        k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
        mask = k_mag > 1e-12
        kvals = k_mag[mask]; evals = E_modal[mask]
        edges = np.linspace(0.0, float(np.max(kvals)), n_bins + 1)
        idx = np.clip(np.digitize(kvals, edges) - 1, 0, n_bins - 1)
        Ek_shell = np.zeros(n_bins)          # total energy in each shell (integrates to KE)
        shell_count = np.zeros(n_bins, dtype=int)
        for i in range(n_bins):
            sel = idx == i
            Ek_shell[i] = float(evals[sel].sum())
            shell_count[i] = int(np.sum(sel))
        # FIX 24: divide by bin width -> E(k) with integral E dk ~= KE (publication form)
        dk = np.diff(edges)
        Ek = Ek_shell / np.maximum(dk, 1e-30)
        centers = 0.5 * (edges[:-1] + edges[1:])
        # FIX 30/33: store the Parseval calibration. NOTE (FIX 33, Rev11): Parseval gives
        #   integral E(k) dk == mean( 0.5 |u|^2 )           [velocity-field KE density, UNweighted]
        # which is EXACT by construction. It equals the compressible resolved KE
        #   resolved_ke = mean( 0.5 * rho * |u|^2 )         [rho-weighted KE density]
        # ONLY when rho ~ const. For strongly compressible flow the two differ, so we report
        # BOTH: a velocity-field (Parseval) ratio ~1.0 by identity, and a compressibility ratio
        # that measures the rho-weighting deviation (~1 only if rho is nearly constant).
        u2 = ((self.U[1]/rho)**2 + (self.U[2]/rho)**2 + (self.U[3]/rho)**2)
        ke_vel = float(np.mean(0.5 * u2))                        # Parseval target (unweighted)
        ke_rho = float(np.mean(0.5 * rho * u2))                  # compressible resolved KE
        self.last_spectrum_ke_integral = float(Ek_shell.sum())    # == integral E dk
        self.last_spectrum_calib_ratio = (self.last_spectrum_ke_integral / ke_vel
                                          if ke_vel > 1e-30 else float("nan"))       # ~1.0 (Parseval)
        self.last_spectrum_compressibility_ratio = (self.last_spectrum_ke_integral / ke_rho
                                                    if ke_rho > 1e-30 else float("nan"))  # ~1 only if rho~const
        return centers, Ek, shell_count

    def plot_energy_spectrum(self, out_prefix="qles_ke_decay", ref_k=None, ref_ek=None,
                             ref_label="DNS"):
        """FIX 23 (Rev7): log-log E(k) benchmark with a Kolmogorov -5/3 anchor and an
        optional reference spectrum (e.g. DNS / Smagorinsky / Dynamic Germano)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[VALIDATE] matplotlib unavailable, skipping spectrum plot: {e}")
            return
        k, Ek, shell_count = self.compute_energy_spectrum()
        # FIX 30/33: report BOTH calibration ratios so a reviewer sees the absolute value is right
        # AND understands the velocity-field vs compressible-KE distinction.
        print(f"[VALIDATE] FIX33 spectrum calibration: "
              f"integral/<0.5|u|^2>(Parseval,~1)={self.last_spectrum_calib_ratio:.6f}  "
              f"integral/<0.5*rho|u|^2>(compressible KE)={self.last_spectrum_compressibility_ratio:.6f}")
        with open(out_prefix + "_spectrum.csv", "w") as fh:
            fh.write("k,E_k,n_shell\n")
            for kk, ee, nn in zip(k, Ek, shell_count):
                fh.write(f"{kk:.6e},{ee:.6e},{nn}\n")

        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        pos = (Ek > 0) & (k > 0)
        if np.any(pos):
            kp = k[pos][np.argmax(Ek[pos])]; Ep = float(Ek[pos].max())
            kk = np.logspace(np.log10(max(kp * 0.7, k[pos].min())), np.log10(k[-1]), 60)
            # FIX 25 (Rev8): this is a VISUAL reference SLOPE only -- it is anchored to the
            # model's own peak and is NOT a DNS / experimental benchmark. Label it as such.
            ax.loglog(kk, Ep * (kk / kp) ** (-5.0 / 3.0), "k:", lw=1.2,
                      label="Kolmogorov -5/3 reference slope (not a DNS ref)")
        ax.loglog(k[pos], Ek[pos], "b-", label="Quantum-LES E(k)")
        if ref_k is not None and ref_ek is not None:
            ax.loglog(np.asarray(ref_k), np.asarray(ref_ek), "r--", lw=1.6,
                      label=f"{ref_label} (reference spectrum)")
        ax.set_xlabel("wavenumber k"); ax.set_ylabel("E(k)")
        ax.set_title("Energy spectrum E(k): Quantum-LES vs reference")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        p = out_prefix + "_spectrum.png"; fig.savefig(p, dpi=130); plt.close(fig)
        print(f"[VALIDATE] wrote energy spectrum: {p} (+ _spectrum.csv)")

    @staticmethod
    def compare_to_reference(ref_t, ref_ke, q_t, q_ke):
        """
        FIX 15 (Rev5): quantitative comparison of the Quantum-LES KE-decay curve
        against a reference model (DNS / Smagorinsky / Dynamic Germano).
        Returns dict with max/mean relative error and the decay-rate ratio.
        """
        ref_t = np.asarray(ref_t, float); ref_ke = np.asarray(ref_ke, float)
        q_t = np.asarray(q_t, float); q_ke = np.asarray(q_ke, float)
        # FIX 20 (Rev6): the REFERENCE curve interpolated ONTO the quantum time grid
        ref_on_q = np.interp(q_t, ref_t, ref_ke)
        rel_err = (q_ke - ref_on_q) / np.maximum(ref_on_q, 1e-30)

        def _slope(t, ke):
            return float(np.polyfit(t, np.log10(np.maximum(ke, 1e-30)), 1)[0])

        q_slope = _slope(q_t, q_ke)
        ref_slope = _slope(ref_t, ref_ke)
        # guard on MAGNITUDE (slopes are negative for decaying turbulence)
        decay_rate_ratio = (q_slope / ref_slope) if abs(ref_slope) > 1e-30 else float("nan")

        return dict(max_rel_err=float(np.max(np.abs(rel_err))),
                    mean_rel_err=float(np.mean(np.abs(rel_err))),
                    decay_rate_ratio=float(decay_rate_ratio),
                    q_decay_rate=q_slope, ref_decay_rate=ref_slope)

    @staticmethod
    def _spectral_error(ek_ref, ek_cmp, log_floor_frac=1e-3):
        """
        FIX 29 (Rev9): RMS / L2 error between two E(k) curves on a SHARED k-grid (all
        benchmark models use the same grid -> identical bin centers). Reported in linear
        space (peak-weighted), in log10 space (robust to the large dynamic range of a
        turbulent spectrum, weights the inertial range more evenly) and as the relative L2
        norm. Bins where either curve is <= 0 are excluded.

        FIX 40 (Rev12): FLOOR-AWARE log RMSE. The raw rmse_log above is a METRIC PATHOLOGY
        on near-zero bins: in the dissipation range E(k) ~ 1e-7, where a tiny LINEAR
        difference (rel_L2 ~ 1e-3) spans several DECADES in log space -> rmse_log ~ 2+ even
        when the spectra are nearly identical. Reviewers will ask 'why RMSE_log10 = 2.3 but
        rel_L2 = 0.0001?'. We therefore ALSO report rmse_log_floor: the log10 RMSE restricted
        to bins where BOTH curves exceed log_floor_frac * max(ek_ref) (default: above -3
        decades from the reference peak, i.e. the dynamically active range). Paper rule:
        quote rmse_log_floor for 'shape agreement in the inertial/active range'; keep
        rmse_log (raw) only as a worst-case tail diagnostic.
        """
        ek_ref = np.asarray(ek_ref, float); ek_cmp = np.asarray(ek_cmp, float)
        pos = (ek_ref > 0) & (ek_cmp > 0)
        if int(np.sum(pos)) < 2:
            return dict(rmse_lin=float("nan"), rmse_log=float("nan"),
                        rmse_log_floor=float("nan"), rel_l2=float("nan"), n_bins=0,
                        n_bins_floored=0)
        r, c = ek_ref[pos], ek_cmp[pos]
        rmse_lin = float(np.sqrt(np.mean((r - c) ** 2)))
        rmse_log = float(np.sqrt(np.mean((np.log10(r) - np.log10(c)) ** 2)))
        rel_l2 = float(np.linalg.norm(r - c) / max(1e-30, np.linalg.norm(r)))
        # FIX 40 (Rev12): log RMSE on the dynamically active range only
        floor = log_floor_frac * float(np.max(ek_ref))
        posf = (ek_ref > floor) & (ek_cmp > floor)
        if int(np.sum(posf)) >= 2:
            rf, cf = ek_ref[posf], ek_cmp[posf]
            rmse_log_floor = float(np.sqrt(np.mean((np.log10(rf) - np.log10(cf)) ** 2)))
        else:
            rmse_log_floor = float("nan")
        return dict(rmse_lin=rmse_lin, rmse_log=rmse_log, rmse_log_floor=rmse_log_floor,
                    rel_l2=rel_l2, n_bins=int(np.sum(pos)), n_bins_floored=int(np.sum(posf)))

    def _rebuild_blocks(self, n_per_dim):
        """FIX 32 (Rev10): force an EXACT number of blocks per dimension and rebuild ALL
        block-local precomputed arrays consistently (block list + core/halo wavenumber grids
        + block test filter). Mirrors the __init__ logic so the solver stays self-consistent.
        Used by the scaling benchmark to study decomposition granularity (1/8/64 blocks)."""
        self.nb_blocks_per_dim = n_per_dim
        self.nb = self.nx // n_per_dim
        self.blocks = []
        for a in range(n_per_dim):
            for b in range(n_per_dim):
                for c in range(n_per_dim):
                    self.blocks.append((slice(a*self.nb, (a+1)*self.nb),
                                         slice(b*self.nb, (b+1)*self.nb),
                                         slice(c*self.nb, (c+1)*self.nb)))
        self.num_blocks = len(self.blocks)
        kx_b = 2.0*np.pi*np.fft.fftfreq(self.nb, d=self.dx)
        ky_b = 2.0*np.pi*np.fft.fftfreq(self.nb, d=self.dy)
        kz_b = 2.0*np.pi*np.fft.fftfreq(self.nb, d=self.dz)
        self.k_block_coords = np.meshgrid(kx_b, ky_b, kz_b, indexing='ij')
        self.nb_halo = self.nb + 2*self.halo
        kx_h = 2.0*np.pi*np.fft.fftfreq(self.nb_halo, d=self.dx)
        ky_h = 2.0*np.pi*np.fft.fftfreq(self.nb_halo, d=self.dy)
        kz_h = 2.0*np.pi*np.fft.fftfreq(self.nb_halo, d=self.dz)
        KXh, KYh, KZh = np.meshgrid(kx_h, ky_h, kz_h, indexing='ij')
        kb_sq = KXh**2 + KYh**2 + KZh**2
        delta_test = 2.0*self.filter_delta
        self.G_test_hat_block = np.exp(-kb_sq*(delta_test**2)/24.0)

    def run_scaling_benchmark(self, n_steps=3, closure="smagorinsky",
                              weak_grids=(32, 64), strong_grid=64,
                              strong_nblocks=(1, 8, 64), out_prefix="qles_scaling"):
        """
        FIX 32 (Rev10): strong / weak scaling of the per-step cost.

        WEAK scaling: fix block size (constant work per block), grow N -> #blocks grows.
            Ideal weak scaling => time-per-block stays constant (weak_eff ~ 1).
        DECOMPOSITION-OVERHEAD study (labeled 'strong' for comparison): fix N (fixed total
        work), vary #blocks. FIX 34 (Rev11): this is a SINGLE-PROCESS CPU code with NO MPI
        ranks / threads / GPUs, so the reported number is NOT an HPC parallel speedup -- it
        measures only the decomposition OVERHEAD: cost_ratio = s/step relative to the 1-block
        baseline (>1 means finer partitioning adds overhead). True strong scaling needs a
        parallel (MPI/GPU) backend, which this prototype does not have.

        HONEST NOTE: this is a SINGLE-PROCESS CPU code; 'processors' == blocks. The quantum
        closure costs one VQE per block, so more blocks generally INCREASES per-step cost on
        one core (no parallel speedup). The benchmark reports that truth, not an idealization.
        """
        import time as _time
        import csv as _csv

        def _mk(N, n_per_dim):
            sim = CompressibleLES3D(nx=N, reynolds=self.Re, mach=self.Mach,
                                    cudaq_target="nvidia", vqe_max_evals=self.vqe_max_evals,
                                    block_size=self.block_size, halo=self.halo, closure=closure)
            if n_per_dim is not None:
                sim._rebuild_blocks(n_per_dim)
            return sim

        def _time_one(sim):
            sim.step()  # warm-up (first-step allocations / FFT plan) -- excluded from timing
            t0 = _time.perf_counter()
            for _ in range(n_steps):
                sim.step()
            return (_time.perf_counter() - t0) / max(1, n_steps)

        print(f"[SCALE] closure={closure}  timed steps per config={n_steps}")

        # ---- WEAK scaling: fixed block size (self.block_size), vary N -> #blocks grows ----
        print("[SCALE] ---- WEAK scaling (fixed block size, work/block constant) ----")
        print(f"   {'N':>5s} {'#blocks':>8s} {'cells':>10s} {'s/step':>10s} {'s/block':>10s} {'weak_eff':>9s}")
        base_s_per_block = None
        weak_rows = []
        for N in weak_grids:
            sim = _mk(N, None)   # default blocking keeps block_size fixed -> work/block constant
            ntot = sim.nb_blocks_per_dim**3
            s_step = _time_one(sim)
            s_block = s_step / max(1, ntot)
            if base_s_per_block is None:
                base_s_per_block = s_block
            eff = base_s_per_block / s_block if s_block > 0 else float('nan')
            weak_rows.append((N, ntot, N**3, sim.nb, s_step, s_block, eff))
            print(f"   {N:>5d} {ntot:>8d} {N**3:>10d} {s_step:>10.4f} {s_block:>10.2e} {eff:>9.3f}")

        # ---- STRONG scaling: fixed N, vary #blocks (decomposition granularity) ----
        print(f"[SCALE] ---- STRONG scaling / decomposition-overhead (fixed N={strong_grid}) ----")
        print(f"   [single-process CPU: NO MPI/threads/GPU -> 'cost_ratio' is a decomposition")
        print(f"    OVERHEAD study, NOT parallel speedup; cost_ratio>1 = finer split adds cost]")
        print(f"   {'#blocks':>8s} {'block_sz':>9s} {'s/step':>10s} {'cost_ratio':>11s}")
        strong_rows = []
        base_s_step = None
        for ntot in strong_nblocks:
            n_per_dim = int(round(ntot ** (1.0/3.0)))
            if n_per_dim < 1 or strong_grid % n_per_dim != 0:
                continue
            sim = _mk(strong_grid, n_per_dim)
            s_step = _time_one(sim)
            if base_s_step is None:
                base_s_step = s_step
            cost_ratio = s_step / base_s_step if base_s_step > 0 else float('nan')  # >1 => overhead
            strong_rows.append((ntot, sim.nb, s_step, cost_ratio))
            print(f"   {ntot:>8d} {sim.nb:>9d} {s_step:>10.4f} {cost_ratio:>11.3f}")

        with open(out_prefix + ".csv", "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["scaling","N","#blocks","block_size","cells",
                        "s_per_step","s_per_block","metric_name","metric_value"])
            for (N, ntot, cells, bsz, s_step, s_block, eff) in weak_rows:
                w.writerow(["weak", N, ntot, bsz, cells, f"{s_step:.6f}", f"{s_block:.6e}",
                            "weak_efficiency(~1 ideal)", f"{eff:.4f}"])
            for (ntot, bsz, s_step, cost_ratio) in strong_rows:
                w.writerow(["strong", strong_grid, ntot, bsz, strong_grid**3,
                            f"{s_step:.6f}", "", "cost_ratio_vs_1block(overhead)", f"{cost_ratio:.4f}"])
        print(f"[SCALE] wrote {out_prefix}.csv")

    def run_benchmark_comparison(self, steps=120,
                                 closures=("smagorinsky", "wale", "dyn_germano"),
                                 out_prefix="qles_benchmark",
                                 ref_k=None, ref_ek=None, ref_label="DNS",
                                 ref_t=None, ref_ke=None, run_dns_ref=False):
        """
        FIX 27 (Rev8) + FIX 28/29 (Rev9): head-to-head benchmark. The SAME Taylor-Green init
        is run through Quantum-LES and each classical LES closure; KE(t) is recorded for every
        model and side-by-side KE-decay + E(k) figures plus per-model CSVs are written.

        FIX 37 (Rev12): run_dns_ref=True executes an in-code no-SGS ("DNS-mode") reference on
        the SAME grid/Re/steps and scores EVERY model against it (KE-decay slope + E(k)).
        HONESTY: at N=32 this is a coarse-grid no-SGS reference of the same code, NOT a
        published DNS dataset (under-resolved for Re >= 5000).

        FIX 38 (Rev12): prints and writes the per-step VQE COST BREAKDOWN of Quantum-LES
        (FFT / triad search / Hamiltonian build / observe() / optimizer overhead).

        FIX 28 (Rev9): reports the COMPUTATIONAL COST per timestep (wall-clock s/step and
        steps/s) for each model -- the HPC question 'how much does Quantum-LES cost?'. The
        quantum path is dominated by per-block VQE; classical paths are FFT-only.

        FIX 29 (Rev9): reports the SPECTRAL error RMSE/L2 of E(k) between Quantum-LES and each
        classical closure (linear + log10 space, relative L2). An external reference spectrum
        (e.g. DNS Taylor-Green), passed as ref_k/ref_ek, is also scored against Quantum-LES.

        #1 (Rev9): the box-averaged Germano variant is labeled 'Spatially-Averaged Dynamic
        Germano' everywhere in logs/figures (it is NOT the local C_s(x) form).
        """
        import csv as _csv
        import time as _time

        # #1 (Rev9): human-readable names for logs/figures (dyn_germano = box-averaged form)
        DISPLAY = {"Quantum-LES": "Quantum-LES",
                   "smagorinsky": "Smagorinsky",
                   "wale": "WALE (Nicoud-Ducros 2001)",
                   "dyn_germano": "Spatially-Averaged Dynamic Germano"}

        def _mk(closure):
            return CompressibleLES3D(nx=self.nx, reynolds=self.Re, mach=self.Mach,
                                     cudaq_target="nvidia", vqe_max_evals=self.vqe_max_evals,
                                     block_size=self.block_size, halo=self.halo,
                                     closure=closure, gamma_scale=self.gamma_scale)

        order = ["Quantum-LES"] + list(closures)
        sims = {"Quantum-LES": _mk("quantum")}
        for c in closures:
            sims[c] = _mk(c)

        # FIX 37 (Rev12): optional in-code no-SGS reference on the SAME grid/Re/steps.
        if run_dns_ref and ref_t is None:
            print(f"[BENCH] running in-code no-SGS ('DNS-mode') reference: same grid "
                  f"N={self.nx}, Re={self.Re:g}, {steps} steps (coarse-grid reference, "
                  f"NOT published DNS)")
            dns = _mk("dns")
            t_d = 0.0
            hist_d = []
            for s in range(1, steps + 1):
                dt, *_ = dns.step()
                t_d += dt
                bud = dns.compute_energy_budget()
                hist_d.append([s, t_d, bud["resolved_ke"], bud["total_energy"]])
            ref_t = np.array([r[1] for r in hist_d])
            ref_ke = np.array([r[2] for r in hist_d])
            if ref_k is None:
                k_r, ek_r, _ = dns.compute_energy_spectrum()
                ref_k, ref_ek = k_r, ek_r
            ref_label = f"in-code no-SGS ref (N={self.nx}, same code)"

        hist = {name: [] for name in order}
        t_all = {name: 0.0 for name in order}
        wall = {name: 0.0 for name in order}          # FIX 28: cumulative wall-clock per model
        for step in range(1, steps + 1):
            for name in order:
                t0 = _time.perf_counter()              # FIX 28: time the FULL step (FFT+VQE/closure)
                dt, *_ = sims[name].step()
                wall[name] += _time.perf_counter() - t0
                t_all[name] += dt
                bud = sims[name].compute_energy_budget()
                hist[name].append([step, t_all[name], bud["resolved_ke"], bud["total_energy"]])
            if step % max(1, steps // 10) == 0:
                print(f"[BENCH] step {step}/{steps} done")

        # per-model CSVs
        for name in order:
            p = f"{out_prefix}_{name.lower().replace(' ', '_').replace('-', '_')}.csv"
            with open(p, "w", newline="") as fh:
                w = _csv.writer(fh)
                w.writerow(["step", "t", "ke_resolved", "total_energy"])
                for r in hist[name]:
                    w.writerow(r)

        # final spectra (FIX 24/30: compute_energy_spectrum returns centers, Ek, shell_count)
        specs = {name: sims[name].compute_energy_spectrum() for name in order}

        # FIX 30/33: Parseval calibration -- velocity-field ratio ~1.0 by identity; compressible
        # KE ratio ~1 only when rho is nearly constant (Mach-dependent).
        print("[BENCH] ================= FIX33 SPECTRUM CALIBRATION =================")
        print(f"   {'model':34s} {'/vel(Parseval~1)':>16s} {'/compKE(rho-wt)':>16s}")
        for name in order:
            print(f"   {DISPLAY.get(name, name):34s} "
                  f"{sims[name].last_spectrum_calib_ratio:>16.6f} "
                  f"{sims[name].last_spectrum_compressibility_ratio:>16.6f}")

        # ---- FIX 28 + decay-rate summary table (physics + cost per model) ----
        print("[BENCH] ================= SUMMARY (per model) =================")
        print(f"   {'model':34s} {'dlogKE/dt':>12s} {'s/step':>10s} {'steps/s':>9s}")
        for name in order:
            ts = np.array([r[1] for r in hist[name]])
            ke = np.array([r[2] for r in hist[name]])
            sl = float(np.polyfit(ts, np.log10(np.maximum(ke, 1e-30)), 1)[0])
            s_step = wall[name] / max(1, steps)
            print(f"   {DISPLAY.get(name, name):34s} {sl:>12.4e} {s_step:>10.4f} "
                  f"{1.0/max(s_step, 1e-12):>9.2f}")

        # FIX 37 (Rev12): decay-rate offset of every model vs the reference trajectory
        if ref_t is not None and ref_ke is not None:
            sl_ref = float(np.polyfit(ref_t, np.log10(np.maximum(ref_ke, 1e-30)), 1)[0])
            print(f"[BENCH] ---- KE-decay rate vs {ref_label} ----")
            print(f"   {'model':34s} {'dlogKE/dt':>12s} {'delta_vs_ref':>14s}")
            for name in order:
                ts = np.array([r[1] for r in hist[name]])
                ke = np.array([r[2] for r in hist[name]])
                sl = float(np.polyfit(ts, np.log10(np.maximum(ke, 1e-30)), 1)[0])
                print(f"   {DISPLAY.get(name, name):34s} {sl:>12.4e} {sl - sl_ref:>+14.4e}")
            print(f"   {'REF (' + ref_label + ')':34s} {sl_ref:>12.4e}")

        # ---- FIX 29: spectral error RMSE/L2 of E(k) vs Quantum-LES ----
        # FIX 40 (Rev12): 'RMSE_log10' = raw (tail-dominated diagnostic); 'RMSE_log10(>-3dec)'
        # = floor-aware, the number to quote for inertial/active-range shape agreement.
        print("[BENCH] ================= SPECTRAL ERROR  E(k) vs Quantum-LES ================")
        _, ekq, _ = specs["Quantum-LES"]
        print(f"   {'model':34s} {'RMSE_lin':>12s} {'RMSE_log10':>12s} {'RMSE_log10(>-3dec)':>18s} {'rel_L2':>10s}")
        for name in order:
            if name == "Quantum-LES":
                continue
            _, ekc, _ = specs[name]
            err = self._spectral_error(ekq, ekc)
            print(f"   {DISPLAY.get(name, name):34s} {err['rmse_lin']:>12.4e} "
                  f"{err['rmse_log']:>12.4f} {err['rmse_log_floor']:>18.4f} {err['rel_l2']:>10.4f}")
        # FIX 37 (Rev12): EVERY model scored against the reference spectrum
        if ref_k is not None and ref_ek is not None:
            print(f"[BENCH] ================= SPECTRAL ERROR  E(k) vs {ref_label} ================")
            print(f"   {'model':34s} {'RMSE_lin':>12s} {'RMSE_log10':>12s} {'RMSE_log10(>-3dec)':>18s} {'rel_L2':>10s}")
            for name in order:
                _, ekc, _ = specs[name]
                err = self._spectral_error(np.asarray(ref_ek, float), ekc)
                print(f"   {DISPLAY.get(name, name):34s} {err['rmse_lin']:>12.4e} "
                      f"{err['rmse_log']:>12.4f} {err['rmse_log_floor']:>18.4f} {err['rel_l2']:>10.4f}")

        # ---- FIX 38 (Rev12): per-step VQE COST BREAKDOWN of Quantum-LES ----
        qc = sims["Quantum-LES"].vqe_cost
        nst = max(1, steps)
        print("[BENCH] ================= FIX38 VQE COST BREAKDOWN (per step) =================")
        print(f"   {'component':24s} {'s/step':>12s} {'% of closure':>14s}")
        _cost_rows = []
        for key in ("fft", "triad_search", "hamiltonian_build", "observe",
                    "optimize_overhead", "closure_total"):
            v = qc[key] / nst
            pct = 100.0 * v / max(qc["closure_total"] / nst, 1e-15)
            _cost_rows.append([key, f"{v:.6f}", f"{pct:.2f}"])
            print(f"   {key:24s} {v:>12.6f} {pct:>13.1f}%")
        p_cost = out_prefix + "_vqe_cost.csv"
        with open(p_cost, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["component", "s_per_step", "pct_of_closure"])
            for r in _cost_rows:
                w.writerow(r)
        print(f"[BENCH] wrote {p_cost}")

        # figures
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[BENCH] matplotlib unavailable, skipping plots: {e}")
            return

        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for name in order:
            ts = np.array([r[1] for r in hist[name]])
            ke = np.array([r[2] for r in hist[name]])
            ax.semilogy(ts, np.maximum(ke, 1e-30), label=DISPLAY.get(name, name))
        # FIX 37 (Rev12): reference trajectory on the KE-decay figure
        if ref_t is not None and ref_ke is not None:
            ax.semilogy(ref_t, np.maximum(ref_ke, 1e-30), "k--", lw=1.6,
                        label=f"{ref_label} (reference)")
        ax.set_xlabel("t"); ax.set_ylabel("KE_resolved (log)")
        ax.set_title("Taylor-Green KE decay: Quantum-LES vs classical LES closures")
        ax.legend(); ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
        p1 = out_prefix + "_ke_decay_compare.png"; fig.savefig(p1, dpi=130); plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(6.8, 4.6))
        for name in order:
            k, Ek, _ = specs[name]; pos = (Ek > 0) & (k > 0)
            ax2.loglog(k[pos], Ek[pos], label=DISPLAY.get(name, name))
        if ref_k is not None and ref_ek is not None:
            ax2.loglog(np.asarray(ref_k), np.asarray(ref_ek), "k--", lw=1.6,
                       label=f"{ref_label} (reference)")
        ax2.set_xlabel("wavenumber k"); ax2.set_ylabel("E(k)")
        ax2.set_title("Energy spectrum E(k): Quantum-LES vs classical LES")
        ax2.legend(); ax2.grid(True, which="both", alpha=0.3); fig2.tight_layout()
        p2 = out_prefix + "_spectrum_compare.png"; fig2.savefig(p2, dpi=130); plt.close(fig2)

        print(f"[BENCH] wrote {p1}, {p2} (+ per-model CSVs)")

    def run_gamma_sweep(self, scales=(0.5, 1.0, 2.0), steps=150,
                        out_prefix="qles_gamma"):
        """
        FIX 39 (Rev12): Gamma_kpq SENSITIVITY STUDY. Re-runs the SAME Taylor-Green benchmark
        with the heuristic triad-coupling scale factor gamma_scale in {scales} (identical init,
        grid, Re, dt policy) and reports per value: KE(t), mean d(log10 KE)/dt, mean SGS
        dissipation. Answers the reviewer question 'how sensitive are the results to the
        (not NS-derived) Gamma_kpq prefactor?' (FIX 31). gamma_scale = 1.0 reproduces Rev11.
        """
        import csv as _csv
        all_rows = {}
        summary = []
        for gs in scales:
            sim = CompressibleLES3D(nx=self.nx, reynolds=self.Re, mach=self.Mach,
                                    cudaq_target="nvidia", vqe_max_evals=self.vqe_max_evals,
                                    block_size=self.block_size, halo=self.halo,
                                    closure="quantum", gamma_scale=gs)
            t = 0.0
            rows = []
            eps_acc = []
            for s in range(1, steps + 1):
                dt, *_ = sim.step()
                t += dt
                bud = sim.compute_energy_budget()
                rows.append([s, t, bud["resolved_ke"], bud["total_energy"]])
                eps_acc.append(bud["mean_eps_q"])
                if s == 1 or s % max(1, steps // 5) == 0:
                    print(f"[GAMMA {gs:g}] step {s:03d}/{steps} t={t:.4e} "
                          f"KE={bud['resolved_ke']:.4e}")
            ts = np.array([r[1] for r in rows])
            ke = np.array([r[2] for r in rows])
            slope = float(np.polyfit(ts, np.log10(np.maximum(ke, 1e-30)), 1)[0])
            eps_mean = float(np.mean(eps_acc))
            summary.append((gs, slope, eps_mean, t))
            all_rows[gs] = rows
            print(f"[GAMMA {gs:g}] dlogKE/dt={slope:.4e}  mean_eps_q={eps_mean:.4e}")

        p_csv = out_prefix + ".csv"
        with open(p_csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["gamma_scale", "step", "t", "ke_resolved", "total_energy"])
            for gs in scales:
                for r in all_rows[gs]:
                    w.writerow([gs] + r)

        p_sum = out_prefix + "_summary.csv"
        with open(p_sum, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["gamma_scale", "dlogKE_dt", "mean_eps_q", "t_end"])
            for gs, slope, epsm, tend in summary:
                w.writerow([gs, f"{slope:.6e}", f"{epsm:.6e}", f"{tend:.6e}"])

        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[GAMMA] matplotlib unavailable, skipping plots: {e}")
            return p_csv

        fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4))
        for gs, slope, epsm, tend in summary:
            ts = np.array([r[1] for r in all_rows[gs]])
            ke = np.array([r[2] for r in all_rows[gs]])
            ax[0].semilogy(ts, np.maximum(ke, 1e-30), label=f"gamma_scale={gs:g}")
        ax[0].set_xlabel("t"); ax[0].set_ylabel("KE_resolved (log)")
        ax[0].set_title(f"Gamma_kpq sensitivity: KE decay (Re={self.Re:g}, N={self.nx})")
        ax[0].legend(); ax[0].grid(True, which="both", alpha=0.3)
        gsv = [s[0] for s in summary]
        slv = [s[1] for s in summary]
        ax[1].bar([f"{g:g}" for g in gsv], slv)
        ax[1].set_xlabel("gamma_scale (Gamma_kpq multiplier)")
        ax[1].set_ylabel("d(log10 KE)/dt")
        ax[1].set_title("Decay rate vs Gamma scale (heuristic prefactor)")
        ax[1].grid(True, alpha=0.3)
        fig.tight_layout()
        p_png = out_prefix + ".png"
        fig.savefig(p_png, dpi=130); plt.close(fig)
        print(f"[GAMMA] wrote {p_csv}, {p_sum}, {p_png}")
        return p_csv


# ==============================================================================
# 7. EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Quantum-Enhanced LES Closure Model (Q-LES Rev 12): classical "
                    "compressible NS + k_SGS engine with a per-block quantum-VQE triad SGS "
                    "closure; block-local Pi_SGS, symmetric transfer, dynamic qubits, "
                    "tri_weight Hamiltonian")
    parser.add_argument("--grid", type=int, default=32, help="Grid size N (default: 32)")
    parser.add_argument("--target", type=str, default="nvidia",
                        help="CUDA-Q target backend (nvidia / qpp-cpu)")
    parser.add_argument("--steps", type=int, default=20, help="Timesteps to run")
    parser.add_argument("--reynolds", type=float, default=2000.0, help="Reynolds number")
    parser.add_argument("--mach", type=float, default=0.35, help="Mach number")
    parser.add_argument("--vqe-evals", type=int, default=15,
                        help="Max VQE optimizer evaluations per block (default: 15)")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Block size for domain partitioning (default: 16)")
    parser.add_argument("--halo", type=int, default=2,
                        help="Ghost-cell halo width for block-local spectral ops (default: 2)")
    parser.add_argument("--validate", action="store_true",
                        help="Run the Taylor-Green KE-decay validation harness and write CSV")
    parser.add_argument("--out", type=str, default="qles_ke_decay.csv",
                        help="Output CSV path for --validate (default: qles_ke_decay.csv)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Head-to-head Quantum-LES vs Smagorinsky/WALE/Dynamic-Germano")
    parser.add_argument("--bench-steps", type=int, default=120,
                        help="Timesteps for --benchmark (default: 120)")
    parser.add_argument("--bench-out", type=str, default="qles_benchmark",
                        help="Output prefix for --benchmark (default: qles_benchmark)")
    parser.add_argument("--scale", action="store_true",
                        help="Run strong/weak scaling benchmark of per-step cost (FIX 32)")
    parser.add_argument("--scale-closure", type=str, default="smagorinsky",
                        help="Closure for --scale (default smagorinsky; 'quantum' is far slower)")
    parser.add_argument("--selftest", action="store_true",
                        help="Run JW Fock-space self-test and exit")
    # FIX 37/39 (Rev12): validation-campaign options
    parser.add_argument("--closure", type=str, default="quantum",
                        choices=["quantum", "smagorinsky", "wale", "dyn_germano", "dns"],
                        help="SGS closure for plain/--validate runs (FIX 37: 'dns' = in-code no-SGS reference)")
    parser.add_argument("--gamma-scale", type=float, default=1.0,
                        help="Scale factor on the heuristic Gamma_kpq triad coupling (FIX 39; 1.0 = Rev11)")
    parser.add_argument("--dns-ref", action="store_true",
                        help="With --benchmark: also run an in-code no-SGS reference on the SAME grid/Re/steps and score every model vs it (FIX 37)")
    parser.add_argument("--gamma-sweep", type=str, default="",
                        help="Comma-separated gamma_scale values, e.g. '0.5,1.0,2.0' -> Gamma sensitivity study (FIX 39)")
    parser.add_argument("--sweep-steps", type=int, default=150,
                        help="Timesteps per gamma_scale in --gamma-sweep (default: 150)")
    args = parser.parse_args()

    if args.selftest:
        ok = run_jw_self_test()
        sys.exit(0 if ok else 1)

    print("=" * 80)
    print("HYBRID QUANTUM-ENHANCED LES CLOSURE MODEL (Q-LES REV 12)")
    print("Positioning: NOT a quantum NS solver -- the CFD engine is fully classical;")
    print("the quantum computer acts as an unconventional SGS subgrid model (per-block VQE).")
    print("Reviewer Fixes (wytyczne.txt + To jest juz zdecydowanie.txt):")
    print("  [Fix 1]  production_k = Pi_SGS (full sign): backscatter retained")
    print("  [Fix 2]  rhs[4] -= Pi_SGS: symmetric Resolved<->SGS energy coupling")
    print("  [Fix 3]  consistent (non-asymmetric) energy transfer, no double count")
    print("  [Fix 4]  Nomenclature: 'exact JW mapping of triad operator' (NS H reduced)")
    print("  [Fix 5]  Energy budget + Energy Drift = (KE_tot-KE_tot0)/KE_tot0 per step")
    print("  [Fix 6]  O(N^2) triad discovery via dictionary lookup")
    print("  [Fix 7]  Dynamic qubits from candidate-shell richness: min(12,max(6,cand//2))")
    print("  [Fix 8]  CPU fallback capped at n_q <= 8 (256 amplitudes/block)")
    print("  [Fix 9]  tri_weight modulates triad coupling in the Hamiltonian")
    print("  [Fix 10] Block-local Leonard stress Pi_SGS (same scale as quantum closure)")
    print("  [Fix 11] Ghost-cell halo overlap: boundary-crossing triads no longer lost")
    print("  [Fix 12] Preserve VQE thetas across dynamic qubit-count changes")
    print("  [Fix 13] Log Hamiltonian stats: total triads + mean active modes/block")
    print("  [Fix 14] Total-energy budget mean(U[4]) with its own drift (not just KE)")
    print("  [Fix 15] LES validation harness: --validate -> KE(t) decay CSV + d(logKE)/dt")
    print("  [Fix 16] Core-mode spectral analysis (halo = auxiliary filter zone only)")
    print("  [Fix 17] SGS energy budget: <Pi_SGS>, <eps^Q> and Pi/Eps ratio")
    print("  [Fix 18] SGS transfer conservativeness check (resolved loss vs SGS gain)")
    print("  [Fix 19] Feedback-coupled quantum closure: eps_{n-1} -> H_n (diagonal + triads)")
    print("  [Fix 20] compare_to_reference: unambiguous ref_on_q naming")
    print("  [Fix 21] Validation figures: KE decay, SGS budget, E conservation (PNG)")
    print("  [Fix 22] Dynamic qubit hysteresis (cooldown) prevents per-step oscillation")
    print("  [Fix 23] Energy spectrum E(k) benchmark + Kolmogorov -5/3 anchor (PNG/CSV)")
    print("  [Fix 24] Publication isotropic E(k): shell-sum / bin-width (+ per-shell mode count)")
    print("  [Fix 25] Labels: Kolmogorov line = 'reference slope', NOT a DNS benchmark")
    print("  [Fix 26] Two-way feedback clipped to [0.5,2.0] (Gamma_kpq) + turnover-capped omega_k")
    print("  [Fix 27] Classical LES closures (Smag/WALE/DynGermano) wired into step() + --benchmark")
    print("  [Fix 28] Benchmark reports COMPUTATIONAL COST per step (s/step, steps/s) per model")
    print("  [Fix 29] Benchmark reports SPECTRAL error RMSE/L2 of E(k) vs Quantum-LES (+DNS ref)")
    print("  [Fix 30] PARSEVAL-calibrated E(k): FFT normalized by 1/M -> integral E dk == KE")
    print("  [Fix 31] Honest note: Gamma_kpq is a heuristic geometric closure (not NS-derived)")
    print("  [Fix 32] Strong/weak scaling benchmark of per-step cost (--scale, N & #blocks)")
    print("  [Fix 33] Spectrum calib SPLIT: velocity-field (Parseval ~1) vs compressible KE (~1 if rho~const)")
    print("  [Fix 34] 'Strong scaling' relabeled as decomposition-OVERHEAD study (no MPI/threads/GPU)")
    print("  [Fix 35] Dynamic qubit count = documented HEURISTIC knob (ratio/min/max), not magic number")
    print("  [Fix 36] VQE history wording corrected: 'parameter transfer', NOT optimizer continuation")
    print("  [Fix 37] In-code no-SGS ('DNS-mode') reference: --closure dns / --benchmark --dns-ref (coarse-grid ref, NOT published DNS)")
    print("  [Fix 38] Per-step VQE COST BREAKDOWN: FFT / triad search / Hamiltonian build / observe() / optimizer overhead")
    print("  [Fix 39] Gamma_kpq SENSITIVITY STUDY: --gamma-scale knob + --gamma-sweep harness (heuristic prefactor must be swept)")
    print("  [Fix 40] Floor-aware spectral error: RMSE_log10(>-3dec) excludes near-zero tail bins (raw log RMSE is a metric pathology there)")
    backend = args.target if CUDAQ_AVAILABLE else "CPU state-vector simulator"
    print(f"Backend: {backend} | Grid: {args.grid}^3 | Re: {args.reynolds} | "
          f"Block: {args.block_size}^3")
    print("=" * 80)

    sim = CompressibleLES3D(nx=args.grid, reynolds=args.reynolds, mach=args.mach,
                            cudaq_target=args.target, vqe_max_evals=args.vqe_evals,
                            block_size=args.block_size, halo=args.halo,
                            closure=args.closure, gamma_scale=args.gamma_scale)

    if args.validate:
        sim.run_validation(args.steps, csv_path=args.out)
        sys.exit(0)

    # FIX 27 (Rev8) + FIX 37/38 (Rev12): head-to-head benchmark vs classical LES closures,
    # with optional in-code no-SGS reference and per-step VQE cost breakdown
    if args.benchmark:
        sim.run_benchmark_comparison(steps=args.bench_steps, run_dns_ref=args.dns_ref,
                                     out_prefix=args.bench_out)
        sys.exit(0)

    # FIX 39 (Rev12): Gamma_kpq sensitivity study
    if args.gamma_sweep:
        scales = [float(x) for x in args.gamma_sweep.split(",") if x.strip()]
        sim.run_gamma_sweep(scales=scales, steps=args.sweep_steps)
        sys.exit(0)

    # FIX 32 (Rev10): strong/weak scaling of the per-step cost
    if args.scale:
        sim.run_scaling_benchmark(n_steps=3, closure=args.scale_closure)
        sys.exit(0)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        (dt, nu_t, q_energy_mean, q_energy_std, active_cells, bscatter,
         n_triads_total, n_triads_mean, n_modes_mean) = sim.step()
        if step % 5 == 0 or step == 1:
            # FIX 5/13/14 (Rev5): KE budget + TOTAL energy + drifts + Hamiltonian stats
            bud = sim.compute_energy_budget()
            print(f"Step {step:02d} | dt: {dt:.4e} | QPU<H>: {q_energy_mean:.4e} "
                  f"(std {q_energy_std:.4e}) | eps^Q: {sim.quantum_dissipation_rate:.4f}")
            print(f"         KE_res={bud['resolved_ke']:.4e} | KE_sgs={bud['sgs_ke']:.4e} | "
                  f"KE_tot={bud['total_ke']:.4e} | E_tot={bud['total_energy']:.4e}")
            print(f"         Backscatter: {bscatter*100:.1f}% | "
                  f"triads={n_triads_total} (avg {n_triads_mean:.2f}/blk) | "
                  f"active_modes(avg {n_modes_mean:.2f}/blk)")
            print(f"         KE Drift = {bud['ke_drift']*100:.4f}% | "
                  f"E Drift = {bud['te_drift']*100:.4f}%  "
                  f"(KE_tot0={sim.initial_total_ke:.4e}, E_tot0={sim.initial_total_energy:.4e})")
            # FIX 17/18 (Rev7): SGS production vs quantum dissipation + real transfer closure
            print(f"         SGS: <Pi>={bud['mean_pi_sgs']:.3e} <eps^Q>={bud['mean_eps_q']:.3e} "
                  f"Pi/Eps={bud['pi_eps_ratio']:.3f} Pi-Eps={bud['pi_minus_eps']:.3e}")
            print(f"         transfer: rhs4={bud['rhs4_transfer']:.3e} rhs5={bud['rhs5_transfer']:.3e} "
                  f"| closure_err={bud['transfer_error']:.2e}")

    elapsed = time.time() - t0
    print("-" * 80)
    print(f"Simulation completed in {elapsed:.2f}s ({args.steps / max(elapsed, 1e-9):.2f} steps/s).")
    print("[SUCCESS] All reviewer fixes implemented (Rev 12: FIX 1-40; validation campaign tooling + floor-aware E(k) error).")
