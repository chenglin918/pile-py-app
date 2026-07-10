"""
pile_solver.py
------------------------------------------------------------------
1D finite element solver for a laterally loaded pile modelled as an
Euler-Bernoulli beam on a (nonlinear) Winkler foundation -- i.e. the
p-y method solved by FEM. Standalone module backing both interactive
apps -- the Jupyter/ipywidgets app (pile_py_app.ipynb) and the
browser app (streamlit_app.py) -- so the two share one FEM
implementation and one results-export format. See the course notebook
("EGBC Course Notes - Numerical Modeling of SSI.ipynb", Modules 1-3)
for the underlying theory and derivations.

Governing equation (no axial load, no distributed load):

        E_p I_p  y'''' + k_h(y) y = 0

    y      = lateral deflection (m)
    E_p I_p= flexural rigidity EI (kN.m^2)
    k_h(y) = secant modulus of subgrade reaction (kN/m^2)

Element : 2-node Hermite-cubic beam, DOFs [y1, theta1, y2, theta2].
Soil    : consistent Winkler foundation matrix, element-averaged
          secant modulus (Picard/secant iteration on k_h).

p-y curves supported (`curve` argument):
  - "elastic"    : constant kh = kh_int (linear).
  - "hyperbolic" : p = y / (1/kh_int + |y|/pult)                  (Eq. 2)
  - "matlock"    : p/pult = 0.5*(y/y50)^(1/3), capped at 8*y50    (Eq. 3)
                   y50 = 2.5*eps50*D                               (Eq. 4)
                   pult(z) = min(pult1(z), pult2)                  (Eq. 5)
                   -- evaluated at each element's own depth z, so
                   pult grows near the surface and saturates at
                   9*Cu*D at depth, matching Matlock (1970) directly
                   (no separate soil-layering input is needed: the
                   depth-dependence is already inside Eq. 5's z terms).

Head boundary condition (`head_bc` argument):
  - "free"  : prescribed shear Vt and moment Mt at the head (node 0).
  - "fixed" : prescribed shear Vt, zero head rotation (theta_0 = 0);
              Mt is ignored (the reaction moment is whatever the
              restrained system produces -- it is not a free input).
Tip (x=L) is always left unrestrained ("long pile" assumption: the
response has decayed to ~0 well before the tip) -- short-pile/floating
-tip conditions are out of scope for this solver.
------------------------------------------------------------------
"""
import numpy as np


# ----------------------------------------------------------------------
# Element matrices
# ----------------------------------------------------------------------
def beam_bending_matrix(EI, Le):
    """4x4 Euler-Bernoulli bending stiffness for one element."""
    L = Le
    return (EI / L**3) * np.array([
        [ 12,    6*L,   -12,    6*L],
        [ 6*L,  4*L*L,  -6*L,  2*L*L],
        [-12,   -6*L,    12,   -6*L],
        [ 6*L,  2*L*L,  -6*L,  4*L*L],
    ])


def winkler_foundation_matrix(kh, Le):
    """4x4 consistent Winkler foundation matrix (= kh * integral N^T N)."""
    L = Le
    return (kh * L / 420.0) * np.array([
        [156,    22*L,    54,   -13*L],
        [22*L,   4*L*L,   13*L, -3*L*L],
        [54,     13*L,    156,  -22*L],
        [-13*L, -3*L*L,  -22*L,  4*L*L],
    ])


# ----------------------------------------------------------------------
# p-y curve: hyperbolic     p = y / (1/kh_int + |y|/pult)          (Eq. 2)
# secant modulus            k_sec(y) = p/y = 1/(1/kh_int + |y|/pult)
# ----------------------------------------------------------------------
def py_curve(y, kh_int, pult):
    """Hyperbolic p-y reaction, direct p(y) form (Eq. 2)."""
    return y / (1.0 / kh_int + np.abs(y) / pult)


def secant_modulus(y, kh_int, pult):
    """Secant modulus for the hyperbolic curve, kh(y) = p(y)/y (Eq. 2)."""
    return 1.0 / (1.0 / kh_int + np.abs(y) / pult)


# ----------------------------------------------------------------------
# p-y curve: Matlock (1970) soft clay                       (Eqs. 3-5)
# ----------------------------------------------------------------------
def matlock_y50(eps50, D):
    """Deflection at half the ultimate resistance, Eq. (4): y50 = 2.5*eps50*D."""
    return 2.5 * eps50 * D


def matlock_pult(z, Cu, D, gamma_prime, J=0.5):
    """Ultimate soil resistance per length at depth z, Eq. (5): the smaller
    of the shallow-wedge (pult1, governs near the surface) and flow-around
    (pult2, governs at depth) mechanisms. Returns (pult, pult1, pult2)."""
    pult1 = (3 + (gamma_prime / Cu) * z + J * (z / D)) * Cu * D
    pult2 = 9 * Cu * D
    return min(pult1, pult2), pult1, pult2


def matlock_py_curve(y, pult, y50):
    """Matlock (1970) soft-clay p-y reaction, direct p(y) form (Eq. 3):
    p/pult = 0.5*(y/y50)^(1/3), held constant at pult once |y| > 8*y50."""
    p = pult * 0.5 * np.sign(y) * np.abs(y / y50) ** (1.0 / 3.0)
    return np.clip(p, -pult, pult)


def matlock_secant_modulus(y, pult, y50, y_floor_frac=1e-6):
    """Secant modulus for Matlock's curve, kh(y) = p(y)/y (Eq. 3). The
    curve's initial slope is infinite (p ~ y^(1/3)), so kh -> inf as
    y -> 0; |y| is floored at a small fraction of y50 to keep the first
    Picard iteration (which starts from y = 0) finite."""
    y_abs = max(abs(y), y_floor_frac * y50)
    p_over_pult = min(0.5 * (y_abs / y50) ** (1.0 / 3.0), 1.0)
    return p_over_pult * pult / y_abs


# ----------------------------------------------------------------------
# Assemble & solve
# ----------------------------------------------------------------------
def solve_pile(L, D, EI, n_elem, Vt, Mt=0.0, head_bc="free", curve="elastic",
               kh_int=None, pult=None,
               Cu=None, eps50=None, gamma_prime=None, J=0.5,
               tol=1e-8, max_iter=200):
    """Long-pile FEM solve. Returns (x, y, theta, M, V, p).

    L, D, EI    : pile length (m), diameter (m), flexural rigidity (kN.m^2).
    Vt, Mt      : head shear (kN) and moment (kN.m). Mt is ignored when
                  head_bc="fixed" (rotation is restrained instead).
    head_bc     : "free" (natural BC on Vt, Mt) or "fixed" (Vt applied,
                  theta_0 = 0 enforced by DOF elimination).
    curve       : "elastic" (needs kh_int), "hyperbolic" (needs kh_int,
                  pult), or "matlock" (needs Cu, eps50, gamma_prime, J;
                  pult is computed per element from its own depth).
    """
    n_node = n_elem + 1
    ndof = 2 * n_node
    Le = L / n_elem
    x = np.linspace(0.0, L, n_node)
    z_elem = 0.5 * (x[:-1] + x[1:])   # element midpoint depths

    if curve == "matlock":
        y50 = matlock_y50(eps50, D)
        pult_elem = np.array([matlock_pult(z, Cu, D, gamma_prime, J)[0] for z in z_elem])

    F = np.zeros(ndof)
    F[0] = Vt
    if head_bc == "free":
        F[1] = Mt
    fixed_dofs = [1] if head_bc == "fixed" else []
    free_dofs = [i for i in range(ndof) if i not in fixed_dofs]

    y_nodes = np.zeros(n_node)
    d = np.zeros(ndof)
    for _ in range(max_iter):
        K = np.zeros((ndof, ndof))
        for e in range(n_elem):
            dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
            if curve == "elastic":
                kh = kh_int
            else:
                y_avg = 0.5 * (y_nodes[e] + y_nodes[e+1])
                if curve == "matlock":
                    kh = matlock_secant_modulus(y_avg, pult_elem[e], y50)
                else:
                    kh = secant_modulus(y_avg, kh_int, pult)
            Ke = beam_bending_matrix(EI, Le) + winkler_foundation_matrix(kh, Le)
            for a in range(4):
                for b in range(4):
                    K[dofs[a], dofs[b]] += Ke[a, b]

        if fixed_dofs:
            K_r = K[np.ix_(free_dofs, free_dofs)]
            F_r = F[free_dofs]
            d = np.zeros(ndof)
            d[free_dofs] = np.linalg.solve(K_r, F_r)
        else:
            d = np.linalg.solve(K, F)

        y_new = d[0::2]
        if np.linalg.norm(y_new - y_nodes, 1) < tol or curve == "elastic":
            y_nodes = y_new
            break
        y_nodes = y_new

    theta = d[1::2]
    M = np.zeros(n_node)
    V = np.zeros(n_node)
    for e in range(n_elem):
        dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
        de = d[dofs]
        B0 = np.array([-6, -4*Le, 6, -2*Le]) / Le**2     # xi = 0
        B1 = np.array([ 6,  2*Le, -6, 4*Le]) / Le**2     # xi = 1
        M[e]   = EI * (B0 @ de)
        M[e+1] = EI * (B1 @ de)
        Bv = np.array([12, 6*Le, -12, 6*Le]) / Le**3     # constant per element
        V[e]   = EI * (Bv @ de)
        V[e+1] = EI * (Bv @ de)

    # direct p(y) reconstruction, using each node's own local pult for Matlock
    p = np.zeros(n_node)
    if curve == "elastic":
        p = kh_int * y_nodes
    elif curve == "hyperbolic":
        p = py_curve(y_nodes, kh_int, pult)
    else:
        pult_node = np.array([matlock_pult(zi, Cu, D, gamma_prime, J)[0] for zi in x])
        p = matlock_py_curve(y_nodes, pult_node, y50)

    return x, y_nodes, theta, M, V, p


def sweep_head_load(L, D, EI, n_elem, loads, head_bc="free", curve="elastic", **curve_kwargs):
    """Run solve_pile once per head load. Returns (loads, y_top_mm, Mmax)."""
    y_top = []
    Mmax = []
    for Vt in loads:
        _, y, _, M, _, _ = solve_pile(L, D, EI, n_elem, Vt, head_bc=head_bc,
                                       curve=curve, **curve_kwargs)
        y_top.append(y[0] * 1000.0)
        Mmax.append(np.max(np.abs(M)))
    return np.asarray(loads, dtype=float), np.asarray(y_top), np.asarray(Mmax)


# ----------------------------------------------------------------------
# Shared results-export format ("<curve>_FEM-py.txt"), used by both apps
# ----------------------------------------------------------------------
def format_export_text(name, L, D, EI, head_bc, Vt, Mt, run):
    """Build the "<curve>_FEM-py.txt" text block for one curve's results.

    `run` is a dict with keys x, y, th, M, V, p, kwargs, sweep_loads,
    sweep_y, sweep_M -- i.e. one entry of the `runs` list built by both
    apps' "Run analysis" callback.
    """
    kwargs = run["kwargs"]
    if kwargs["curve"] == "hyperbolic":
        curve_line = (f"# Curve params: kh_int={kwargs['kh_int']:.2f} kN/m^2, "
                       f"pult={kwargs['pult']:.2f} kN/m\n")
    else:
        curve_line = (f"# Curve params: Cu={kwargs['Cu']:.2f} kPa, eps50={kwargs['eps50']:.4f}, "
                       f"gamma'={kwargs['gamma_prime']:.2f} kN/m^3, J={kwargs['J']:.2f}\n")

    lines = [
        f"# FEM-py results -- {name} p-y curve\n",
        f"# Pile: L={L:.3f} m, D={D:.4f} m, EI={EI:,.0f} kN.m^2\n",
        curve_line,
        f"# Boundary condition: {head_bc} head, Vt={Vt:.3f} kN, Mt={Mt:.3f} kN.m\n",
        "#\n# --- Response profile ---\n",
        f"#{'x(m)':>11}{'y(mm)':>13}{'theta(mrad)':>15}{'M(kN.m)':>13}{'V(kN)':>13}{'p(kN/m)':>13}\n",
    ]
    for xi, yi, thi, Mi, Vi, pi in zip(run["x"], run["y"], run["th"], run["M"], run["V"], run["p"]):
        lines.append(f"{xi:12.4f}{yi*1000:13.4f}{thi*1000:15.4f}{Mi:13.4f}{Vi:13.4f}{pi:13.4f}\n")
    lines.append("#\n# --- Design curve: load sweep ---\n")
    lines.append(f"#{'Vt(kN)':>11}{'y_top(mm)':>13}{'Mmax(kN.m)':>13}\n")
    for Vi, yi, Mi in zip(run["sweep_loads"], run["sweep_y"], run["sweep_M"]):
        lines.append(f"{Vi:12.4f}{yi:13.4f}{Mi:13.4f}\n")
    return "".join(lines)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ---- Case-study pile (matches the course notebook's Module 3/5) ----
    D, L = 0.40, 20.0
    Ep = 69.2e6                        # kN/m^2
    Ip = np.pi * D**4 / 64.0
    EI = Ep * Ip
    print(f"EI = {EI:,.0f} kN.m^2  (spreadsheet lists 87,000)")

    Vt, Mt = 136.4, 0.0
    kh_int, pult = 20.0e3, 50.0

    print("\n--- Elastic mesh convergence (free head) ---")
    for n in (20, 50, 100, 400):
        x, y, th, M, V, p = solve_pile(L, D, EI, n, Vt, Mt, curve="elastic", kh_int=kh_int)
        print(f"  n={n:>3}  y_top = {y[0]*1000:.4f} mm")

    print("\n--- Nonlinear hyperbolic, free head ---")
    x, y, th, M, V, p = solve_pile(L, D, EI, 200, Vt, Mt, curve="hyperbolic",
                                    kh_int=kh_int, pult=pult)
    imax = np.argmax(np.abs(M))
    print(f"  y_top = {y[0]*1000:.3f} mm   Mmax = {M[imax]:.1f} kN.m at x = {x[imax]:.2f} m ({x[imax]/D:.1f} D)")

    print("\n--- Nonlinear Matlock (depth-varying pult), free head ---")
    Cu, eps50, gamma_prime = 80.0, 0.02, 8.0
    x, y, th, M, V, p = solve_pile(L, D, EI, 200, Vt, Mt, curve="matlock",
                                    Cu=Cu, eps50=eps50, gamma_prime=gamma_prime)
    imax = np.argmax(np.abs(M))
    print(f"  y_top = {y[0]*1000:.3f} mm   Mmax = {M[imax]:.1f} kN.m at x = {x[imax]:.2f} m ({x[imax]/D:.1f} D)")
    pult_0, pult1_0, pult2_0 = matlock_pult(0.5, Cu, D, gamma_prime)
    pult_deep, pult1_deep, pult2_deep = matlock_pult(15.0, Cu, D, gamma_prime)
    print(f"  pult at z=0.5m:  pult1={pult1_0:.1f}, pult2={pult2_0:.1f} -> pult={pult_0:.1f} kN/m")
    print(f"  pult at z=15 m:  pult1={pult1_deep:.1f}, pult2={pult2_deep:.1f} -> pult={pult_deep:.1f} kN/m")

    print("\n--- Fixed head vs free head, hyperbolic ---")
    x_free, y_free, th_free, M_free, V_free, p_free = solve_pile(
        L, D, EI, 200, Vt, Mt, head_bc="free", curve="hyperbolic", kh_int=kh_int, pult=pult)
    x_f, y_f, th_f, M_f, V_f, p_f = solve_pile(
        L, D, EI, 200, Vt, Mt, head_bc="fixed", curve="hyperbolic", kh_int=kh_int, pult=pult)
    print(f"  free head:  y_top = {y_free[0]*1000:.3f} mm, theta_0 = {th_free[0]:.6e} rad")
    print(f"  fixed head: y_top = {y_f[0]*1000:.3f} mm, theta_0 = {th_f[0]:.6e} rad")
