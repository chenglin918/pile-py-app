"""
streamlit_app.py
------------------------------------------------------------------
Browser version of the laterally loaded pile / p-y app. Same inputs,
same outputs, same pile_solver.py backend as pile_py_app.ipynb (the
Jupyter/ipywidgets version) -- pick whichever front end suits how
you're working; both stay in sync because they share one solver
module and one results-export format.

Run with:
    streamlit run streamlit_app.py
------------------------------------------------------------------
"""
import time

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from pile_solver import solve_pile, sweep_head_load, matlock_pult, format_export_text

st.set_page_config(page_title="Laterally Loaded Pile -- p-y Analysis", layout="wide")

st.title("Laterally Loaded Pile — Interactive *p–y* Analysis")
st.markdown(
    "Browser version of the companion app to *EGBC Course Notes — Numerical Modeling "
    "of SSI* (Modules 1–3). Backed by the same `pile_solver.py` used by "
    "`pile_py_app.ipynb` — both apps share one FEM implementation, extended with a "
    "fixed-rotation head option and Matlock's $p_{ult}$ varying properly with depth."
)

col_pile, col_soil, col_bc = st.columns(3)

# ---------------- Pile parameters ----------------
with col_pile:
    st.subheader("Pile parameters")
    L = st.number_input("L (m)", value=20.0, min_value=0.1, step=1.0)
    D = st.number_input("D (m)", value=0.319, min_value=0.01, step=0.01, format="%.4f")
    shape = st.radio("Section", ["Solid", "Hollow (pipe)"], index=1, horizontal=True)
    if shape == "Hollow (pipe)":
        t = st.number_input("wall t (m)", value=0.0128, min_value=0.0, step=0.001, format="%.4f")
        Di = max(D - 2 * t, 0.0)
        Ip = np.pi * (D**4 - Di**4) / 64.0
    else:
        Ip = np.pi * D**4 / 64.0
    Ep = st.number_input("Ep (kN/m^2)", value=2.18e8, min_value=1.0, format="%.4e")
    EI = Ep * Ip
    st.caption(f"EI = {EI:,.0f} kN·m²")

# ---------------- Soil parameters (p-y curve) ----------------
with col_soil:
    st.subheader("Soil parameters (p-y curve)")
    curve = st.radio("Primary curve", ["Hyperbolic", "Matlock"], horizontal=True)
    compare = st.checkbox("Overlay other curve for comparison")

    def hyperbolic_inputs(key_prefix=""):
        kh_int = st.number_input("kh_int (kN/m^2)", value=20000.0, min_value=1.0,
                                  format="%.1f", key=key_prefix + "kh_int")
        pult = st.number_input("pult (kN/m)", value=50.0, min_value=0.1,
                                key=key_prefix + "pult")
        return dict(curve="hyperbolic", kh_int=kh_int, pult=pult)

    def matlock_inputs(key_prefix=""):
        Cu = st.number_input("Cu (kPa)", value=32.3, min_value=0.1, key=key_prefix + "Cu")
        eps50 = st.number_input("eps50", value=0.02, min_value=0.001, format="%.4f",
                                 key=key_prefix + "eps50")
        gamma_prime = st.number_input("gamma' (kN/m^3)", value=11.2, min_value=0.1,
                                       key=key_prefix + "gamma")
        J = st.number_input("J", value=0.5, min_value=0.0, key=key_prefix + "J")
        return dict(curve="matlock", Cu=Cu, eps50=eps50, gamma_prime=gamma_prime, J=J)

    other_name, other_kwargs = None, None
    if curve == "Hyperbolic":
        st.caption("Hyperbolic parameters (Eq. 2)")
        primary_kwargs = hyperbolic_inputs()
        if compare:
            st.caption("Matlock parameters (Eqs. 3-5) — typical eps50: 0.02 soft, "
                       "0.01 medium, 0.007 stiff clay")
            other_name, other_kwargs = "Matlock", matlock_inputs(key_prefix="cmp_")
    else:
        st.caption("Matlock parameters (Eqs. 3-5) — typical eps50: 0.02 soft, "
                   "0.01 medium, 0.007 stiff clay")
        primary_kwargs = matlock_inputs()
        if compare:
            st.caption("Hyperbolic parameters (Eq. 2)")
            other_name, other_kwargs = "Hyperbolic", hyperbolic_inputs(key_prefix="cmp_")

# ---------------- Boundary conditions + load sweep ----------------
with col_bc:
    st.subheader("Boundary conditions")
    head_choice = st.radio("Head BC", ["Free head", "Fixed head"], horizontal=True)
    head_bc = "fixed" if head_choice == "Fixed head" else "free"
    Vt = st.number_input("Vt (kN)", value=136.4, min_value=0.0)
    Mt = st.number_input("Mt (kN.m)", value=0.0, disabled=(head_bc == "fixed"))
    if head_bc == "fixed":
        Mt = 0.0
        st.caption("Mt is ignored for a fixed head (rotation is restrained instead "
                   "of the moment being prescribed).")

    st.subheader("Load sweep (design curves)")
    Vt_max = st.number_input("Sweep max Vt (kN)", value=150.0, min_value=1.0)
    n_pts = st.slider("Sweep points", 3, 20, 8)

run_clicked = st.button("Run analysis", type="primary")

if "results" not in st.session_state:
    st.session_state.results = None

if run_clicked:
    t_start = time.perf_counter()
    n_elem = 100
    curve_names = [curve]
    kwargs_map = {curve: primary_kwargs}
    colors = {curve: "tab:blue"}
    if compare:
        curve_names.append(other_name)
        kwargs_map[other_name] = other_kwargs
        colors[other_name] = "darkorange"

    runs = []
    for name in curve_names:
        kwargs = kwargs_map[name]
        x, y, th, M, V, p = solve_pile(L, D, EI, n_elem, Vt, Mt, head_bc=head_bc, **kwargs)
        runs.append({"name": name, "x": x, "y": y, "th": th, "M": M, "V": V,
                     "p": p, "kwargs": kwargs, "color": colors[name]})

    loads = np.linspace(max(Vt * 0.1, 1e-3), Vt_max, n_pts)
    for r in runs:
        loads_arr, y_arr, M_arr = sweep_head_load(L, D, EI, n_elem, loads,
                                                    head_bc=head_bc, **r["kwargs"])
        r["sweep_loads"], r["sweep_y"], r["sweep_M"] = loads_arr, y_arr, M_arr
    t_compute = time.perf_counter() - t_start

    st.session_state.results = {"L": L, "D": D, "EI": EI, "head_bc": head_bc,
                                 "Vt": Vt, "Mt": Mt, "runs": runs, "t_compute": t_compute}

results = st.session_state.results

if results is None:
    st.info("Set your parameters, then click Run analysis.")
else:
    st.success(f"### TIMING: FEM solve took {results['t_compute']*1000:.0f} ms "
               f"(server-side only, excludes network/page load)")
    t_render_start = time.perf_counter()
    runs = results["runs"]
    D_r, L_r = results["D"], results["L"]
    head_label = "fixed head" if results["head_bc"] == "fixed" else "free head"

    # ---- 5-panel response profile ----
    fig, axes = plt.subplots(1, 5, figsize=(15, 6), sharey=True)
    panels = [("y", 1000.0, r"$y$ (mm)"), ("th", 1000.0, r"$\theta$ (mrad)"),
              ("M", 1.0, r"$M$ (kN$\cdot$m)"), ("V", 1.0, r"$V$ (kN)"),
              ("p", 1.0, r"$p$ (kN/m)")]
    for ax, (key, scale, xlabel) in zip(axes, panels):
        for r in runs:
            ax.plot(r[key] * scale, r["x"], color=r["color"], label=r["name"].lower())
        ax.axvline(0, color="0.7", lw=0.7)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("depth $x$ (m), head at top")
    axes[0].invert_yaxis()
    axes[0].legend()
    fig.suptitle(f"Response profiles ($D$={D_r:.2f} m, $L$={L_r:.1f} m, "
                 f"$V_t$={results['Vt']:.1f} kN, {head_label})")
    fig.tight_layout()
    st.pyplot(fig)

    # ---- load sweep: design curves ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    for r in runs:
        axes2[0].plot(r["sweep_y"], r["sweep_loads"], "o-", color=r["color"], label=r["name"].lower())
        axes2[1].plot(r["sweep_loads"], r["sweep_M"], "o-", color=r["color"], label=r["name"].lower())
    axes2[0].set_xlabel("Head deflection (mm)")
    axes2[0].set_ylabel("Head load $V_t$ (kN)")
    axes2[0].set_title("Load-deflection design curve")
    axes2[0].grid(True)
    axes2[0].legend()
    axes2[1].set_xlabel("Head load $V_t$ (kN)")
    axes2[1].set_ylabel("$M_{max}$ (kN.m)")
    axes2[1].set_title("$M_{max}$-load design curve")
    axes2[1].grid(True)
    axes2[1].legend()
    fig2.tight_layout()
    st.pyplot(fig2)

    t_render = time.perf_counter() - t_render_start
    st.caption(f"⏱️ FEM solve: {results['t_compute']*1000:.0f} ms  |  "
               f"Plot render: {t_render*1000:.0f} ms  "
               f"(timings are for the server-side work only, not network/page load)")

    # ---- summary ----
    for r in runs:
        imax = np.argmax(np.abs(r["M"]))
        msg = (f"**{r['name']}**: head deflection = {r['y'][0]*1000:.2f} mm, "
               f"Mmax = {r['M'][imax]:.1f} kN.m at x = {r['x'][imax]:.2f} m "
               f"({r['x'][imax]/D_r:.1f} D)")
        st.write(msg)
        if r["name"] == "Matlock":
            kwargs = r["kwargs"]
            pult_shallow, _, _ = matlock_pult(D_r, kwargs["Cu"], D_r, kwargs["gamma_prime"], kwargs["J"])
            pult_deep, _, _ = matlock_pult(L_r, kwargs["Cu"], D_r, kwargs["gamma_prime"], kwargs["J"])
            st.caption(f"Matlock pult: {pult_shallow:.1f} kN/m at z=1D, "
                       f"{pult_deep:.1f} kN/m at z=L (tip)")

    # ---- export ----
    st.subheader("Export data")
    for r in runs:
        text = format_export_text(r["name"], results["L"], results["D"], results["EI"],
                                   results["head_bc"], results["Vt"], results["Mt"], r)
        st.download_button(f"Download {r['name'].lower()}_FEM-py.txt", data=text,
                            file_name=f"{r['name'].lower()}_FEM-py.txt", mime="text/plain",
                            key=f"dl_{r['name']}")
