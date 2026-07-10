# Laterally Loaded Pile — Interactive *p–y* Analysis

A small Streamlit app for laterally loaded pile / *p–y* analysis, written as
teaching material for an EGBC short course on numerical modeling of
soil-structure interaction.

Given pile parameters, a soil / *p–y* curve, and a head boundary condition, it
runs a 1D finite element solver (Euler-Bernoulli beam on a nonlinear Winkler
foundation) and plots:

- Response profiles: deflection, slope, moment, shear, and soil reaction vs. depth
- Head load–deflection design curve
- Head load–maximum-moment design curve

Two *p–y* curves are supported, selectable in the app (with an optional
side-by-side comparison overlay):

- **Hyperbolic**: `p = y / (1/kh_int + |y|/pult)`
- **Matlock (1970) soft clay**: `p/pult = 0.5*(y/y50)^(1/3)`, with `pult`
  varying properly with depth (the shallow-wedge mechanism growing with depth,
  capped by the deep flow-around mechanism)

Two head boundary conditions are supported: free head (shear + moment) and
fixed head (shear with zero head rotation).

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files

- `streamlit_app.py` — the app's UI
- `pile_solver.py` — the FEM solver (no UI dependency); also usable standalone,
  e.g. `python3 pile_solver.py` runs a small self-test against known reference
  values
