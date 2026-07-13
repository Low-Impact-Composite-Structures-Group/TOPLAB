# Pressure Buffer System — Sensitivity Optimisation

This directory contains the gradient-based sensitivity study for the two-tank
pressure-buffer hydrogen system used as the onboard fuel source in an ATR-72-class
hydrogen-electric aircraft.  For the earlier Cartesian sweep study see
`../pressure_buffer_sweep/`.

---

## System Description

The pressure buffer pairs two tanks through a spring-loaded pressure-compensation
valve:

| Tank | Fluid | Nominal pressure | Temperature | Role |
|------|-------|-----------------|-------------|------|
| Tank 1 | CH₂ (compressed gas) | 800 bar | 288 K | High-pressure buffer |
| Tank 2 | CcH₂ (compact cryo) | 400 bar | 53 K | Mission discharge tank |

**Valve hysteresis** (applied to every evaluated design):
- Opens when Tank 2 pressure ≤ 16 bar
- Closes when Tank 2 pressure ≥ 30 bar

**Venting pressure** (applied to every evaluated design):
- Each tank vents at 1.5 × its nominal initial pressure (1 200 bar for CH₂,
  600 bar for CcH₂)

**Mission**: ATR-72 fuel-flow profile (CSV), duration ≈ 1.025 h, assigned to
Tank 2.  Tank 1 replenishes Tank 2 via the valve.

---

## Studies

### 1. Cartesian Sweep  →  `../pressure_buffer_sweep/`

Coarse exploration over tank radius and phi (L/r aspect ratio) on a Cartesian
product grid.  Use this as a sanity-check baseline or to seed a new design
region before running the sensitivity study.

**Run:**
```bash
python optimization/pressure_buffer_sweep/driver_pressure_buffer_opt.py
```

**Output:** `../pressure_buffer_sweep/output/pressure_buffer_sweep.txt`

---

### 2. Sensitivity-Step Optimisation  (this directory)

Gradient-sign backtracking line search over three independent objectives, each
with its own set of active design variables.

#### Design Variables

| Variable | Physical meaning | Active for |
|----------|-----------------|-----------|
| `radius_scale` | Multiplier on both tank radii (baseline: r₁=0.50 m, r₂=0.586 m) | grav, vol |
| `length_scale` | Multiplier on both cylindrical lengths (baseline: phi=3.0 for both) | grav, vol |
| `insulation_scale` | Multiplier on insulation thickness; HTC scaled inversely (U ∝ 1/d) | vent time |

Both tanks are scaled by the **same** factor (coupled geometry).
Tank aspect ratio is capped at **phi\_max = 8.0** to prevent unrealistic cylinders.

Baseline geometry (from `coupled_ch2_cch2_config.yaml`):
- Tank 1: r = 0.500 m, L = 1.500 m, phi = 3.000, insulation = 50 mm G10
- Tank 2: r = 0.586 m, L = 1.759 m, phi = 3.000, insulation = 50 mm G10

#### Objectives

| Objective | Sense | Physical meaning |
|-----------|-------|-----------------|
| `gravimetric_efficiency` | max | η_g = m_fuel / (m_fuel + m_structure) |
| `volumetric_efficiency` | max | η_v = V_inner / V_outer |
| `vent_time_after_mission_s` | max | Time from full-tank dormancy start to first vent event [s] |

The three objectives are optimised **independently** — no Pareto front is
computed.  Decoupling was chosen deliberately:

- Geometry (radius, length) has near-zero measured effect on vent time at this
  scale.
- Insulation has near-zero measured effect on structural mass efficiency.
- Independent optimisation avoids wasted simulations (≈ 3× runtime saving).

#### Algorithm

At each outer iteration:

1. **Sensitivity probe** — evaluate objective at `design + initial_step` for
   each active variable (finite-difference gradient direction).
2. **Backtracking line search** — step in the gradient-sign direction with step
   size `initial_step`; if no improvement, halve the step and retry up to
   `max_backtracks` times.
3. **Accept or converge** — if any step size produces an improvement, update
   the design point and record it in history; otherwise stop early.

Key parameters (from `pressure_buffer_sensitivity.yaml`):

```
initial_step : radius/length 20 %, insulation 30 %
min_step     : radius/length  1 %, insulation  2 %
max_backtracks : 5
bounds       : radius [0.5×, 2.0×], length [0.5×, 2.5×], insulation [0.10×, 4.0×]
phi_max      : 8.0
dormancy window : 30 days (2 592 000 s)
```

#### How to Run

```bash
# Sensitivity optimisation (all three objectives)
python optimization/presure_buffer_opt/driver_pressure_buffer_sensitivity.py

# Custom config path (optional)
python optimization/presure_buffer_opt/driver_pressure_buffer_sensitivity.py \
    optimization/presure_buffer_opt/pressure_buffer_sensitivity.yaml

# Visualise results
python optimization/presure_buffer_opt/plot_sensitivity_results.py
```

---

## Output Files

| File | Contents |
|------|----------|
| `output/pressure_buffer_sensitivity_gravimetric_efficiency.txt` | Per-iteration history: geometry, discharge metrics |
| `output/pressure_buffer_sensitivity_volumetric_efficiency.txt` | Per-iteration history: geometry, discharge metrics |
| `output/pressure_buffer_sensitivity_vent_time_after_mission_s.txt` | Per-iteration history: insulation scale, dormancy vent time |
| `output/pressure_buffer_sensitivity_summary.csv` | All three objectives combined (radius\_scale, length\_scale, insulation\_scale, insulation\_thickness\_mm, efficiencies, vent time) |

---

## Results Summary  (run 2026-07-10, 10 iterations)

### Gravimetric efficiency  (maximize η_g)

| | Baseline | Best (iter 10) | Change |
|-|---------|---------------|--------|
| η_g | 0.3995 | 0.4427 | **+10.8 %** |
| radius scale | 1.00 | ~0.75 | −25 % |
| length scale | 1.00 | ~1.85 | +85 % |
| phi | 3.00 | ~7.4 (→ phi_max) | +147 % |

The optimizer consistently moves toward **smaller radius, longer tank** (higher
phi).  The physical explanation: composite wall mass ∝ r², so a given required
volume is served more efficiently by an elongated capsule than a short fat one.
The phi_max = 8 constraint was reached at iteration 5; the remaining improvement
came from the backtracking search refining the radius/length balance.

The curve is bending at iteration 10 — 2–4 more steps would confirm whether a
plateau has been reached.

### Volumetric efficiency  (maximize η_v)

| | Baseline | Best (iter 5) | Change |
|-|---------|--------------|--------|
| η_v | 0.7860 | 0.8681 | **+10.4 %** |
| radius scale | 1.00 | 2.00 (bound) | +100 % |
| length scale | 1.00 | 2.00 (bound) | +100 % |
| phi | 3.00 | 3.00 (unchanged) | 0 % |

The optimizer reaches the **radius upper bound (2.0×)** at iteration 5 and
terminates (converged to bound, not physical optimum).  Physical explanation:
wall thickness is fixed by hoop-stress design; as tank radius grows, wall
thickness becomes a proportionally smaller fraction of the total radius, so
η_v improves monotonically.  Extending the upper bound to 3.0× would continue
the improvement.

phi stays constant because radius and length scale together equally
(length\_scale = radius\_scale throughout), leaving phi = L/r unchanged.

### Vent time  (maximize dormancy before first vent)

| | Baseline | Best (iter 10) | Change |
|-|---------|---------------|--------|
| Vent time | 139.7 h | ~103 h | **−26 %** |
| Insulation scale | 1.00 (50 mm) | ~2.80 (140 mm) | +180 % |

> ⚠️ **Unexpected direction — investigation required.**
> With the correct `max` sense, the optimizer should move toward *longer* vent
> times (thicker insulation → slower heat ingress).  The previous run used
> `min` sense by mistake, so the trajectory shown above is the *opposite* of
> the intended objective.  On the next run (with `max` sense) the insulation
> should move toward the upper bound and vent time should *increase* from
> baseline — provided the HTC scaling in `_apply_insulation_scaling` is acting
> in the correct direction.  Verify with a single manual test: two dormancy
> configs differing only in HTC should produce different vent times in the
> expected direction before trusting automated results.

Sensitivity slope (linear region): approximately **4 h of dormancy per 10 %
insulation change** (based on the linear portion before ~iter 6).

---

## Known Limitations

1. **Bound-constrained outcomes** — both η_v and vent time reach design
   bounds, not physical optima.  The optimised values are engineering limits, not
   true extrema of the objective functions.

2. **Independent objectives** — no Pareto surface is computed.  Maximising η_v
   (larger tanks) and maximising η_g (smaller tanks) are in direct conflict.

3. **Coupled geometry** — both tanks scale identically.  For a tighter design
   study, Tank 1 and Tank 2 could be decoupled (4 design variables instead of 2).

4. **Insulation thermal direction** — see vent-time caveat above.

5. **Post-mission dormancy not studied** — the vent-time objective uses a
   fully-loaded pre-mission starting state.  Post-mission dormancy (depleted
   tanks, different temperatures) has a different time scale and is not covered
   by the current implementation.

6. **No feasibility penalty** — mission completion is enforced as a hard
   pass/fail.  Designs near the minimum volume boundary may fail silently.

---

## File Structure

```
presure_buffer_opt/                  ← this directory
├── README.md                        ← this file
├── __init__.py                      ← exports sensitivity API
├── driver_pressure_buffer_sensitivity.py
├── plot_sensitivity_results.py
├── pressure_buffer_sensitivity.py   ← study class + backtracking algorithm
├── pressure_buffer_sensitivity.yaml ← run parameters
└── output/
    ├── pressure_buffer_sensitivity_gravimetric_efficiency.txt
    ├── pressure_buffer_sensitivity_volumetric_efficiency.txt
    ├── pressure_buffer_sensitivity_vent_time_after_mission_s.txt
    └── pressure_buffer_sensitivity_summary.csv

../pressure_buffer_sweep/            ← legacy Cartesian sweep
├── __init__.py
├── driver_pressure_buffer_opt.py
├── pressure_buffer_sweep.py
├── pressure_buffer_sweep.yaml
└── output/
    └── pressure_buffer_sweep.txt
```
