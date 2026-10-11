# Dataset card

- Solver version: 0.1.0; physics: isothermal rigid-viscoplastic (Norton-Hoff), regularised Tresca friction, axisymmetric half ring, no remeshing.
- Training box: {'od_id': (1.6, 2.4), 'h_id': (0.4, 1.0), 'mf': (0.0, 1.0), 'm': (0.05, 0.3)}; out-of-distribution geometry: {'od_id': (2.4, 2.8), 'h_id': (1.0, 1.25)}; target reduction 0.35.
- Runs: 13 planned, 13 ok, 0 failed, 2 stopped early by fold-over (kept, flagged).
- Splits: train: 9, test_random: 1, test_ood_geometry: 2, test_fine_mesh: 1
- Worst volume drift: 0.66%.
- Known limits: no remeshing (sticking runs stop near 30% reduction); no temperature; flat dies; 2D axisymmetric only, so no ovality.
- Force is normalised by K (V/h0)^m A0; shape results are independent of K and V (checked).
- Newton: 0 solves did not reach the 1e-8 tolerance within the iteration cap (see each run's warnings).
- IMPORTANT: QoIs in this index are at each run's FINAL reduction, which differs for runs that stopped early. Build ML targets from the per-step curves at fixed reductions instead.
- Inner diameter: id_change_pct is at mid-height; id_min_change_pct is the minimum over the inner surface (the two differ by a few points when the inner surface barrels).
