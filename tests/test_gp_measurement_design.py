"""Grid posterior with a latent rate sensitivity m, on a synthetic forward model."""
import numpy as np

from surrogates.gp_bo import measurement_design as md


def _toy():
    mf = md.MF_GRID[None, :, None]
    m = md.M_GRID[:, None, None]
    s = md.S_FORCE[None, None, :]
    LF = 0.15 * mf * (s / 0.3) + 0.8 * m * (s / 0.3)             # m shifts the force a lot
    r = np.array(md.R_ID)[None, None, :]
    ID = 20 - (30 + 40 * m) * mf * (r / 0.3)                     # and the ID a little
    return LF, ID


def test_more_id_measurements_shrink_the_posterior_and_unknown_m_widens_it():
    LF, ID = _toy()
    i_f, i_m = 50, 4                                             # m_f = 0.25, m = 0.15
    y_force = np.log(1.07) + LF[i_m, i_f]
    one = md.posterior_std(LF, ID, y_force, [ID[i_m, i_f, 2]], [2], None)[0]
    two = md.posterior_std(LF, ID, y_force, [ID[i_m, i_f, 2], ID[i_m, i_f, 2]], [2, 2], None)[0]
    known = md.posterior_std(LF, ID, y_force, [ID[i_m, i_f, 2]], [2], i_m)[0]
    force_only = md.posterior_std(LF, ID, y_force, [], [], None)[0]
    assert two < one                                             # a repeated independent measurement helps
    assert known <= one + 1e-12                                  # knowing m cannot hurt
    assert one < force_only
    mean = md.posterior_std(LF, ID, y_force, [ID[i_m, i_f, 2]], [2], i_m)[1]
    assert abs(mean - 0.25) < 0.01
