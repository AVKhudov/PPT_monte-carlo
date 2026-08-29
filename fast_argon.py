import os
import numpy as np

import ion_config as cfg
import ion_library as ion
import ion_numba as fast_ion
import ion_data_process_library as ion_proc

import electron_library as el_lib


def run_fast_argon_simulation():
    time_array = np.arange(-cfg.t_0, cfg.t_0, cfg.delta_t, dtype=float)

    placed_cells = ion.placing_cells_with_ions_motion(
        cfg.n_atoms,
        cfg.start_charge_number_of_particles
    )

    max_ionizations = cfg.z_max - cfg.start_charge_number_of_particles

    ionization_times, ionization_counts = fast_ion.parallel_ionization_numba(
        time_array,
        placed_cells,

        max_ionizations,

        cfg.ionization_potentials,
        cfg.ionization_order_array,
        cfg.n_star_array,
        cfg.c_n_l_array,
        cfg.b_l_m_array,

        cfg.delta_t,

        cfg.w_0,
        cfg.eps,
        cfg.right_or_left,

        cfg.atomic_field
    )

    electron_motion_input = ion.make_electron_motion_input(
        placed_cells,
        ionization_times,
        ionization_counts
    )

    number_of_electrons = np.size(electron_motion_input, axis=0)

    electrons_18_sort_idx = np.where(electron_motion_input[:, 1] == 18)[0]
    number_of_fully_ionized_states = np.size(electrons_18_sort_idx)

    fully_ionized_indices = np.where(placed_cells[:, 6] == cfg.z_max)[0]
