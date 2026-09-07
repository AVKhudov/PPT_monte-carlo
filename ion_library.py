import ion_config as cfg
import ion_numba as fast_ion
import numpy as np
import time as tm

'''
ARGON 18. Библиотечный файл.
'''


def placing_cells_with_ions_motion(number, start_charge_number):
    """
    :param number: число атомов в симуляции
    :param start_charge_number: начальный заряд (1 - ионизуем нейтральные атомы, 2 - ионы 1+ и т. д.)
    :return: массив размерности (number, 10), 10 - координаты, импульс, 4 параметра
    """
    placed_cells = np.empty((number, 10))

    positions = np.random.uniform(-0.5, 0.5, (number, 3)) * cfg.form
    positions[:, 0] = np.where(np.abs(positions[:, 0]) < 1e-10, 1e-6, positions[:, 0])
    momenta = np.zeros((number, 3))
    params = cfg.ionization_order_array[start_charge_number]

    placed_cells[:, :3] = positions
    placed_cells[:, 3:6] = momenta
    placed_cells[:, 6:] = params

    return placed_cells


def placed_cells_ionization_rk4_numba(atoms, t_array, count_momenta=False):
    number_of_atoms = len(atoms)
    fully_ionized_mask = np.zeros(number_of_atoms, dtype=bool)

    local_ionization_array = np.zeros((len(t_array), cfg.z_max), dtype=int)
    ion_times = np.full((number_of_atoms, cfg.z_max), cfg.t_0 + 1.0)
    electron_motion_array = np.zeros((cfg.z_max * number_of_atoms, 5))

    print('Ионизация...')
    start_ionization_time = tm.perf_counter()

    motion_counter = fast_ion.simulate_ionization_numba(
        atoms,

        t_array,
        ion_times,
        local_ionization_array,
        electron_motion_array,
        fully_ionized_mask,
        cfg.ionization_potentials,
        cfg.n_star_array,
        cfg.c_n_l_array,
        cfg.b_l_m_array,

        cfg.delta_t,
        cfg.w_0,
        cfg.eps,
        cfg.right_or_left,
        cfg.atomic_field,
        cfg.tau,

        cfg.ionization_order_array,
        cfg.z_max
    )

    end_ionization_time = tm.perf_counter()
    print(f"Ионизация завершена, время: {end_ionization_time - start_ionization_time:.6f} сек")

    if count_momenta:
        print(f"Расчёт ионов...")
        start_time = tm.perf_counter()

        atoms[:, 3:6] = fast_ion.integrate_all_ions_numba(
            atoms[:, :3], ion_times, t_array,
            cfg.w_0, cfg.eps, cfg.right_or_left, cfg.a_0_ion, cfg.tau, cfg.start_charge_number_of_particles
        )

        end_time = tm.perf_counter()
        print(f"Расчёт ионов завершен, время: {end_time - start_time:.6f} сек")

    electron_motion_data = electron_motion_array[:motion_counter]
    amount_of_electrons = motion_counter
    amount_of_fully_ionized_states = np.sum(fully_ionized_mask)
    fully_ionized_indices_general = np.where(fully_ionized_mask)[0]

    return atoms, local_ionization_array, electron_motion_data, amount_of_electrons, \
           amount_of_fully_ionized_states, fully_ionized_indices_general
