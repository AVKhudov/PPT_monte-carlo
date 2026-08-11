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


def placed_cells_ionization_rk4(atoms, t_array):
    number_of_atoms = len(atoms)
    fully_ionized_mask = np.zeros(number_of_atoms, dtype=bool)

    local_ionization_array = np.zeros((len(t_array), cfg.z_max),
                                      dtype=int)  # массив с данными о пот-х ионизации электронов
    ion_times = np.full((number_of_atoms, cfg.z_max), cfg.t_0 + 1.0)
    # массив с зависимостями зар-в атомов/ионов от времени

    electron_motion_array = np.zeros((cfg.z_max * number_of_atoms, 5))
    motion_counter = 0

    start_ionization_time = tm.perf_counter()
    for i, current_moment in enumerate(t_array):
        progress = (i + 1) / len(t_array) * 100
        print(f'\rИонизация: {i + 1}/{len(t_array)} ({progress:.1f}%)', end='')

        # =ИОНИЗАЦИЯ=
        random_values = np.random.random(number_of_atoms)
        probabilities = fast_ion.w_ppt_numba(
            current_moment,
            atoms,
            cfg.ionization_potentials,
            cfg.n_star_array,
            cfg.c_n_l_array,
            cfg.b_l_m_array,
            cfg.delta_t,
            cfg.w_0,
            cfg.eps,
            cfg.right_or_left,
            cfg.atomic_field
        )

        ionization_mask = (random_values < probabilities) & (~fully_ionized_mask)
        ionization_indices = np.where(ionization_mask)[0]

        if ionization_indices.size == 0:
            continue

        charges = atoms[ionization_indices, 6].astype(int)  # заряды АТОМНЫХ ОСТАТКОВ

        fully_ionized_mask_local = charges == cfg.z_max

        fully_ionized_indices = ionization_indices[fully_ionized_mask_local]
        not_fully_ionized_indices = ionization_indices[~fully_ionized_mask_local]

        if fully_ionized_indices.size > 0:
            fully_ionized_mask[fully_ionized_indices] = True

            ion_times[fully_ionized_indices, cfg.z_max - 1] = current_moment
            local_ionization_array[i, cfg.z_max - 1] += 1

            number_fully = fully_ionized_indices.size
            electron_motion_array[motion_counter:motion_counter + number_fully, 0] = current_moment
            electron_motion_array[motion_counter:motion_counter + number_fully, 1] = atoms[fully_ionized_indices, 6]
            electron_motion_array[motion_counter:motion_counter + number_fully, 2:] = atoms[fully_ionized_indices, :3]
            motion_counter += number_fully

        if not_fully_ionized_indices.size > 0:
            not_fully_charges = atoms[not_fully_ionized_indices, 6].astype(int)

            ion_times[not_fully_ionized_indices, not_fully_charges - 1] = current_moment
            local_ionization_array[i, not_fully_charges - 1] += 1

            number_not_fully = not_fully_ionized_indices.size
            electron_motion_array[motion_counter:motion_counter + number_not_fully, 0] = current_moment
            electron_motion_array[motion_counter:motion_counter + number_not_fully, 1] = \
                atoms[not_fully_ionized_indices, 6]
            electron_motion_array[motion_counter:motion_counter + number_not_fully, 2:] = \
                atoms[not_fully_ionized_indices, :3]
            motion_counter += number_not_fully

            atoms[not_fully_ionized_indices, 6:] = cfg.ionization_order_array[not_fully_charges]

    end_ionization_time = tm.perf_counter()
    print(f"Ионизация завершена, время: {end_ionization_time - start_ionization_time:.6f} сек")
    print('\n')

    # Расчет импульса ионов в статичном режиме
    print(f"Расчёт ионов...")
    start_time = tm.perf_counter()

    atoms[:, 3:6] = fast_ion.integrate_all_ions_numba(
        atoms[:, :3], ion_times, t_array,
        cfg.w_0, cfg.eps, cfg.right_or_left, cfg.a_0_ion, cfg.start_charge_number_of_particles
    )

    end_time = tm.perf_counter()
    print(f"Расчёт ионов закончен, время: {end_time - start_time:.6f} сек")

    electron_motion_data = electron_motion_array[:motion_counter]
    amount_of_electrons = motion_counter
    amount_of_fully_ionized_states = np.sum(fully_ionized_mask)
    fully_ionized_indices_general = np.where(fully_ionized_mask)[0]

    # electron_motion_data содержит в качестве элементов массивы:
    # [время рождения электрона, его сорт (1-й п-л ионизации, 2-й, 3-й, ..., последний), место рождения (x, y, z)]
    return atoms, local_ionization_array, electron_motion_data, amount_of_electrons, \
           amount_of_fully_ionized_states, fully_ionized_indices_general


def placed_cells_ionization_rk4_numba(atoms, t_array):
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
        cfg.ionization_order_array,
        cfg.z_max
    )

    end_ionization_time = tm.perf_counter()
    print(f"Ионизация завершена, время: {end_ionization_time - start_ionization_time:.6f} сек")

    print(f"Расчёт ионов...")
    start_time = tm.perf_counter()

    atoms[:, 3:6] = fast_ion.integrate_all_ions_numba(
        atoms[:, :3], ion_times, t_array,
        cfg.w_0, cfg.eps, cfg.right_or_left, cfg.a_0_ion, cfg.start_charge_number_of_particles
    )

    end_time = tm.perf_counter()
    print(f"Расчёт ионов завершен, время: {end_time - start_time:.6f} сек")

    electron_motion_data = electron_motion_array[:motion_counter]
    amount_of_electrons = motion_counter
    amount_of_fully_ionized_states = np.sum(fully_ionized_mask)
    fully_ionized_indices_general = np.where(fully_ionized_mask)[0]

    return atoms, local_ionization_array, electron_motion_data, amount_of_electrons, \
           amount_of_fully_ionized_states, fully_ionized_indices_general


def pulse_energy_estimation():  # Оценка энергии пучка. Интегрируем плотность потока энергии (в-р Пойнтинга)
    # по сечению мишени x=0 и по времени (см. ниже)
    a = cfg.form[1] / 2.  # пределы интегрирования поперек мишени
    b = 3  # интегрирование по времени от -b * cfg.tau до b * cfg.tau
    gauss_energy = cfg.gauss_field ** 2 * cfg.gauss_wavelength ** 3
    joule_energy = gauss_energy * cfg.one_erg_in_joules

    def time_function(t):
        return np.cos(t) ** 2 * np.exp(-2 * t ** 2 / cfg.tau ** 2)

    def spatial_function(x):
        return np.exp(-2 * x ** 2 / cfg.w_0 ** 2)

    time_result = quad(time_function, 0, b * cfg.tau)[0]
    spatial_result = quad(spatial_function, 0, a * cfg.w_0)[0]

    pulse_energy = joule_energy / np.pi ** 2 * time_result * spatial_result ** 2

    return pulse_energy
