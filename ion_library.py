from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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

    electron_motion_array = np.zeros((cfg.z_max * number_of_atoms, 4))
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
            electron_motion_array[motion_counter:motion_counter + number_fully, 1:] = atoms[fully_ionized_indices, :3]
            motion_counter += number_fully

        if not_fully_ionized_indices.size > 0:
            not_fully_charges = atoms[not_fully_ionized_indices, 6].astype(int)

            ion_times[not_fully_ionized_indices, not_fully_charges - 1] = current_moment
            local_ionization_array[i, not_fully_charges - 1] += 1

            number_not_fully = not_fully_ionized_indices.size
            electron_motion_array[motion_counter:motion_counter + number_not_fully, 0] = current_moment
            electron_motion_array[motion_counter:motion_counter + number_not_fully, 1:] = \
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

    return atoms, local_ionization_array, electron_motion_data, amount_of_electrons, \
           amount_of_fully_ionized_states, fully_ionized_indices_general


def placed_cells_ionization_rk4_numba(atoms, t_array):
    number_of_atoms = len(atoms)
    fully_ionized_mask = np.zeros(number_of_atoms, dtype=bool)

    local_ionization_array = np.zeros((len(t_array), cfg.z_max), dtype=int)
    ion_times = np.full((number_of_atoms, cfg.z_max), cfg.t_0 + 1.0)
    electron_motion_array = np.zeros((cfg.z_max * number_of_atoms, 4))

    print('Ионизация...')
    print('\n')
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


def pulse_field(time, rad_vec):
    x_coord, y_coord, z_coord = rad_vec
    r_squared = y_coord * y_coord + z_coord * z_coord

    x_r = np.pi * cfg.w_0 * cfg.w_0  # рэлеевская длина волны

    x_coord_x_r = x_coord / x_r
    x_r_x_coord = x_r / x_coord

    radius = np.sqrt(1 + x_coord_x_r * x_coord_x_r)  # без w_0!!! ТАК НАДО!!
    spatial_structure = 1 / radius * np.exp(-r_squared / (cfg.w_0 * radius) ** 2)

    phi = np.arctan(x_coord_x_r)
    rho = x_coord * (1 + x_r_x_coord * x_r_x_coord)
    phase = 2 * np.pi * x_coord - time - phi + np.pi * r_squared / rho

    envelope = np.exp(-(time - 2 * np.pi * x_coord) ** 2 / cfg.tau ** 2)

    ellipticity = [1 / np.sqrt(1 + cfg.eps * cfg.eps), cfg.eps / np.sqrt(1 + cfg.eps * cfg.eps)]

    cos_comp = spatial_structure * envelope * np.cos(phase)
    sin_comp = spatial_structure * envelope * np.sin(phase)

    e_field = np.array([0,
               cfg.a_0 * cos_comp * ellipticity[0],
               cfg.a_0 * sin_comp * ellipticity[1] * cfg.right_or_left])

    h_field = np.array([0,
               -cfg.a_0 * sin_comp * ellipticity[1] * cfg.right_or_left,
               cfg.a_0 * cos_comp * ellipticity[0]])

    return np.hstack((e_field, h_field))


def electron_lorenz_equation(time, variables_vector):  # заряд учтен!!!
    x, y, z, momenta_x, momenta_y, momenta_z = variables_vector

    current_position = np.array([x, y, z])
    charge = -1  # заряд электрона в элементарных зарядах
    field = pulse_field(time, current_position) * charge

    electric, magnetic = field[:3], field[3:]

    energy = np.sqrt(1 + momenta_x ** 2 + momenta_y ** 2 + momenta_z ** 2)

    cross_product = [momenta_y * magnetic[2] - momenta_z * magnetic[1],
                     momenta_z * magnetic[0] - momenta_x * magnetic[2],
                     momenta_x * magnetic[1] - momenta_y * magnetic[0]]

    coefficient = 1 / (2 * np.pi)

    x_eq = coefficient * momenta_x / energy
    y_eq = coefficient * momenta_y / energy
    z_eq = coefficient * momenta_z / energy

    p_x_eq = electric[0] + cross_product[0] / energy
    p_y_eq = electric[1] + cross_product[1] / energy
    p_z_eq = electric[2] + cross_product[2] / energy

    return np.array([x_eq, y_eq, z_eq, p_x_eq, p_y_eq, p_z_eq])


def solve_electron_motion(t_start, initial_r_v):
    t_span = (t_start, cfg.t_0)

    sol = solve_ivp(
        electron_lorenz_equation,
        t_span,
        np.hstack((initial_r_v, np.zeros(3))),
        method='RK45'
    )
    return sol.y[3, -1], sol.y[4, -1], sol.y[5, -1]


def single_electron_process(initial_data):
    initial_t = initial_data[0]
    initial_position = initial_data[1:]
    return solve_electron_motion(initial_t, initial_position)


def electron_parallel_simulation(ionization_data, n_processes=None):  # возвращает массивы Numpy!!!
    if n_processes is None:
        n_processes = cpu_count()

    print(f"\nЭлектронов: {len(ionization_data)}")
    print(f"Процессов: {n_processes}")

    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(single_electron_process, ionization_data),
            total=len(ionization_data),
            desc='Расчёт'
        ))

    p_x_arr = np.array([r[0] for r in results])
    p_y_arr = np.array([r[1] for r in results])
    p_z_arr = np.array([r[2] for r in results])

    return p_x_arr, p_y_arr, p_z_arr


def solve_momenta_vs_time(t_start, initial_r_v):
    t_stop = cfg.t_0
    t_span = (t_start, t_stop)
    t_eval = np.arange(t_start, t_stop, cfg.delta_t)

    sol = solve_ivp(
        electron_lorenz_equation,
        t_span,
        np.hstack((initial_r_v, np.zeros(3))),
        t_eval=t_eval,
        method='RK45',
    )
    return sol.t, sol.y[3], sol.y[4], sol.y[5]
