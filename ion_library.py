from scipy.integrate import solve_ivp
from numba import njit
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import ion_config as cfg
import numpy as np

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


@njit
def beam_components(x, y, z, moment_of_time, beam_radius):
    """
    :param x: x-координата поля
    :param y: y-координата поля
    :param z: z-координата поля
    :param moment_of_time: момент времени, в который вычисляется поле
    :param beam_radius: радиус гауссова пучка
    :return: косинус- и синус-компонента поля (для последующего использования в других функциях)
    """

    r_squared = y * y + z * z
    x_r = np.pi * beam_radius * beam_radius

    x_xr = x / x_r
    xr_x = x_r / x

    phi = np.arctan(x_xr)
    rho = x * (1 + xr_x * xr_x)
    phase = 2 * np.pi * x - moment_of_time - phi + np.pi * r_squared / rho

    radius = np.sqrt(1 + x_xr * x_xr)
    spatial_part = 1.0 / radius * np.exp(-r_squared / (cfg.w_0 * radius) ** 2)

    envelope = np.exp(-(moment_of_time - 2 * np.pi * x) ** 2 / cfg.tau ** 2)

    cos_comp = spatial_part * envelope * np.cos(phase)
    sin_comp = spatial_part * envelope * np.sin(phase)

    return cos_comp, sin_comp


@njit
def w_ppt_numba(
        moment_of_time,
        particles,
        ion_potentials,
        n_array,
        c_array,
        b_array,
        time_step,
        beam_radius, ell_value,
        r_or_l, atomic_field_value
):
    """
    :param moment_of_time: момент времени, в который вычисляется вероятность
    :param particles: массив атомов (результат работы placing_cells_with_ions_motion)
    :param ion_potentials: массив с потенциалами ионизации (см. ion_config)
    :param n_array: массив из коэффициентов для формулы w_PPT (см. ion_config)
    :param c_array: массив из коэффициентов для формулы w_PPT (см. ion_config)
    :param b_array: массив из коэффициентов для формулы w_PPT (см. ion_config)
    :param time_step: шаг по времени (передаем delta_t как локальную переменную, а не глобальную)
    :param beam_radius: радиус гауссова пучка (передаем w_0 как локальную переменную, а не глобальную)
    :param ell_value: эллиптичность поля (локальная, а не глобальная переменная)
    :param r_or_l: левая или праваля поялризация (+1 - правая)
    :param atomic_field_value: величина поля в атомных единицах
    :return: массив той же размерности, что и particles - вероятности для всех атомов сразу
     в конкретный момент времени
    """
    n = np.shape(particles)[0]
    result = np.empty(n)

    for i in range(n):
        x = particles[i, 0]
        y = particles[i, 1]
        z = particles[i, 2]

        charge = int(particles[i, 6])  # заряд АТОМНОГО ОСТАТКА
        m_value = particles[i, 8]
        g_m_value = particles[i, 9]

        cos_comp, sin_comp = beam_components(x, y, z, moment_of_time, beam_radius)

        electric_field_abs_value = np.sqrt(
            (cos_comp / np.sqrt(1 + ell_value * ell_value)) ** 2 +
            (sin_comp * cfg.eps / np.sqrt(1 + ell_value * ell_value) * r_or_l) ** 2
        ) * atomic_field_value

        ionization_order_array_index = charge - 1
        i_p = ion_potentials[ionization_order_array_index]
        c_value = c_array[ionization_order_array_index]  # c_l_n^2
        b_value = b_array[ionization_order_array_index]  # b_l_m

        field_char = (2 * i_p) ** 1.5
        field = electric_field_abs_value / field_char

        if field < 1e-12:
            field = 1e-12

        field_power = 2 * n_array[ionization_order_array_index] - abs(m_value) - 1

        result[i] = (
            4
            * c_value
            * b_value
            * i_p
            * (2 / field) ** field_power
            * np.exp(-2 / (3 * field))
            * g_m_value
            * time_step
        )

    return result


@njit
def dp_dt_numba(moment_of_time, momenta, position, charge, beam_radius, ell_value, r_or_l, a0_field_value):
    """
    :param moment_of_time: момент времени, в который вычисляется правая часть уравнения Лоренца
    :param momenta: импульс одного атома (массив из трех элементов)
    :param position: положение одного атома (массив из трех элементов)
    :param charge: заряд атома/иона (реальный, а не заряд атомного остатка)
    :param beam_radius: радиус гауссова пучка (передаем w_0 как локальную переменную, а не глобальную)
    :param ell_value: эллиптичность поля (локальная, а не глобальная переменная)
    :param r_or_l: левая или праваля поялризация (+1 - правая)
    :param a0_field_value: величина поля в единицах a_0 (как аргумент сюда подаем a_0_ion!!!)
    :return: правую часть уравнений Лоренца для одного атома (массив размерности 3)
    """
    px, py, pz = momenta[0], momenta[1], momenta[2]
    x, y, z = position[0], position[1], position[2]

    cos_comp, sin_comp = beam_components(x, y, z, moment_of_time, beam_radius)

    ell_y = 1.0 / np.sqrt(1.0 + ell_value * ell_value)
    ell_z = ell_value / np.sqrt(1.0 + ell_value * ell_value)

    e_x = 0.0
    e_y = a0_field_value * cos_comp * ell_y
    e_z = a0_field_value * sin_comp * ell_z * r_or_l

    h_x = 0.0
    h_y = -a0_field_value * sin_comp * ell_z * r_or_l
    h_z = a0_field_value * cos_comp * ell_y

    gamma_factor = np.sqrt(1 + px * px + py * py + pz * pz)
    inv_gamma = 1.0 / gamma_factor

    cross_x = py * h_z - pz * h_y
    cross_y = pz * h_x - px * h_z
    cross_z = px * h_y - py * h_x

    dpx = charge * (e_x + cross_x * inv_gamma)
    dpy = charge * (e_y + cross_y * inv_gamma)
    dpz = charge * (e_z + cross_z * inv_gamma)

    return np.array([dpx, dpy, dpz])


@njit
def rk4_step_numba(t, p, r, charge, dt, beam_radius, ell_value, r_or_l, a0_field_value):
    """
    Один шаг RK4 для импульса. Аргументы - см. dp_dt_numba.
    Возвращает проитерированный массив из трех компонент импульса
    """
    k1 = dp_dt_numba(t, p, r, charge, beam_radius, ell_value, r_or_l, a0_field_value)

    p2 = p + 0.5 * dt * k1
    k2 = dp_dt_numba(t + 0.5 * dt, p2, r, charge, beam_radius, ell_value, r_or_l, a0_field_value)

    p3 = p + 0.5 * dt * k2
    k3 = dp_dt_numba(t + 0.5 * dt, p3, r, charge, beam_radius, ell_value, r_or_l, a0_field_value)

    p4 = p + dt * k3
    k4 = dp_dt_numba(t + dt, p4, r, charge, beam_radius, ell_value, r_or_l, a0_field_value)

    return p + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit
def integrate_ion_momentum_numba(position, ion_times, t_array,
                                 beam_radius, ell_value, r_or_l, a0_field_value):
    p = np.zeros(3)
    event_index = 0

    for i in range(len(t_array) - 1):
        t = t_array[i]
        dt = t_array[i+1] - t

        while event_index < cfg.z_max and t >= ion_times[event_index]:
            event_index += 1

        charge = event_index + cfg.start_charge_number_of_particles
        if charge == 0:
            continue

        p = rk4_step_numba(t, p, position, charge, dt, beam_radius, ell_value, r_or_l, a0_field_value)
    return p


def placed_cells_ionization_rk4(atoms, t_array):
    number_of_atoms = len(atoms)
    fully_ionized_mask = np.zeros(number_of_atoms, dtype=bool)

    local_ionization_array = np.zeros((len(t_array), cfg.z_max),
                                      dtype=int)  # массив с данными о пот-х ионизации электронов
    ion_times = np.full((number_of_atoms, cfg.z_max), cfg.t_0 + 1.0)
    # массив с зависимостями зар-в атомов/ионов от времени

    electron_motion_array = np.zeros((cfg.z_max * number_of_atoms, 4))
    motion_counter = 0

    for i, current_moment in enumerate(t_array):
        progress = (i + 1) / len(t_array) * 100
        print(f'\rПрогресс: {i + 1}/{len(t_array)} ({progress:.1f}%)', end='')

        # =ИОНИЗАЦИЯ=
        random_values = np.random.random(number_of_atoms)
        probabilities = w_ppt_numba(current_moment, atoms, cfg.ionization_potentials,
                                    cfg.n_star_array, cfg.c_n_l_array, cfg.b_l_m_array,
                                    cfg.delta_t, cfg.w_0, cfg.eps, cfg.right_or_left, cfg.atomic_field)

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

    for k, atom in enumerate(tqdm(atoms, desc="Processing atoms")):
        atom[3:6] = integrate_ion_momentum_numba(atom[:3], ion_times[k], t_array,
                                                 cfg.w_0, cfg.eps, cfg.right_or_left, cfg.a_0_ion)

    electron_motion_data = electron_motion_array[:motion_counter]
    amount_of_electrons = motion_counter
    amount_of_fully_ionized_states = np.sum(fully_ionized_mask)

    return atoms, local_ionization_array, electron_motion_data, amount_of_electrons, amount_of_fully_ionized_states


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
            desc='Расчет'
        ))

    p_x_arr = np.array([r[0] for r in results])
    p_y_arr = np.array([r[1] for r in results])
    p_z_arr = np.array([r[2] for r in results])

    return p_x_arr, p_y_arr, p_z_arr
