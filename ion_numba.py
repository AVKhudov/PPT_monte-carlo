from numba import njit, prange
import ion_config as cfg
import numpy as np


@njit
def cos_envelope(moment_of_time, x):
    env_phase = moment_of_time - 2 * np.pi * x
    if (-cfg.tau <= env_phase) & (env_phase <= cfg.tau):
        envelope = np.cos(np.pi * env_phase / cfg.tau / 2) ** 2
    else:
        envelope = 0.
    return envelope


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

    envelope = cos_envelope(moment_of_time, x)

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

        beam_radius,
        ell_value,
        r_or_l,

        atomic_field_value
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
            (sin_comp * ell_value / np.sqrt(1 + ell_value * ell_value) * r_or_l) ** 2
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
def single_atom_ionization_probability(
        moment_of_time,
        atom,

        ion_potentials,
        n_array,
        c_array,
        b_array,

        time_step,

        beam_radius,
        ell_value,
        r_or_l,

        atomic_field_value
):
    x = atom[0]
    y = atom[1]
    z = atom[2]

    charge = int(atom[6])
    m_value = atom[8]
    g_m_value = atom[9]

    cos_comp, sin_comp = beam_components(x, y, z, moment_of_time, beam_radius)

    electric_field_abs_value = np.sqrt(
        (cos_comp / np.sqrt(1.0 + ell_value * ell_value)) ** 2 +
        (sin_comp * ell_value / np.sqrt(1.0 + ell_value * ell_value) * r_or_l) ** 2
    ) * atomic_field_value

    ionization_order_array_index = charge - 1
    i_p = ion_potentials[ionization_order_array_index]
    c_value = c_array[ionization_order_array_index]
    b_value = b_array[ionization_order_array_index]

    field_char = (2.0 * i_p) ** 1.5
    field = electric_field_abs_value / field_char

    if field < 1e-12:
        field = 1e-12

    field_power = 2.0 * n_array[ionization_order_array_index] - abs(m_value) - 1.0

    probability = (
        4.0
        * c_value
        * b_value
        * i_p
        * (2.0 / field) ** field_power
        * np.exp(-2.0 / (3.0 * field))
        * g_m_value
        * time_step
    )

    return probability


@njit
def simulate_ionization_numba(
        atoms,
        t_array,

        ion_times,

        local_ionization_array,
        electron_motion_array,
        fully_ionized_mask,
        ion_potentials,
        n_array,
        c_array,
        b_array,

        time_step,

        beam_radius,
        ell_value,
        r_or_l,

        atomic_field_value,

        ionization_order_array,

        z_maximum
):
    motion_counter = 0
    number_of_atoms = atoms.shape[0]
    number_of_time_steps = len(t_array)

    for i in range(number_of_time_steps):
        current_moment = t_array[i]

        for j in range(number_of_atoms):
            if fully_ionized_mask[j]:
                continue

            charge = int(atoms[j, 6])

            probability = single_atom_ionization_probability(
                current_moment,
                atoms[j],
                ion_potentials,
                n_array,
                c_array,
                b_array,
                time_step,
                beam_radius,
                ell_value,
                r_or_l,
                atomic_field_value
            )

            if np.random.random() >= probability:
                continue

            if charge == z_maximum:
                fully_ionized_mask[j] = True

                ion_times[j, z_maximum - 1] = current_moment
                local_ionization_array[i, z_maximum - 1] += 1

                electron_motion_array[motion_counter, 0] = current_moment
                electron_motion_array[motion_counter, 1] = atoms[j, 6]  # charge!!!
                electron_motion_array[motion_counter, 2] = atoms[j, 0]
                electron_motion_array[motion_counter, 3] = atoms[j, 1]
                electron_motion_array[motion_counter, 4] = atoms[j, 2]
                motion_counter += 1
            else:
                ion_times[j, charge - 1] = current_moment
                local_ionization_array[i, charge - 1] += 1

                electron_motion_array[motion_counter, 0] = current_moment
                electron_motion_array[motion_counter, 1] = atoms[j, 6]  # charge!!!
                electron_motion_array[motion_counter, 2] = atoms[j, 0]
                electron_motion_array[motion_counter, 3] = atoms[j, 1]
                electron_motion_array[motion_counter, 4] = atoms[j, 2]
                motion_counter += 1

                atoms[j, 6:] = ionization_order_array[charge]

    return motion_counter


# Новый процесс для одного иона
@njit
def single_ion_process_numba(
        t_array,
        atom,

        max_ionizations,

        ion_potentials,
        ionization_order_array,
        n_array,
        c_array,
        b_array,

        time_step,

        beam_radius,
        ell_value,
        r_or_l,

        atomic_field_value
):
    single_ionization_times = np.full(max_ionizations, np.inf)

    ionization_count = 0

    random_values = np.random.random(len(t_array))

    for i in range(len(t_array)):
        moment_of_time = t_array[i]
        probability = single_atom_ionization_probability(
            moment_of_time,
            atom,

            ion_potentials,
            n_array,
            c_array,
            b_array,

            time_step,

            beam_radius,
            ell_value,
            r_or_l,

            atomic_field_value
        )

        if random_values[i] < probability:
            single_ionization_times[ionization_count] = moment_of_time
            ionization_count += 1

            if ionization_count == max_ionizations:
                break

            atom[6:] = ionization_order_array[ionization_count]

    return single_ionization_times, ionization_count  # когда атом ионизовывался, сколько раз атом ионизовывался


@njit(parallel=True)
def parallel_ionization_numba(
        t_array,
        atoms,

        max_ionizations,

        ion_potentials,
        ionization_order_array,
        n_array,
        c_array,
        b_array,

        time_step,

        beam_radius,
        ell_value,
        r_or_l,

        atomic_field_value
):
    number_of_atoms = len(atoms)

    ionization_times = np.full((number_of_atoms, max_ionizations), np.inf)
    ion_counts = np.zeros(number_of_atoms, dtype=np.int64)

    for atom_index in prange(number_of_atoms):
        times, count = single_ion_process_numba(
            t_array,
            atoms[atom_index],

            max_ionizations,

            ion_potentials,
            ionization_order_array,
            n_array,
            c_array,
            b_array,

            time_step,

            beam_radius,
            ell_value,
            r_or_l,

            atomic_field_value
        )
        ionization_times[atom_index, :] = times
        ion_counts[atom_index] = count

    return ionization_times, ion_counts


@njit
def dp_dt_numba(
        moment_of_time,
        momenta,

        position,
        charge,

        beam_radius,
        ell_value,
        r_or_l,
        a0_field_value):
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
def rk4_step_numba(
        t,
        p,
        r,
        charge,
        dt,

        beam_radius,
        ell_value,
        r_or_l,
        a0_field_value):
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
def integrate_ion_momentum_numba(
        position,

        ion_times,
        t_array,

        beam_radius,
        ell_value,
        r_or_l,
        a0_field_value,

        start_charge,
        z_maximum):
    p = np.zeros(3)
    event_index = 0

    for i in range(len(t_array) - 1):
        t = t_array[i]
        dt = t_array[i+1] - t

        while event_index < z_maximum and t >= ion_times[event_index]:
            event_index += 1

        charge = event_index + start_charge
        if charge == 0:
            continue

        p = rk4_step_numba(t, p, position, charge, dt, beam_radius, ell_value, r_or_l, a0_field_value)
    return p


@njit(parallel=True)
def integrate_all_ions_numba(
        positions,
        ion_times,

        t_array,

        beam_radius,
        ell_value,
        r_or_l,
        a0_field_value,

        start_charge):
    n_atoms = positions.shape[0]
    result = np.zeros((n_atoms, 3))
    for k in prange(n_atoms):
        p = np.zeros(3)

        for i in range(len(t_array) - 1):
            t = t_array[i]
            dt = t_array[i + 1] - t

            event_index = np.sum(t >= ion_times[k])

            charge = event_index + start_charge
            if charge == 0:
                continue

            p = rk4_step_numba(
                t, p, positions[k],
                charge, dt,
                beam_radius, ell_value, r_or_l, a0_field_value
            )

        result[k] = p

    return result
