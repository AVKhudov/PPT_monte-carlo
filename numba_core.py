from numba import njit, prange
import numpy as np


@njit
def cos_envelope(moment_of_time, x, tau):
    """
    :param moment_of_time: момент времени, в который вычисляется поле
    :param x: x координата
    :param tau: длительность импульса
    :return: cos^2 огибающая поля
    """
    env_phase = moment_of_time - 2 * np.pi * x
    if (-tau <= env_phase) & (env_phase <= tau):
        envelope = np.cos(np.pi * env_phase / tau / 2) ** 2
    else:
        envelope = 0.
    return envelope


@njit
def beam_components(x, y, z, moment_of_time, beam_radius, tau):
    """
    :param x: x координата
    :param y: y координата
    :param z: z координата
    :param moment_of_time: момент времени, в который вычисляется поле
    :param beam_radius: радиус гауссова пучка
    :param tau: длительность импульса
    :return: косинус- и синус-компонента поля
    """

    envelope = cos_envelope(moment_of_time, x, tau)

    if envelope == 0.:
        cos_comp = 0.
        sin_comp = 0.

    else:
        r_squared = y * y + z * z
        x_r = np.pi * beam_radius * beam_radius

        x_xr = x / x_r
        xr_x = x_r / x

        phi = np.arctan(x_xr)
        rho = x * (1 + xr_x * xr_x)
        phase = 2 * np.pi * x - moment_of_time - phi + np.pi * r_squared / rho

        radius = np.sqrt(1 + x_xr * x_xr)
        spatial_part = 1. / radius * np.exp(-r_squared / (beam_radius * radius) ** 2)

        cos_comp = spatial_part * envelope * np.cos(phase)
        sin_comp = spatial_part * envelope * np.sin(phase)

    return cos_comp, sin_comp


@njit
def boris_field(x, y, z, moment_of_time, beam_radius, tau, a_0_value, ell_value, r_or_l, x_r):
    """
        :param x: x координата
        :param y: y координата
        :param z: z координата
        :param moment_of_time: момент времени, в который вычисляется поле
        :param beam_radius: радиус гауссова пучка
        :param tau: длительность импульса
        :param a_0_value: амплитуда поля в едницах a_0
        :param ell_value: эллиптичность поля
        :param r_or_l: правая или левая поляризация (+1 - правая)
        :param x_r: рэлеевская длина волны
        :return: косинус- и синус-компонента поля
        """

    envelope = cos_envelope(moment_of_time, x, tau)

    if envelope == 0.:
        electric_field = np.zeros(3)
        magnetic_field = np.zeros(3)

    else:
        r_squared = y * y + z * z

        x_xr = x / x_r
        xr_x = x_r / x

        phi = np.arctan(x_xr)
        rho = x * (1 + xr_x * xr_x)
        phase = 2 * np.pi * x - moment_of_time - phi + np.pi * r_squared / rho

        radius = np.sqrt(1 + x_xr * x_xr)
        spatial_part = 1. / radius * np.exp(-r_squared / (beam_radius * radius) ** 2)

        cos_comp = spatial_part * envelope * np.cos(phase)
        sin_comp = spatial_part * envelope * np.sin(phase)

        ell_denominator = np.sqrt(1 + ell_value * ell_value)

        ellipticity_0 = 1. / ell_denominator
        ellipticity_1 = ell_value / ell_denominator

        electric_field = np.array([0.,
                                   a_0_value * cos_comp * ellipticity_0,
                                   a_0_value * sin_comp * ellipticity_1 * r_or_l])

        magnetic_field = np.array([0.,
                                   -a_0_value * sin_comp * ellipticity_1 * r_or_l,
                                   a_0_value * cos_comp * ellipticity_0])

    return electric_field, magnetic_field


@njit
def boris_relativistic_step(
        time,
        radius_vector,
        p_half,

        delta_t,

        beam_radius,
        tau,
        a_0_value,
        ell_value,
        r_or_l,
        x_r
):
    x, y, z = radius_vector

    electric_field, magnetic_field = boris_field(x, y, z, time, beam_radius, tau, a_0_value, ell_value, r_or_l, x_r)

    if np.all(electric_field == 0.) and np.all(magnetic_field == 0.):
        gamma_half = np.sqrt(1 + np.dot(p_half, p_half))
        v_half = p_half / gamma_half

        radius_vector_new = radius_vector + v_half * delta_t
        p_new_half = p_half

    else:
        p_minus = p_half + 0.5 * electric_field * delta_t
        gamma_minus = np.sqrt(1. + np.dot(p_minus, p_minus))

        t_vector = 0.5 * magnetic_field * delta_t / gamma_minus
        s_vector = 2. * t_vector / (1. + np.dot(t_vector, t_vector))

        p_prime = p_minus + np.cross(p_minus, t_vector)
        p_plus = p_minus + np.cross(p_prime, s_vector)

        p_new_half = p_plus + 0.5 * electric_field * delta_t
        gamma_new = np.sqrt(1. + np.dot(p_new_half, p_new_half))

        v_new_half = p_new_half / gamma_new

        radius_vector_new = radius_vector + v_new_half * delta_t

    return radius_vector_new, p_new_half


@njit
def single_boris_process(
        initial_data,
        el_sim_duration,

        delta_t,

        beam_radius,
        tau,
        a_0_value,
        ell_value,
        r_or_l,
        x_r
):
    initial_t = initial_data[0]
    sort = initial_data[1]

    current_time = initial_data[0]
    current_r = initial_data[2:5]
    current_p = np.zeros(3)

    while current_time < el_sim_duration:
        current_r, current_p = boris_relativistic_step(
            current_time,
            current_r,
            current_p,

            delta_t,

            beam_radius,
            tau,
            a_0_value,
            ell_value,
            r_or_l,
            x_r
        )

        current_time += delta_t

    p_x_final, p_y_final, p_z_final = current_p

    single_output = np.array([sort, initial_t, p_x_final, p_y_final, p_z_final])

    return single_output


@njit(parallel=True)
def boris_parallel_simulation(
        el_m_input,
        el_sim_duration,

        delta_t,

        beam_radius,
        tau,
        a_0_value,
        ell_value,
        r_or_l,
        x_r
):
    number_of_electrons = len(el_m_input)
    result = np.zeros((number_of_electrons, 5))

    for electron_idx in prange(number_of_electrons):
        result[electron_idx] = single_boris_process(
            el_m_input[electron_idx],
            el_sim_duration,

            delta_t,

            beam_radius,
            tau,
            a_0_value,
            ell_value,
            r_or_l,
            x_r
        )

    sorts_arr = result[:, 0]
    initial_t_arr = result[:, 1]
    p_x_arr = result[:, 2]
    p_y_arr = result[:, 3]
    p_z_arr = result[:, 4]

    return sorts_arr, initial_t_arr, p_x_arr, p_y_arr, p_z_arr


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
        atomic_field_value,
        tau
):
    x = atom[0]
    y = atom[1]
    z = atom[2]

    charge = int(atom[6])
    m_value = atom[8]
    g_m_value = atom[9]

    cos_comp, sin_comp = beam_components(x, y, z, moment_of_time, beam_radius, tau)

    electric_field_abs_value = np.sqrt(
        (cos_comp / np.sqrt(1.0 + ell_value * ell_value)) ** 2 +
        (sin_comp * ell_value / np.sqrt(1.0 + ell_value * ell_value) * r_or_l) ** 2
    ) * atomic_field_value

    ionization_order_array_index = charge - 1
    i_p = ion_potentials[ionization_order_array_index]
    c_value = c_array[ionization_order_array_index]
    b_value = b_array[ionization_order_array_index]

    field_char = (2. * i_p) ** 1.5
    field = electric_field_abs_value / field_char

    if field < 1e-12:
        field = 1e-12

    field_power = 2. * n_array[ionization_order_array_index] - abs(m_value) - 1.0

    probability = (
        4.
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
        tau,

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
                atomic_field_value,
                tau
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


@njit
def dp_dt_numba(moment_of_time, momenta, position, charge, beam_radius, ell_value, r_or_l, a0_field_value, tau):
    """
    :param moment_of_time: момент времени, в который вычисляется правая часть уравнения Лоренца
    :param momenta: импульс одного атома (массив из трех элементов)
    :param position: положение одного атома (массив из трех элементов)
    :param charge: заряд атома/иона (реальный, а не заряд атомного остатка)
    :param beam_radius: радиус гауссова пучка (передаем w_0 как локальную переменную, а не глобальную)
    :param ell_value: эллиптичность поля (локальная, а не глобальная переменная)
    :param r_or_l: левая или праваля поялризация (+1 - правая)
    :param a0_field_value: величина поля в единицах a_0 (как аргумент сюда подаем a_0_ion!!!)
    :param tau: длительность импульса
    :return: правую часть уравнений Лоренца для одного атома (массив размерности 3)
    """
    px, py, pz = momenta[0], momenta[1], momenta[2]
    x, y, z = position[0], position[1], position[2]

    cos_comp, sin_comp = beam_components(x, y, z, moment_of_time, beam_radius, tau)

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
def rk4_step_numba(t, p, r, charge, dt, beam_radius, ell_value, r_or_l, a0_field_value, tau):
    """
    Один шаг RK4 для импульса. Аргументы - см. dp_dt_numba.
    Возвращает проитерированный массив из трех компонент импульса
    """
    k1 = dp_dt_numba(t, p, r, charge, beam_radius, ell_value, r_or_l, a0_field_value, tau)

    p2 = p + 0.5 * dt * k1
    k2 = dp_dt_numba(t + 0.5 * dt, p2, r, charge, beam_radius, ell_value, r_or_l, a0_field_value, tau)

    p3 = p + 0.5 * dt * k2
    k3 = dp_dt_numba(t + 0.5 * dt, p3, r, charge, beam_radius, ell_value, r_or_l, a0_field_value, tau)

    p4 = p + dt * k3
    k4 = dp_dt_numba(t + dt, p4, r, charge, beam_radius, ell_value, r_or_l, a0_field_value, tau)

    return p + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit
def integrate_ion_momentum_numba(position, ion_times, t_array,
                                 beam_radius, ell_value, r_or_l, a0_field_value, tau, start_charge, z_maximum):
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

        p = rk4_step_numba(t, p, position, charge, dt, beam_radius, ell_value, r_or_l, a0_field_value, tau)
    return p


@njit(parallel=True)
def integrate_all_ions_numba(positions, ion_times, t_array,
                             beam_radius, ell_value, r_or_l, a0_field_value, tau, start_charge):
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
                beam_radius, ell_value, r_or_l, a0_field_value, tau
            )

        result[k] = p

    return result
