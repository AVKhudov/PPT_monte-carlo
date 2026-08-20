from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import ion_config as cfg
import numpy as np


def pulse_envelope(time, x):
    env_phase = time - 2 * np.pi * x
    if (-cfg.tau <= env_phase) & (env_phase <= cfg.tau):
        envelope = np.cos(np.pi * env_phase / cfg.tau / 2) ** 2
    else:
        envelope = 0.
    return envelope


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

    envelope = pulse_envelope(time, x_coord)

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


def pulse_field_with_longitudinal_component(time, rad_vec):
    x, y, z = rad_vec
    r_squared = y * y + z * z

    xr = np.pi * cfg.w_0 * cfg.w_0  # рэлеевская длина волны

    w = cfg.w_0 * np.sqrt(1 + (x / xr) * (x / xr))  # w(x)
    rho = x * (1 + (xr / x) * (xr / x))
    rho_s = 1 - (xr / x) * (xr / x)
    phi = np.arctan(x / xr)
    phase = 2 * np.pi * x - time - phi + np.pi * r_squared / rho

    envelope = pulse_envelope(time, x)

    e_field = np.array([0,
                        cfg.a_0 * cfg.w_0 / w * np.exp(-r_squared / w ** 2) * np.cos(phase) * envelope,
                        0])

    h_z_1 = (1 + (x / xr) ** 2) ** (-1.5) * np.sin(phase) * x / xr ** 2
    h_z_2 = (1 + (x / xr) ** 2) ** (-1) * np.sin(phase) * 2 * r_squared * cfg.w_0 * x / (w ** 3 * xr ** 2)
    h_z_3 = (1 + (x / xr) ** 2) ** (-0.5) * np.cos(phase) * (
        2 * np.pi - np.pi * r_squared * rho_s / rho ** 2 - (1 + (x / xr) ** 2) ** (-1) / xr
    )

    # продольная компонента поля:
    h_x = np.exp(-r_squared / w ** 2) * (1 + (x / xr) ** 2) ** (-0.5) * (
            np.sin(phase) * 2 * z / w ** 2 - np.cos(phase) * 2 * np.pi * z / rho
    ) * envelope
    # поперечная компонента поля:
    h_z = np.exp(-r_squared / w ** 2) * (-h_z_1 + h_z_2 + h_z_3) * envelope

    h_field = np.array([cfg.a_0 * h_x,
                        0,
                        cfg.a_0 * h_z])

    return np.hstack((e_field, h_field))


def electron_lorenz_equation(time, variables_vector):  # заряд учтен!!!
    x, y, z, momenta_x, momenta_y, momenta_z = variables_vector

    current_position = np.array([x, y, z])
    charge = -1  # заряд электрона в элементарных зарядах

    if cfg.longitudinal_component:
        field = pulse_field_with_longitudinal_component(time, current_position) * charge
    else:
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
    sort = initial_data[1]
    initial_position = initial_data[2:]
    p_x_final, p_y_final, p_z_final = solve_electron_motion(initial_t, initial_position)
    return sort, initial_t, p_x_final, p_y_final, p_z_final


def electron_parallel_simulation(ionization_data, n_processes=None):  # возвращает массивы Numpy!!!
    if n_processes is None:
        n_processes = cpu_count()

    print(f"\nЭлектронов: {len(ionization_data)}")
    print(f"Процессов: {n_processes}")

    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(single_electron_process, ionization_data),  # порядок данных в выводе СОХРАНЯЕТСЯ
            total=len(ionization_data),
            desc='Расчёт'
        ))

    sorts_arr = np.array([r[0] for r in results])
    initial_t_arr = np.array([r[1] for r in results])
    p_x_arr = np.array([r[2] for r in results])
    p_y_arr = np.array([r[3] for r in results])
    p_z_arr = np.array([r[4] for r in results])

    return sorts_arr, initial_t_arr, p_x_arr, p_y_arr, p_z_arr


def select_electrons_for_trajectories(
        electron_motion_input,
        stride=100,
        rare_sort_threshold=50,
        custom_sorts=None
):
    selected_chunks = []

    if custom_sorts is None:
        custom_sorts = []

    all_sorts = electron_motion_input[:, 1].astype(int)
    unique_sorts = np.unique(all_sorts)

    for sort_value in unique_sorts:
        sort_mask = all_sorts == sort_value
        sort_electrons = electron_motion_input[sort_mask]

        # Для заданных сортов сохраняем все электроны
        if sort_value in custom_sorts:
            selected_chunks.append(sort_electrons)

        # Для редких сортов тоже сохраняем все
        elif len(sort_electrons) <= rare_sort_threshold:
            selected_chunks.append(sort_electrons)

        # Для остальных берём каждый stride-й электрон
        else:
            selected_chunks.append(sort_electrons[::stride])

    return np.vstack(selected_chunks)


def solve_electron_trajectory(t_start, initial_r_v):
    time_resolution = 0.1
    trajectory_duration = 15000.
    n_steps = int(trajectory_duration / time_resolution)
    t_eval = t_start + np.arange(n_steps) * time_resolution
    t_span = (t_eval[0], t_eval[-1])

    sol = solve_ivp(
        electron_lorenz_equation,
        t_span,
        np.hstack((initial_r_v, np.zeros(3))),
        t_eval=t_eval,
        method='RK45',
    )

    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4], sol.y[5]


def single_electron_trajectory_process(initial_data):
    initial_t = initial_data[0]
    electron_sort = int(initial_data[1])
    initial_position = initial_data[2:]

    t_arr, x_arr, y_arr, z_arr, p_x_arr, p_y_arr, p_z_arr = solve_electron_trajectory(
        initial_t,
        initial_position,
    )

    return electron_sort, t_arr, x_arr, y_arr, z_arr, p_x_arr, p_y_arr, p_z_arr


def electron_trajectories_simulation(selected_ionization_data, n_processes=None):
    if n_processes is None:
        n_processes = cpu_count()

    print(f"\nТраекторий для расчёта: {len(selected_ionization_data)}")
    print(f"Процессов: {n_processes}")

    worker_input = [
        selected_ionization_data[i] for i in range(len(selected_ionization_data))
    ]

    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(single_electron_trajectory_process, worker_input),  # порядок данных в выводе СОХРАНЯЕТСЯ
            total=len(worker_input),
            desc='Траектории'
        ))

    n_selected = len(results)
    n_steps = len(results[0][1])

    electron_sorts = np.empty(n_selected, dtype=int)
    t_array = np.empty((n_selected, n_steps))
    x_array = np.empty((n_selected, n_steps))
    y_array = np.empty((n_selected, n_steps))
    z_array = np.empty((n_selected, n_steps))
    p_x_array = np.empty((n_selected, n_steps))
    p_y_array = np.empty((n_selected, n_steps))
    p_z_array = np.empty((n_selected, n_steps))

    for i, result in enumerate(results):
        electron_sorts[i] = result[0]
        t_array[i] = result[1]
        x_array[i] = result[2]
        y_array[i] = result[3]
        z_array[i] = result[4]
        p_x_array[i] = result[5]
        p_y_array[i] = result[6]
        p_z_array[i] = result[7]

    return {
        "sorts": electron_sorts,
        "t": t_array,
        "x": x_array,
        "y": y_array,
        "z": z_array,
        "p_x": p_x_array,
        "p_y": p_y_array,
        "p_z": p_z_array,
    }


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
