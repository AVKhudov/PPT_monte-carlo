from scipy.integrate import solve_ivp
import os
from datetime import datetime
from numba import njit
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm, ListedColormap

from ion_config import *

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

    positions = np.random.uniform(-0.5, 0.5, (number, 3)) * form
    positions[:, 0] = np.where(np.abs(positions[:, 0]) < 1e-10, 1e-6, positions[:, 0])
    momenta = np.zeros((number, 3))
    params = ionization_order_array[start_charge_number]

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
    spatial_part = 1.0 / radius * np.exp(-r_squared / (w_0 * radius) ** 2)

    envelope = np.exp(-(moment_of_time - 2 * np.pi * x) ** 2 / tau ** 2)

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
            (sin_comp * eps / np.sqrt(1 + ell_value * ell_value) * r_or_l) ** 2
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

        while event_index < z_max and t >= ion_times[event_index]:
            event_index += 1

        charge = event_index
        if charge == 0:
            continue

        p = rk4_step_numba(t, p, position, charge, dt, beam_radius, ell_value, r_or_l, a0_field_value)
    return p


def placed_cells_ionization_rk4(atoms):
    fully_ionized_mask = np.zeros(len(atoms), dtype=bool)

    ion_times = np.full((len(atoms), z_max), t_0 + 1.0)  # массив с зависимостями зарядов атомов/ионов от времени

    electron_motion_array = np.zeros((z_max * len(atoms), 4))
    motion_counter = 0

    for i, current_moment in enumerate(time_array):
        progress = (i + 1) / len(time_array) * 100
        print(f'\rПрогресс: {i + 1}/{len(time_array)} ({progress:.1f}%)', end='')

        # =ИОНИЗАЦИЯ=
        random_values = np.random.random(n_atoms)
        probabilities = w_ppt_numba(current_moment, atoms, ionization_potentials,
                                    n_star_array, c_n_l_array, b_l_m_array,
                                    delta_t, w_0, eps, right_or_left, atomic_field)

        ionization_mask = (random_values < probabilities) & (~fully_ionized_mask)
        ionization_indices = np.where(ionization_mask)[0]

        if ionization_indices.size == 0:
            continue

        charges = atoms[ionization_indices, 6].astype(int)  # заряды АТОМНЫХ ОСТАТКОВ

        fully_ionized_mask_local = charges == z_max

        fully_ionized_indices = ionization_indices[fully_ionized_mask_local]
        not_fully_ionized_indices = ionization_indices[~fully_ionized_mask_local]

        if fully_ionized_indices.size > 0:
            fully_ionized_mask[fully_ionized_indices] = True

            ion_times[fully_ionized_indices, z_max - 1] = current_moment
            ionization_array[i, z_max - 1] += 1

            number_fully = fully_ionized_indices.size
            electron_motion_array[motion_counter:motion_counter + number_fully, 0] = current_moment
            electron_motion_array[motion_counter:motion_counter + number_fully, 1:] = atoms[fully_ionized_indices, :3]
            motion_counter += number_fully

        if not_fully_ionized_indices.size > 0:
            not_fully_charges = atoms[not_fully_ionized_indices, 6].astype(int)

            ion_times[not_fully_ionized_indices, not_fully_charges - 1] = current_moment
            ionization_array[i, not_fully_charges - 1] += 1

            number_not_fully = not_fully_ionized_indices.size
            electron_motion_array[motion_counter:motion_counter + number_not_fully, 0] = current_moment
            electron_motion_array[motion_counter:motion_counter + number_not_fully, 1:] = \
                atoms[not_fully_ionized_indices, :3]
            motion_counter += number_not_fully

            atoms[not_fully_ionized_indices, 6:] = ionization_order_array[not_fully_charges]

    for k, atom in enumerate(atoms):
        atom[3:6] = integrate_ion_momentum_numba(atom[:3], ion_times[k], time_array, w_0, eps, right_or_left, a_0_ion)

    electron_motion_data = electron_motion_array[:motion_counter]
    number_of_electrons = motion_counter
    number_of_fully_ionized_states = np.sum(fully_ionized_mask)

    return electron_motion_data, atoms, number_of_electrons, number_of_fully_ionized_states


def pulse_field(time, rad_vec):
    x_coord, y_coord, z_coord = rad_vec
    r_squared = y_coord * y_coord + z_coord * z_coord

    x_r = np.pi * w_0 * w_0  # рэлеевская длина волны

    x_coord_x_r = x_coord / x_r
    x_r_x_coord = x_r / x_coord

    radius = np.sqrt(1 + x_coord_x_r * x_coord_x_r)  # без w_0!!! ТАК НАДО!!
    spatial_structure = 1 / radius * np.exp(-r_squared / (w_0 * radius) ** 2)

    phi = np.arctan(x_coord_x_r)
    rho = x_coord * (1 + x_r_x_coord * x_r_x_coord)
    phase = 2 * np.pi * x_coord - time - phi + np.pi * r_squared / rho

    envelope = np.exp(-(time - 2 * np.pi * x_coord) ** 2 / tau ** 2)

    ellipticity = [1 / np.sqrt(1 + eps * eps), eps / np.sqrt(1 + eps * eps)]

    cos_comp = spatial_structure * envelope * np.cos(phase)
    sin_comp = spatial_structure * envelope * np.sin(phase)

    e_field = np.array([0,
               a_0 * cos_comp * ellipticity[0],
               a_0 * sin_comp * ellipticity[1] * right_or_left])

    h_field = np.array([0,
               -a_0 * sin_comp * ellipticity[1] * right_or_left,
               a_0 * cos_comp * ellipticity[0]])

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
    t_span = (t_start, t_0)

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


def solve_electron_trajectory(t_start, initial_r_v):
    t_stop = t_0
    t_span = (t_start, t_stop)
    t_eval = np.arange(t_start, t_stop, delta_t)

    sol = solve_ivp(
        electron_lorenz_equation,
        t_span,
        np.hstack((initial_r_v, np.zeros(3))),
        t_eval=t_eval,
        method='RK45',
        vectorized=True
    )
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


def solve_momenta_vs_time(t_start, initial_r_v):
    t_stop = t_0
    t_span = (t_start, t_stop)
    t_eval = np.arange(t_start, t_stop, delta_t)

    sol = solve_ivp(
        electron_lorenz_equation,
        t_span,
        np.hstack((initial_r_v, np.zeros(3))),
        t_eval=t_eval,
        method='RK45',
        vectorized=True
    )
    return sol.t, sol.y[3], sol.y[4], sol.y[5]


def angle_calculation(y_comp, x_comp):
    if (y_comp > 0 and x_comp > 0) or (y_comp < 0 < x_comp):
        return np.arctan(y_comp / x_comp)
    if y_comp > 0 > x_comp:
        return np.pi + np.arctan(y_comp / x_comp)
    if y_comp < 0 and x_comp < 0:
        return np.arctan(y_comp / x_comp) - np.pi


def create_timestamp_folder(base_path=r'C:\Users\Dns\Desktop\MC_Ion'):
    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    folder_path = os.path.join(base_path, timestamp)  # создаем путь к папке
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_file(folder_path, filename, data):
    file_path = os.path.join(folder_path, filename)  # создаем путь к файлу
    np.save(file_path, data)


def bar_plotter(atoms, save=False, title='ions_bar.pdf'):

    charge_numbers = atoms[:, 3].astype(int)
    counts = np.bincount(charge_numbers, minlength=18)[1:18]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(1, 18), counts, color='#4CAF50', edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{height}', ha='center')

    ax.set(title='Количество ионов различной кратности',
           xlabel='Кратность',
           ylabel='Количество')
    ax.grid(axis='y', linestyle='--')

    if save:
        plt.savefig(title)
    else:
        plt.show()


def electrons_visualisation(t_array, ion_array, row_gap=1000, save=False, title='electron_visualisation.pdf'):

    # Создаем массив для столбцов
    n_segments = len(t_array) // row_gap
    bar_array = np.zeros((n_segments, z_max))
    x_array = t_array[::row_gap]

    # Вычисляем сумму ионизаций для каждого промежутка
    for i in range(0, len(t_array), row_gap):
        bar_array[i // row_gap] = np.sum(ion_array[i:i + row_gap], axis=0)

    # Рассчитываем пределы
    max_electrons = np.max(np.sum(bar_array, axis=1))
    y_upper_limit = max(max_electrons, np.exp(-(t_array / tau) ** 2).max() * max_electrons) * 1.1

    # Цвета
    colors = ['#440154', '#481567', '#482677', '#453781', '#404788',
              '#39558C', '#33638D', '#2D708E', '#287D8E', '#238A8D',
              '#1F968B', '#20A387', '#29AF7F', '#3CBB75', '#55C677',
              '#73D055', '#95D840', '#B8DE29']

    # Создаем график
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set(xlim=(t_array[0], t_array[-1]), ylim=(0, y_upper_limit))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Рисуем столбцы
    width = t_array[row_gap] - t_array[0]
    bottom = np.zeros_like(x_array)

    for k in range(z_max):
        ax.bar(x_array, bar_array[:, k], bottom=bottom, color=colors[k], width=width)
        bottom += bar_array[:, k]

    # Рисуем огибающую
    envelope = np.exp(-(t_array / tau) ** 2) * max_electrons
    ax.plot(t_array, envelope, 'k-', linewidth=2, label='Огибающая')

    bounds = np.arange(z_max + 1)
    norm = BoundaryNorm(bounds, len(colors))
    cbar = ColorbarBase(plt.axes([0.92, 0.15, 0.02, 0.7]),
                        cmap=ListedColormap(colors),
                        norm=norm,
                        ticks=np.arange(z_max) + 0.5)
    cbar.set_ticklabels(np.arange(1, z_max + 1))
    cbar.set_label('Кратность ионизации', fontsize=12)

    # Подписи
    ax.set_xlabel('Время', fontsize=12)
    ax.set_ylabel('Количество электронов', fontsize=12)
    ax.legend()

    if save:
        plt.savefig(title)
    else:
        plt.show()


def electrons_momenta_distribution(p_x, p_y, save=False, title='импульсный спектр, интенсивность 1,5.pdf'):

    plt.figure(figsize=(10, 10))
    plt.scatter(p_x, p_y, color='blue', s=10, alpha=0.5)

    plt.xlim(-max(np.abs(p_x)) * 1.1, max(np.abs(p_x)) * 1.1)
    plt.ylim(-max(np.abs(p_y)) * 1.1, max(np.abs(p_y)) * 1.1)

    plt.xlabel('p_x')
    plt.ylabel('p_y')

    plt.grid(True)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    if save:
        plt.savefig(title)
    else:
        plt.show()


def electrons_angle_distribution(angles, save=False, title='angle_distribution.pdf'):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')

    ax.hist(np.array(angles), bins=36, alpha=0.7, color='red')

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.set_title('Углы вылета электронов', pad=20)

    if save:
        plt.savefig(title)
    else:
        plt.show()


def electrons_energy_distribution(energies, number_of_bins=100, save=False, title='energy_distribution.pdf'):
    mev_energies = energies * 0.5
    print('max energy, MeV =', np.max(mev_energies))
    print('min energy, MeV =', np.min(mev_energies))

    plt.figure(figsize=(10, 6))

    log_min = np.log10(np.min(mev_energies))
    log_max = np.log10(np.max(mev_energies))

    # Используем логарифмические бины
    log_bins = np.logspace(log_min, log_max, number_of_bins)

    counts, bins = np.histogram(mev_energies, log_bins)
    counts = counts / np.size(mev_energies)  # нормируем на общее число электронов

    plt.loglog(bins[1:], counts)
    plt.xlabel('Энергия электронов, МэВ')
    plt.ylabel('dN/NdE')

    if save:
        plt.savefig(title)
    else:
        plt.show()


def electrons_energy_distribution_different(energies, number_of_bins=100, save=False, title='energy_distribution.pdf'):
    mev_energies = energies * 0.5
    print('max energy, MeV =', np.max(mev_energies))
    print('min energy, MeV =', np.min(mev_energies))

    plt.figure(figsize=(10, 6))

    lower_limit = 1
    upper_limit = 1e3

    log_bins = np.logspace(np.log10(lower_limit), np.log10(upper_limit), number_of_bins)

    counts, bins = np.histogram(mev_energies, log_bins)
    counts = counts / np.size(mev_energies)

    plt.loglog(bins[:-1], counts)
    plt.xlabel('Энергия электронов, МэВ')
    plt.ylabel('dN/NdE')

    if save:
        plt.savefig(title)
    else:
        plt.show()


def electrons_energy_distribution_gpt(energies, number_of_bins=100, save=False, title='energy_distribution.pdf'):
    mev_energies = energies * 0.5
    print('max energy, MeV =', np.max(mev_energies))
    print('min energy, MeV =', np.min(mev_energies))

    plt.figure(figsize=(10, 6))

    log_min = np.log10(np.min(mev_energies))
    log_max = np.log10(np.max(mev_energies))

    log_bins = np.logspace(log_min, log_max, number_of_bins + 1)

    counts, bins = np.histogram(mev_energies, log_bins)

    bin_widths = np.diff(bins)
    counts = counts / (np.size(mev_energies) * bin_widths)

    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    plt.loglog(bin_centers, counts)

    plt.xlabel('Энергия электронов, МэВ')
    plt.ylabel('dN/NdE')

    if save:
        plt.savefig(title)
    else:
        plt.show()


def ions_visualization(atoms):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = atoms[:, 0]
    y = atoms[:, 1]
    z = atoms[:, 2]

    color_data = atoms[:, 3]

    scatter = ax.scatter(x, y, z, c=color_data, s=50, alpha=0.8)
    color_bar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    color_bar.set_label('Кратность иона')

    plt.tight_layout()
    plt.show()
