from scipy.special import gamma
from scipy.integrate import solve_ivp
import os
from datetime import datetime
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

n_star_array = ionization_order_array[:, 0] / np.sqrt(2 * ionization_potentials)


input_array = np.column_stack((
    n_star_array,
    ionization_order_array[:, 1],  # значения l
    ionization_order_array[:, 2]   # значения m
))


# Функции, считающая коэффициенты C и B в w_ppt и w_ppt_vectorized
def c_n_l_squared(elem):
    return 2 ** (2 * elem[:, 0] - 2) / (elem[:, 0] * gamma(elem[:, 0] + elem[:, 1] + 1) *
                                        gamma(elem[:, 0] - elem[:, 1]))


def b_l_m(elem):
    return (2 * elem[:, 1] + 1) * gamma(elem[:, 1] + abs(elem[:, 2]) + 1)\
           / (2 ** abs(elem[:, 2]) * gamma(abs(elem[:, 2]) + 1) * gamma(elem[:, 1] - abs(elem[:, 2]) + 1))


# Заполнение массивов коэффициентов B и C
c_n_l_array = c_n_l_squared(input_array)
c_n_l_array[0] = 1.

b_l_m_array = b_l_m(input_array)


def placing_cells(number, start_charge_number):  # генерация атомов
    placed_cells = np.zeros((number, 7))
    placed_cells[:, 3:] = ionization_order_array[start_charge_number]  # начальные значения Z, l, m, g_|m|

    positions = np.random.randint(-100, 101, (number, 3))
    positions[:, 0] = np.where(positions[:, 0] == 0., 1., positions[:, 0])  # Избегаем x=0

    positions[:, 0] = positions[:, 0] * (form[0] / 2)  # Настраиваем форму мишени
    positions[:, 1] = positions[:, 1] * (form[1] / 2)
    positions[:, 2] = positions[:, 2] * (form[2] / 2)

    placed_cells[:, :3] = positions * step

    return placed_cells


def placing_cells_different(number, start_charge_number):
    placed_cells = np.empty((number, 7))

    positions = np.random.uniform(-0.5, 0.5, (number, 3)) * form
    positions[:, 0] = np.where(np.abs(positions[:, 0]) < 1e-10, 1e-6, positions[:, 0])

    placed_cells[:, :3] = positions
    placed_cells[:, 3:] = ionization_order_array[start_charge_number]

    return placed_cells


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
    right_or_left = +1

    cos_comp = spatial_structure * envelope * np.cos(phase)
    sin_comp = spatial_structure * envelope * np.sin(phase)

    e_field = [0,
               a_0 * cos_comp * ellipticity[0],
               a_0 * sin_comp * ellipticity[1] * right_or_left]

    h_field = [0,
               -a_0 * sin_comp * ellipticity[1] * right_or_left,
               a_0 * cos_comp * ellipticity[0]]

    return e_field, h_field


def pulse_field_vectorized(time, vector_array):
    x_array, y_array, z_array = vector_array[:, 0], vector_array[:, 1], vector_array[:, 2]
    r_squared_array = y_array * y_array + z_array * z_array

    x_r = np.pi * w_0 * w_0  # рэлеевская длина волны
    number = np.shape(vector_array)[0]

    x_array_x_r = x_array / x_r
    x_r_x_array = x_r / x_array

    phi = np.arctan(x_array_x_r)
    rho = x_array * (1 + x_r_x_array * x_r_x_array)
    phase = 2 * np.pi * x_array - time - phi + np.pi * r_squared_array / rho
    radius = np.sqrt(1 + x_array_x_r * x_array_x_r)
    spatial_structure = 1 / radius * np.exp(-r_squared_array / (w_0 * radius) ** 2)

    envelope = np.exp(-(time - 2 * np.pi * x_array) ** 2 / tau ** 2)

    ellipticity = np.array([1 / np.sqrt(1 + eps * eps), eps / np.sqrt(1 + eps * eps)])
    right_or_left = +1

    cos_comp_array = spatial_structure * envelope * np.cos(phase)
    sin_comp_array = spatial_structure * envelope * np.sin(phase)

    field_array = np.array([np.zeros(number), cos_comp_array * ellipticity[0],
                            sin_comp_array * ellipticity[1] * right_or_left]) * atomic_field

    abs_values_array = np.sqrt(np.sum(field_array * field_array, axis=0))

    return abs_values_array


def w_ppt(moment_of_time, particle):  # устаревшая функция
    rad_vec = particle[:3]
    state = particle[3:]

    current_index = int(state[0] - 1)

    electric_field = [component * atomic_field / a_0 for component in pulse_field(moment_of_time, rad_vec)[0]]
    abs_value_of_electric_field = np.sqrt(electric_field[1] ** 2 + electric_field[2] ** 2)

    i_p = ionization_potentials[current_index]
    field_char = (2 * i_p) ** (3 / 2)  # Характерное поле в атомных единицах

    field = abs_value_of_electric_field / field_char

    return 4 * c_n_l_array[current_index] * b_l_m_array[current_index] * i_p * \
           (2 / field) ** (2 * n_star_array[current_index] - abs(state[2]) - 1) * \
           np.exp(-2 / (3 * field)) * state[3] * delta_t


def w_ppt_vectorized(moment_of_time, particles):
    r_array = particles[:, :3]
    states_array = particles[:, 3:]

    current_index_array = states_array[:, 0].astype(int) - 1

    abs_value_of_electric_field_array = pulse_field_vectorized(moment_of_time, r_array)

    i_p_array = ionization_potentials[current_index_array]

    field_char_array = (2 * i_p_array) ** (3 / 2)  # Характерное поле в атомных единицах
    field_array = abs_value_of_electric_field_array / field_char_array

    return 4 * c_n_l_array[current_index_array] * b_l_m_array[current_index_array] * i_p_array * \
           (2 / field_array) ** (2 * n_star_array[current_index_array] - abs(states_array[:, 2]) - 1) * \
           np.exp(-2 / (3 * field_array)) * states_array[:, 3] * delta_t


def placed_cells_ionization(atoms):
    fully_ionized_states_set = set()

    electron_motion_list = []

    # Цикл с ионизацией атомов, без вычисления движения электронов
    for i, current_moment in enumerate(time_array):

        progress = (i + 1) / len(time_array) * 100  # прогресс обратботки массива в процентах
        print(f'\rПрогресс: {i + 1}/{len(time_array)} ({progress:.1f}%)', end='')  # вывод значения прогресса

        # генерация случайных чисел для каждого атома для сравнения с w_ppt на каждом шаге по времени
        random_values = np.random.random(n_atoms)
        # вероятность ионизации каждого атома в момент времени current_moment:
        probabilities = w_ppt_vectorized(current_moment, atoms)  # уже умножено на delta_t (см. ion_library)
        mask = random_values < probabilities  # работаем только с теми атомами, которые ионизовались в данный момент
        true_indices = np.where(mask)[0]  # массив индексов атомов/ионов, которые ПО ПОЛОЖЕНИЮ В ПРОСТРАНСТВЕ
        # подходят для ионизации. Часть индексов может соответствовать полностью ионизованным состояниям, которые
        # необходимо исключить из рассмотрения в цикле ниже.
        not_fully_ionized_states = np.array([index for index in true_indices
                                             if index not in fully_ionized_states_set])  # исключение
        # индексов полностью ионизованных состояний их списка индексов, где верна маска mask
        for j in not_fully_ionized_states:  # j пробегает значения тех индексов, где в массиве mask стоит True,
            # и при этом их нет в списке индексов, соответствующих полностью ионизованным состояниям.
            if atoms[j, 3] == z_max:  # обрабатываем состояния с зарядом +17 с одним электроном
                fully_ionized_states_set.add(j)
                ionization_array[i, int(atoms[j, 3]) - 1] += 1
                electron_motion_list.append(np.append(current_moment, atoms[j, :3]))
                # не инкрементируем placed_cells[j, 3] - больше некуда
                continue
            ionization_array[i, int(atoms[j, 3]) - 1] += 1  # инкрементируем число электронов в данный момент
            # времени для данного потенциала ионизации
            electron_motion_list.append(np.append(current_moment, atoms[j, :3]))  # готовим список из начальных
            # положений и моментов ионизации электронов для последующих параллельных вычислений
            # при решении уравнений Лоренца
            atoms[j, 3:] = ionization_order_array[int(atoms[j, 3])]  # переводим атом/ион в следующее
            # состояние

    motion_data = np.array(electron_motion_list)
    number_of_electrons = len(motion_data)
    number_of_fully_ionized_states = len(fully_ionized_states_set)
    return motion_data, number_of_electrons, number_of_fully_ionized_states


def placed_cells_ionization_without_list(atoms):
    fully_ionized_states_set = set()

    electron_motion_array = np.zeros((z_max * len(atoms), 4))  # предвыделение памяти
    motion_counter = 0

    # Цикл с ионизацией атомов, без вычисления движения электронов
    for i, current_moment in enumerate(time_array):

        progress = (i + 1) / len(time_array) * 100  # прогресс обратботки массива в процентах
        print(f'\rПрогресс: {i + 1}/{len(time_array)} ({progress:.1f}%)', end='')  # вывод значения прогресса

        # генерация случайных чисел для каждого атома для сравнения с w_ppt на каждом шаге по времени
        random_values = np.random.random(n_atoms)
        # вероятность ионизации каждого атома в момент времени current_moment:
        probabilities = w_ppt_vectorized(current_moment, atoms)  # уже умножено на delta_t (см. ion_library)
        mask = random_values < probabilities  # работаем только с теми атомами, которые ионизовались в данный момент
        true_indices = np.where(mask)[0]  # массив индексов атомов/ионов, которые ПО ПОЛОЖЕНИЮ В ПРОСТРАНСТВЕ
        # подходят для ионизации. Часть индексов может соответствовать полностью ионизованным состояниям, которые
        # необходимо исключить из рассмотрения в цикле ниже.
        not_fully_ionized_states = np.array([index for index in true_indices
                                             if index not in fully_ionized_states_set])  # исключение
        # индексов полностью ионизованных состояний их списка индексов, где верна маска mask
        for j in not_fully_ionized_states:  # j пробегает значения тех индексов, где в массиве mask стоит True,
            # и при этом их нет в списке индексов, соответствующих полностью ионизованным состояниям.
            if atoms[j, 3] == z_max:  # обрабатываем состояния с зарядом +17 с одним электроном
                fully_ionized_states_set.add(j)
                ionization_array[i, int(atoms[j, 3]) - 1] += 1
                motion_data_element = np.append(current_moment, atoms[j, :3])
                electron_motion_array[motion_counter] = motion_data_element
                motion_counter += 1
                # не инкрементируем placed_cells[j, 3] - больше некуда
                continue
            ionization_array[i, int(atoms[j, 3]) - 1] += 1  # инкрементируем число электронов в данный момент
            # времени для данного потенциала ионизации
            motion_data_element = np.append(current_moment, atoms[j, :3])
            electron_motion_array[motion_counter] = motion_data_element  # готовим массив из начальных
            # положений и моментов ионизации электронов для последующих параллельных вычислений
            # при решении уравнений Лоренца
            motion_counter += 1
            atoms[j, 3:] = ionization_order_array[int(atoms[j, 3])]  # переводим атом/ион в следующее
            # состояние

    motion_data = electron_motion_array[:motion_counter]
    number_of_fully_ionized_states = len(fully_ionized_states_set)
    return motion_data, motion_counter, number_of_fully_ionized_states


def lorenz_equation(time, variables_vector):
    x, y, z, momenta_x, momenta_y, momenta_z = variables_vector

    current_position = np.array([x, y, z])
    electric, magnetic = pulse_field(time, current_position)

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


def solve_lorenz_motion(t_start, initial_r_v):
    t_stop = t_0
    t_span = (t_start, t_stop)

    sol = solve_ivp(
        lorenz_equation,
        t_span,
        np.hstack((initial_r_v, np.zeros(3))),
        method='RK45',
        vectorized=True
    )
    return sol.y[3, -1], sol.y[4, -1], sol.y[5, -1]


def solve_electron_trajectory(t_start, initial_r_v):
    t_stop = t_0
    t_span = (t_start, t_stop)
    t_eval = np.arange(t_start, t_stop, delta_t)

    sol = solve_ivp(
        lorenz_equation,
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
        lorenz_equation,
        t_span,
        np.hstack((initial_r_v, np.zeros(3))),
        t_eval=t_eval,
        method='RK45',
        vectorized=True
    )
    return sol.t, sol.y[3], sol.y[4], sol.y[5]


def single_electron_process(initial_data):
    initial_t = initial_data[0]
    initial_position = initial_data[1:]
    return solve_lorenz_motion(initial_t, initial_position)


def parallel_simulation(ionization_data, n_processes=None):  # возвращает массивы Numpy!!!
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
