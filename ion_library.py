from scipy.special import gamma
from scipy.integrate import solve_ivp
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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


def placing_cells(number):
    # Генерация атомов
    placed_cells = np.zeros((number, 7))
    placed_cells[:, 3:] = ionization_order_array[0]  # начальные значения Z, l, m, g_|m|

    positions = np.random.randint(-100, 101, (number, 3))  # от -w_0 до w_0
    positions[:, 0] = np.where(positions[:, 0] == 0, 1, positions[:, 0])  # Избегаем x=0

    positions[:, 0] = positions[:, 0] * (form[0] / 2)  # Настраиваем форму мишени
    positions[:, 1] = positions[:, 1] * (form[1] / 2)
    positions[:, 2] = positions[:, 2] * (form[2] / 2)

    placed_cells[:, :3] = positions * step

    return placed_cells


def pulse_field(time, rad_vec):
    x_coord, y_coord, z_coord = rad_vec
    r_squared = y_coord ** 2 + z_coord ** 2

    phi = np.arctan(abs(x_coord) / (np.pi * w_0 ** 2))
    rho = abs(x_coord) * (1 + (np.pi * w_0 ** 2 / abs(x_coord)) ** 2)
    phase = 2 * np.pi * x_coord - time - phi + np.pi * r_squared / rho

    radius = np.sqrt(1 + (abs(x_coord) / (np.pi * w_0 ** 2)) ** 2)
    spatial_structure = 1 / radius * np.exp(- r_squared / (w_0 * radius) ** 2)

    envelope = np.exp(-(time - 2 * np.pi * x_coord) ** 2 / tau ** 2)

    ellipticity = [1 / np.sqrt(1 + eps ** 2), eps / np.sqrt(1 + eps ** 2)]
    right_or_left = +1

    cos_comp = spatial_structure * envelope * np.cos(phase)
    sin_comp = spatial_structure * envelope * np.sin(phase)

    e_field = [0,
               cos_comp * ellipticity[0],
               sin_comp * ellipticity[1] * right_or_left]

    h_field = [0,
               -sin_comp * ellipticity[1] * right_or_left,
               cos_comp * ellipticity[0]]

    return e_field, h_field


def pulse_field_vectorized(time, vector_array):
    x_array, y_array, z_array = vector_array[:, 0], vector_array[:, 1], vector_array[:, 2]
    r_squared_array = y_array ** 2 + z_array ** 2

    number = np.shape(vector_array)[0]

    phi = np.arctan(abs(x_array) / (np.pi * w_0 ** 2))
    rho = abs(x_array) * (1 + (np.pi * w_0 ** 2 / abs(x_array)) ** 2)
    phase = 2 * np.pi * x_array - time - phi + np.pi * r_squared_array / rho
    radius = np.sqrt(1 + (abs(x_array) / (np.pi * w_0 ** 2)) ** 2)
    spatial_structure = 1 / radius * np.exp(- r_squared_array / (w_0 * radius) ** 2)

    envelope = np.exp(-(time - 2 * np.pi * x_array) ** 2 / tau ** 2)

    ellipticity = np.array([1 / np.sqrt(1 + eps ** 2), eps / np.sqrt(1 + eps ** 2)])
    right_or_left = +1

    cos_comp_array = spatial_structure * envelope * np.cos(phase)
    sin_comp_array = spatial_structure * envelope * np.sin(phase)

    # np.array(e_x, e_y, e_z, h_x, h_y, h_z)
    return np.array([np.zeros(number), cos_comp_array * ellipticity[0], sin_comp_array * ellipticity[1] * right_or_left,
                     np.zeros(number), -sin_comp_array * ellipticity[1] * right_or_left,
                     cos_comp_array * ellipticity[0]])


def w_ppt(moment_of_time, particle):
    rad_vec = particle[:3]
    state = particle[3:]

    current_index = int(state[0] - 1)

    electric_field = [component * atomic_field for component in pulse_field(moment_of_time, rad_vec)[0]]
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

    electric_field_array = pulse_field_vectorized(moment_of_time, r_array)[:3] * atomic_field
    abs_value_of_electric_field_array = np.sqrt(np.sum(electric_field_array ** 2, axis=0))

    i_p_array = ionization_potentials[current_index_array]

    field_char_array = (2 * i_p_array) ** (3 / 2)  # Характерное поле в атомных единицах
    field_array = abs_value_of_electric_field_array / field_char_array

    return 4 * c_n_l_array[current_index_array] * b_l_m_array[current_index_array] * i_p_array * \
           (2 / field_array) ** (2 * n_star_array[current_index_array] - abs(states_array[:, 2]) - 1) * \
           np.exp(-2 / (3 * field_array)) * states_array[:, 3] * delta_t


def lorenz_equation(time, variables_vector):
    x, y, z, momenta_x, momenta_y, momenta_z = variables_vector

    current_position = np.array([x, y, z])

    electric = [component * a_0 for component in pulse_field(time, current_position)[0]]
    magnetic = [component * a_0 for component in pulse_field(time, current_position)[1]]

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
