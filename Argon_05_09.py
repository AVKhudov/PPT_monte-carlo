from scipy.special import gamma
import numpy as np
from scipy.integrate import solve_ivp
import json
import os
from datetime import datetime


'''
Аргон 18
'''

# Параметры
n = 50  # Число атомов
intensity = 3 * 1e22  # Интенсивность поля в фокусе в единицах Вт/см^2
w_0 = 2.654  # радиус перетяжки гауссова пучка в длинах волн
tau = 44  # длительность импульса в единицах 1/omega
t_0 = 250  # Длительность моделирования в единицах 1/omega
delta_t = 0.1  # Разрешение по времени
step = 0.01  # Разрешение в пространстве
z_max = 18  # Максимальное зарядовое число
form = (4, 2, 2)  # форма ящика с атомами в формате длина по X, длина по Y и длина по Z. Импульс распространаяется
# вдоль оси X, расстояния указываются в длинах волн

# Константы
ionization_potentials = np.array([
    15.76,
    27.63,
    40.74,
    59.81,
    75.02,
    91.01,
    124.32,
    143.46,
    422.45,
    478.69,
    538.96,
    618.26,
    686.10,
    755.74,
    854.77,
    918.03,
    4120.89,
    4426.23
]) / 27.20

ionization_order_array = np.array([
    [1., 1., 0., 2.],
    [2., 1., 0., 1.],
    [3., 1., -1., 4.],
    [4., 1., -1., 3.],
    [5., 1., 1., 2.],
    [6., 1., 1., 1.],
    [7., 0., 0., 2.],
    [8., 0., 0., 1.],
    [9., 1., 0., 2.],
    [10., 1., 0., 1.],
    [11., 1., -1., 4.],
    [12., 1., -1., 3.],
    [13., 1., 1., 2.],
    [14., 1., 1., 1.],
    [15., 0., 0., 2.],
    [16., 0., 0., 1.],
    [17., 0., 0., 2.],
    [18., 0., 0., 1.]
])


n_star_array = np.array([element[0] / np.sqrt(2 * ionization_potentials[i])
                         for i, element in enumerate(ionization_order_array)])

input_array = np.array([
    [n_star_array[0], 1, 0],
    [n_star_array[1], 1, 0],
    [n_star_array[2], 1, -1],
    [n_star_array[3], 1, -1],
    [n_star_array[4], 1, 1],
    [n_star_array[5], 1, 1],
    [n_star_array[6], 0, 0],
    [n_star_array[7], 0, 0],
    [n_star_array[8], 1, 0],
    [n_star_array[9], 1, 0],
    [n_star_array[10], 1, -1],
    [n_star_array[11], 1, -1],
    [n_star_array[12], 1, 1],
    [n_star_array[13], 1, 1],
    [n_star_array[14], 0, 0],
    [n_star_array[15], 0, 0],
    [n_star_array[16], 0, 0],
    [n_star_array[17], 0, 0]
])


# Функции, считающая коэффициенты C и B в w_PPT
def c_n_l_squared(elem):
    return 2 ** (2 * elem[0] - 2) / (elem[0] * gamma(elem[0] + elem[1] + 1) * gamma(elem[0] - elem[1]))


def b_l_m(elem):
    return (2 * elem[1] + 1) * gamma(elem[1] + abs(elem[2]) + 1) / (2 ** abs(elem[2]) * gamma(abs(elem[2]) + 1)
                                                                  * gamma(elem[1] - abs(elem[2]) + 1))


# Заполнение массивов коэффициентов B и C
c_n_l_array = np.array([c_n_l_squared(element) for element in input_array])
c_n_l_array[0] = 1.

b_l_m_array = np.array([b_l_m(element) for element in input_array])


def pulse_field(time, rad_vec):
    x_coord, y_coord, z_coord = rad_vec
    r_squared = y_coord ** 2 + z_coord ** 2

    phi = np.arctan(abs(x_coord) / (np.pi * w_0 ** 2))
    rho = abs(x_coord) * (1 + (np.pi * w_0 ** 2 / abs(x_coord)) ** 2)
    phase = 2 * np.pi * x_coord - time - phi + np.pi * r_squared / rho

    radius = np.sqrt(1 + (abs(x_coord) / (np.pi * w_0 ** 2)) ** 2)
    spatial_structure = 1 / radius * np.exp(- r_squared / (w_0 * radius) ** 2)

    envelope = np.exp(-(time - 2 * np.pi * x_coord) ** 2 / tau ** 2)

    eps = 0.15
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


def w_ppt(moment_of_time, particle):
    rad_vec = particle[:3]
    state = particle[3:]

    current_index = int(state[0] - 1)

    intensity_0 = 3.5 * 10 ** 16  # Интенсивность поля, соответствующая 1 атомной единице поля
    amplitude = (intensity / intensity_0) ** (1 / 2)  # Перевод интенсивности поля в амплитуду поля В АТОМНЫХ ЕДИНИЦАХ

    electric_field = [component * amplitude for component in pulse_field(moment_of_time, rad_vec)[0]]
    abs_value_of_electric_field = np.sqrt(electric_field[1] ** 2 + electric_field[2] ** 2)

    i_p = ionization_potentials[current_index]
    field_char = (2 * i_p) ** (3 / 2)  # Характерное поле в атомных единицах

    field = abs_value_of_electric_field / field_char

    return 4 * c_n_l_array[current_index] * b_l_m_array[current_index] * i_p * \
           (2 / field) ** (2 * n_star_array[current_index] - abs(state[2]) - 1) * \
           np.exp(-2 / (3 * field)) * state[3]


def w_ppt_other(moment_of_time, particle):
    rad_vec = particle[:3]
    state = particle[3:]

    current_index = int(state[0] - 1)

    intensity_0 = 3.5 * 10 ** 16  # Интенсивность поля, соответствующая 1 атомной единице поля
    amplitude = (intensity / intensity_0) ** (1 / 2)  # Перевод интенсивности поля в амплитуду поля В АТОМНЫХ ЕДИНИЦАХ

    # electric_field = amplitude * pulse_field(moment_of_time, rad_vec)[0]
    electric_field = [component * amplitude for component in pulse_field(moment_of_time, rad_vec)[0]]
    abs_value_of_electric_field = np.sqrt(electric_field[1] ** 2 + electric_field[2] ** 2)

    i_p = ionization_potentials[current_index]
    field_char = (2 * i_p) ** (3 / 2)  # Характерное поле в атомных единицах

    field = abs_value_of_electric_field / field_char

    return c_n_l_array[current_index] * b_l_m_array[current_index] * i_p * \
           field ** (1 + abs(state[2]) - 2 * n_star_array[current_index]) * \
           np.exp(-2 / (3 * field)) * state[3]


def lorenz_equation(time, variables_vector):
    x, y, z, momenta_x, momenta_y, momenta_z = variables_vector

    current_position = np.array([x, y, z])

    a_0 = 130.

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


def angle_calculation(y_comp, x_comp):
    if (y_comp > 0 and x_comp > 0) or (y_comp < 0 and x_comp > 0):
        return np.arctan(y_comp / x_comp)
    if y_comp > 0 and x_comp < 0:
        return np.pi + np.arctan(y_comp / x_comp)
    if y_comp < 0 and x_comp < 0:
        return np.arctan(y_comp / x_comp) - np.pi


def create_timestamp_folder(base_path=r'C:\Users\Dns\Desktop\MC_Ion'):
    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    folder_path = os.path.join(base_path, timestamp)  # созадем путь к папке
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_file(folder_path, filename, data):
    file_path = os.path.join(folder_path, filename)  # создаем путь к файлу
    np.save(file_path, data)


# Генерация атомов
placed_cells = np.zeros((n, 7))
placed_cells[:, 3:] = ionization_order_array[0]  # начальные значения Z, l, m, g_|m|

positions = np.random.randint(-100, 101, (n, 3))  # от -w_0 до w_0
positions[:, 0] = np.where(positions[:, 0] == 0, 1, positions[:, 0])  # Избегаем x=0

positions[:, 0] = positions[:, 0] * (form[0] / 2)  # Настраиваем форму мишени
positions[:, 1] = positions[:, 1] * (form[1] / 2)
positions[:, 2] = positions[:, 2] * (form[2] / 2)

placed_cells[:, :3] = positions * step

time_array = np.arange(-t_0, t_0, delta_t)
ionization_array = np.zeros((len(time_array), z_max))  # массив с данными о потенциалах ионизации электронов

# Основной цикл
all_p_x = []
all_p_y = []
all_p_z = []

all_energies = []
all_angles = []

fully_ionized_states_list = []
probabilities_list = []

file = open(r'C:\Users\Dns\Desktop\MC_Ion\placed_cells.json', 'w')  # открываем файл, куда будут заноситься данные о
# начальных координатах и начальных моментах времени вылетающего электрона

for i, current_moment in enumerate(time_array):

    progress = (i + 1) / len(time_array) * 100  # прогресс обратботки массива в процентах
    print(f'\rПрогресс: {i + 1}/{len(time_array)} ({progress:.1f}%)', end='')  # вывод значения прогресса

    # генерация случайных чисел для каждого атома для сравнения с w_ppt на каждом шаге по времени
    random_values = np.random.random((len(time_array), n))
    # вероятность ионизации каждого атома в момент времени current_moment:
    current_prob_list = [w_ppt_other(current_moment, atom) * delta_t for atom in placed_cells]
    probabilities_list.extend(current_prob_list)
    probabilities = np.array(current_prob_list)
    # работаем только с теми атомами, которые ионизовались на данном шаге по времени
    mask = random_values[i] < probabilities
    true_indices_in_mask = np.where(mask)[0]  # массив индексов атомов/ионов, которые ПО ПОЛОЖЕНИЮ В ПРОСТРАНСТВЕ
    # подходят для ионизации. Часть индексов может соответствовать полностью ионизованным состояниям, которые
    # необходимо исключить из рассмотрения в цикле ниже.

    true_indices_in_mask_list = true_indices_in_mask.tolist()

    not_fully_ionized_states_list = [index for index in true_indices_in_mask_list
                                     if index not in fully_ionized_states_list]  # исключение
    # индексов полностью ионизованных состояний их списка индексов, где верна маска mask

    not_fully_ionized_states_array = np.array(not_fully_ionized_states_list)

    for j in not_fully_ionized_states_array:  # j пробегает значения тех индексов, где в массиве mask стоит True,
        # и при этом их нет в списке индексов, соответствующих полностью ионизованным состояниям.
        if placed_cells[j, 3] == z_max:
            fully_ionized_states_list.append(j)
            continue
        ionization_array[i, int(placed_cells[j, 3]) - 1] += 1  # инкрементируем число электронов в данный момент
        # времени для данного потенциала ионизации
        json.dump([placed_cells[j].tolist(), current_moment], file)  # работа с файлом
        file.write('\n')  # делаем запись в файл с переходом на новую строку
        placed_cells[j, 3:] = ionization_order_array[int(placed_cells[j, 3])]  # переводим атом/ион в следующее
        # состояние

        p_x, p_y, p_z = solve_lorenz_motion(current_moment, placed_cells[j, :3])  # счет импульсов вылетающего электрона

        all_p_x.append(p_x)
        all_p_y.append(p_y)
        all_p_z.append(p_z)

        all_energies.append(np.sqrt(1 + p_x ** 2 + p_y ** 2 + p_z ** 2))

        all_angles.append(angle_calculation(p_x, p_y))

file.close()

number_of_electrons = len(all_p_x)
number_of_fully_ionized_states = len(fully_ionized_states_list)

print('\n')
print('number of electrons =', number_of_electrons)
print('number of full ionized states =', number_of_fully_ionized_states)

# Вывод данных в файлы:

output_path = create_timestamp_folder()  # создаем папку с текущей датой и временем

params = [n, tau, t_0, delta_t, step, z_max]  # те параметры, которые необходимые для обработки данных
save_file(output_path, 'params.npy', np.array(params))

save_file(output_path, 'time_array.npy', time_array)
save_file(output_path, 'ionization_array.npy', ionization_array)

save_file(output_path, 'placed_cells.npy', placed_cells)

save_file(output_path, 'all_electron_p_x.npy', np.array(all_p_x))
save_file(output_path, 'all_electron_p_y.npy', np.array(all_p_y))
save_file(output_path, 'all_electron_p_z.npy', np.array(all_p_z))
save_file(output_path, 'all_electron_energies.npy', np.array(all_energies))
save_file(output_path, 'all_electron_angles.npy', np.array(all_angles))
save_file(output_path, 'probabilities_of_ionization.npy', np.array(probabilities_list))
