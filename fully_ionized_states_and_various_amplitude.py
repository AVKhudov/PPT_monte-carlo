from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt

'''
Аргон 18. Получаем зависимость числа полностью ионизованных состояний от величины поля. 
'''

# Параметры
n = 50  # Число атомов
wavelength = 0.33  # Длина волны в единицах радиуса перетяжки
tau = 5  # Время включения поля
t_0 = 45  # Длительность моделирования
cut_time = 40  # Время отсечки, после которого электрон считается свободным
delta_t = 0.1  # Разрешение по времени
step = 0.01  # Разрешение в пространстве
z_max = 18  # Максимальное зарядовое число остатка


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


def c_n_l_squared(elem):
    return 2 ** (2 * elem[0] - 2) / (elem[0] * gamma(elem[0] + elem[1] + 1) * gamma(elem[0] - elem[1]))


def b_l_m(elem):
    return (2 * elem[1] + 1) * gamma(elem[1] + abs(elem[2]) + 1) / (2 ** abs(elem[2]) * gamma(abs(elem[2]) + 1)
                                                                  * gamma(elem[1] - abs(elem[2]) + 1))


# Заполнение массивов коэффициентов B и C
c_n_l_array = np.array([c_n_l_squared(element) for element in input_array])
c_n_l_array[0] = 1.

b_l_m_array = np.array([b_l_m(element) for element in input_array])


def phi(z_coordinate):
    return np.arctan(wavelength * abs(z_coordinate) / np.pi)


def rho(z_coordinate):
    return z_coordinate * (1 + (np.pi / (wavelength * abs(z_coordinate))) ** 2)


def radius(z_coordinate):
    return np.sqrt(1 + (wavelength * abs(z_coordinate) / np.pi) ** 2)


def pulse_field(time, rad_vec, amplitude):
    return amplitude * np.cos(2 * np.pi * rad_vec[2] / wavelength - time - phi(rad_vec[2])
                  + np.pi / wavelength * (rad_vec[0] ** 2 + rad_vec[1] ** 2) / rho(rad_vec[2])) * \
           np.exp(-4 * (rad_vec[0] ** 2 + rad_vec[2] ** 2) / radius(rad_vec[2]) ** 2) * \
           np.exp(-(time - 2 * np.pi * rad_vec[2] / wavelength) ** 2 / tau ** 2) / radius(rad_vec[2])


def w_ppt(moment_of_time, particle, amplitude):

    rad_vec = particle[:3]
    state = particle[3:6]

    current_index = int(state[0] - 1)

    i_p = ionization_potentials[current_index]
    field_char = (2 * i_p) ** (3 / 2)  # Характерное поле в атомных единицах
    field = abs(pulse_field(moment_of_time, rad_vec, amplitude) / field_char)

    return 4 * c_n_l_array[current_index] * b_l_m_array[current_index] * i_p * \
           (2 / field) ** (2 * n_star_array[current_index] - abs(state[2]) - 1) * \
           np.exp(-2 / (3 * field))


# Генерация атомов
placed_cells = np.zeros((n, 7))
placed_cells[:, 3:] = ionization_order_array[0]  # начальные значения Z, l, m, g_|m|
positions = np.random.randint(-100, 101, (n, 3))
positions[:, 2] = np.where(positions[:, 2] == 0, 1, positions[:, 2])  # Избегаем z=0
placed_cells[:, :3] = positions * step

time_array = np.arange(-t_0, t_0, delta_t)


def main_part(amplitude):
    random_values = np.random.random((len(time_array), n))
    count_of_fully_ionized_states = 0

    for i, current_moment in enumerate(time_array):
        # вероятность ионизации каждого атома в момент времени current_moment:
        probabilities = np.array([w_ppt(current_moment, atom, amplitude) * delta_t for atom in placed_cells])
        mask = random_values[i] < probabilities
        true_indices_in_mask = np.where(mask)[0]

        for j in true_indices_in_mask:  # j пробегает значения тех индексов, где в массиве mask стоит True
            if placed_cells[j, 3] == z_max:
                continue
            placed_cells[j, 3:] = ionization_order_array[int(placed_cells[j, 3])]  # инкремент за счет разницы
            # между нулем и единицей

    for charge_state in placed_cells[:, 3]:
        if charge_state == z_max:
            count_of_fully_ionized_states += 1

    return count_of_fully_ionized_states


fields = np.arange(100, 2050, 50)
counts = np.array([main_part(field) for field in fields])

plt.plot(fields, counts)
plt.show()
