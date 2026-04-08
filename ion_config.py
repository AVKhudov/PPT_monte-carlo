import numpy as np
from scipy.special import gamma

'''
Аргон 18. Конфигурационный файл.
'''

# Параметры
n_atoms = 100  # число атомов
intensity = 3.e22  # интенсивность поля в фокусе в единицах Вт/см^2
w_0 = 2.654  # радиус перетяжки гауссова пучка в длинах волн
tau = 44.  # длительность импульса в единицах 1/omega
t_0 = 700.  # длительность моделирования в единицах 1/omega
delta_t = 0.001  # разрешение по времени
z_max = 18  # максимальное зарядовое число
start_charge_number_of_particles = 0  # 0 - ионизуем нейтральные атомы (начальное число Z - 1)
c = 1.  # масштаб для формы form мишени
form = np.array([4., 2., 2.]) * w_0 * c
# форма ящика с атомами в формате длина по X, длина по Y и длина по Z.
# импульс распространаяется вдоль оси X, расстояния указываются в длинах волн.

eps = 0.15  # эллиптичность поля
right_or_left = +1  # +1 - правая поляризация, -1 - левая поляризация

frequency = 2 * np.pi * 3e8 / 1e-6  # частота поля в СИ, соотв. длине волны 1 мкм
atomic_frequency_unit = 4.1e16  # атомная единица частоты в СИ
atomic_frequency = frequency / atomic_frequency_unit  # частота в атомных единицах

atomic_intensity_unit = 3.5e16  # атомная интенсивность в СИ
atomic_field = np.sqrt(intensity / (atomic_intensity_unit * (1 + eps ** 2)))  # амплитуда поля в атомных единицах

a_0 = atomic_field / (137 * atomic_frequency)  # единица поля для решения уравнений Лоренца для электронов
mass_coefficient = 8e4  # во сколько раз масса иона больше массы электрона
a_0_ion = a_0 / mass_coefficient  # единица поля для решения уравнения Лоренца для ионов (заряд не учитывается)


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

'''
ionization_order_array = np.array([
    [1, 1, 0, 2],
    [2, 1, 0, 1],
    [3, 1, -1, 4],
    [4, 1, -1, 3],
    [5, 1, 1, 2],
    [6, 1, 1, 1],
    [7, 0, 0, 2],
    [8, 0, 0, 1],
    [9, 1, 0, 2],
    [10, 1, 0, 1],
    [11, 1, -1, 4],
    [12, 1, -1, 3],
    [13, 1, 1, 2],
    [14, 1, 1, 1],
    [15, 0, 0, 2],
    [16, 0, 0, 1],
    [17, 0, 0, 2],
    [18, 0, 0, 1]
])
'''

ionization_order_array = np.array([
    [1, 1, 0, 1],
    [2, 1, 0, 1],
    [3, 1, 0, 1],
    [4, 1, 0, 1],
    [5, 1, 0, 1],
    [6, 1, 0, 1],
    [7, 0, 0, 1],
    [8, 0, 0, 1],
    [9, 1, 0, 1],
    [10, 1, 0, 1],
    [11, 1, 0, 1],
    [12, 1, 0, 1],
    [13, 1, 0, 1],
    [14, 1, 0, 1],
    [15, 0, 0, 1],
    [16, 0, 0, 1],
    [17, 0, 0, 1],
    [18, 0, 0, 1]
])


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

time_array = np.arange(-t_0, t_0, delta_t, dtype=float)  # сетка по времени
ionization_array = np.zeros((len(time_array), z_max), dtype=int)  # массив с данными о потенциалах ионизации электронов
