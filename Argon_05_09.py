from scipy.special import gamma
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import MaxNLocator

'''
Аргон 18
'''

# Параметры
n = 100  # Число атомов
intensity = 1e21  # Интенсивность поля в фокусе в единицах Вт/см^2
focus_radius = 3  # Радиус фокусировки в мкм. Длина волны излучения фиксирована и равна 0.8 мкм
tau = 43.995  # длительность импульса
t_0 = 120  # Длительность моделирования
cut_time = 90  # Время отсечки, после которого электрон считается свободным
delta_t = 0.1  # Разрешение по времени
step = 0.01  # Разрешение в пространстве
z_max = 18  # Максимальное зарядовое число

form = (2, 2, 2)  # форма ящика с атомами в формате длина по X, длина по Y и длина по Z. Импульс распространаяется
# вдоль оси X

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

wavelength = 1/focus_radius  # Длина волны в единицах радиуса перетяжки


def phi(x_coordinate):
    return np.arctan(wavelength * abs(x_coordinate) / np.pi)


def rho(x_coordinate):
    return x_coordinate * (1 + (np.pi / (wavelength * abs(x_coordinate))) ** 2)


def radius(x_coordinate):
    return np.sqrt(1 + (wavelength * abs(x_coordinate) / np.pi) ** 2)


intensity_0 = 3.5 * 10 ** 16

amplitude = (intensity / intensity_0) ** (1/2)


def pulse_field(time, rad_vec):
    return amplitude * np.cos(2 * np.pi * rad_vec[0] / wavelength - time - phi(rad_vec[0])
                              + np.pi / wavelength * (rad_vec[1] ** 2 + rad_vec[2] ** 2) / rho(rad_vec[0]) + np.pi / 2) * \
           np.exp(-4 * (rad_vec[1] ** 2 + rad_vec[0] ** 2) / radius(rad_vec[0]) ** 2) * \
           np.exp(-(time - 2 * np.pi * rad_vec[0] / wavelength) ** 2 / tau ** 2) / radius(rad_vec[0])


def w_ppt(moment_of_time, particle):

    rad_vec = particle[:3]
    state = particle[3:6]

    current_index = int(state[0] - 1)

    i_p = ionization_potentials[current_index]
    field_char = (2 * i_p) ** (3 / 2)  # Характерное поле в атомных единицах
    field = abs(pulse_field(moment_of_time, rad_vec) / field_char)

    return 4 * c_n_l_array[current_index] * b_l_m_array[current_index] * i_p * \
           (2 / field) ** (2 * n_star_array[current_index] - abs(state[2]) - 1) * \
           np.exp(-2 / (3 * field))


def bar_plotter():

    charge_numbers = placed_cells[:, 3].astype(int)
    counts = np.bincount(charge_numbers, minlength=11)[1:11]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(1, 11), counts, color='#4CAF50', edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{height}', ha='center')

    ax.set(title='Количество ионов различной кратности',
           xlabel='Кратность',
           ylabel='Количество')
    ax.grid(axis='y', linestyle='--')
    plt.savefig(f'статистика, длина волны {wavelength}, амплитуда {amplitude}.pdf')
    #plt.show()


def momenta_plotter(save=False):
    plt.figure(figsize=(10, 10))
    plt.scatter(all_p_x, all_p_y, color='blue', s=10, alpha=0.5)
    plt.xlim(-max(np.abs(all_p_x)) * 1.1, max(np.abs(all_p_x)) * 1.1)
    plt.ylim(-max(np.abs(all_p_y)) * 1.1, max(np.abs(all_p_y)) * 1.1)
    plt.xlabel('p_x')
    plt.ylabel('p_y')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    if save:
        plt.savefig('импульсный спектр, интенсивность 1,5.pdf')
    else:
        plt.show()


def electrons_angle_distribution():

    # Создаем полярный график
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')

    # Строим гистограмму
    ax.hist(np.array(all_angles), bins=36, alpha=0.7, color='red')

    # Настройки
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.set_title('Углы вылета электронов', pad=20)

    plt.show()


def velocity_plotter():
    plt.figure(figsize=(10, 10))
    plt.scatter(all_v_x, all_v_y, color='blue', s=10, alpha=0.5)
    plt.xlim(-max(np.abs(all_v_x)) * 1.1, max(np.abs(all_v_x)) * 1.1)
    plt.ylim(-max(np.abs(all_v_y)) * 1.1, max(np.abs(all_v_y)) * 1.1)
    plt.title(f'Импульсное распределение (всего электронов: {len(all_v_x)}),'
              f'длина волны: {wavelength}, амплитуда: {amplitude}')
    plt.xlabel('v_x')
    plt.ylabel('v_z')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.savefig(f'ск. распред. длина волны {wavelength}, амплитуда {amplitude}.pdf')
    #plt.show()


def electrons_visualisation(row_gap=100):
    # Создаем массив для столбцов
    bar_array = np.zeros((int(len(time_array) / row_gap), z_max))
    x_array = time_array[0:len(time_array):row_gap]

    # Вычисляем сумму ионизаций для каждого промежутка
    for i in range(0, len(time_array), row_gap):
        bar_array[int(i / row_gap)] = np.sum(ionization_array[i:i + row_gap], axis=0)

    max_number_of_electrons = np.max(np.sum(bar_array, axis=1))
    max_envelope_value = np.max(np.exp(-(time_array / tau) ** 2) * max_number_of_electrons)

    # Увеличиваем верхний предел на 10% от максимального значения
    y_upper_limit = max(max_number_of_electrons, max_envelope_value) * 1.1

    color_array = np.array(['#440154', '#481567', '#482677', '#453781', '#404788',
                            '#39558C', '#33638D', '#2D708E', '#287D8E', '#238A8D',
                            '#1F968B', '#20A387', '#29AF7F', '#3CBB75', '#55C677',
                            '#73D055', '#95D840', '#B8DE29'])

    fig, ax = plt.subplots(figsize=(14, 6))  # Увеличиваем ширину для colorbar

    # Устанавливаем границы осей
    ax.set_ylim([0, y_upper_limit])
    ax.set_xlim([time_array[0], time_array[-1]])

    width = time_array[row_gap] - time_array[0]  # Ширина столбцов по размеру промежутка

    # Рисуем столбцы
    bottom_positions = np.zeros_like(x_array)
    for k in range(bar_array.shape[1]):
        ax.bar(x_array, bar_array.T[k], bottom=bottom_positions, color=color_array[k], width=width)
        bottom_positions += bar_array.T[k]

    # Рисуем огибающую
    ax.plot(time_array, np.exp(-(time_array / tau) ** 2) * max_number_of_electrons,
            color='black', linewidth=2, label='Огибающая')

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # СОЗДАЕМ COLORBAR ДЛЯ STACKED BAR CHART
    custom_cmap = mcolors.ListedColormap(color_array)

    # Создаем нормализацию для дискретных значений
    bounds = np.arange(len(color_array) + 1)
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

    # Создаем ScalarMappable объект для colorbar
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])

    # Добавляем colorbar
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation='vertical',
        shrink=0.8,
        pad=0.02,
        ticks=np.arange(len(color_array)) + 0.5  # Центрируем метки по середине цветовых сегментов
    )

    # Настраиваем подписи - целые числа от 0 до количества цветов - 1
    cbar.set_ticklabels(np.arange(1, 19))
    cbar.set_label('Кратность ионизации', fontsize=12)

    # Добавляем легенду и подписи
    ax.set_xlabel('Время', fontsize=12)
    ax.set_ylabel('Количество электронов', fontsize=12)

    plt.tight_layout()
    plt.show()


def ions_visualization():

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = placed_cells[:, 0]
    y = placed_cells[:, 1]
    z = placed_cells[:, 2]

    color_data = placed_cells[:, 3]

    scatter = ax.scatter(x, y, z, c=color_data, s=50, alpha=0.8)
    color_bar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    color_bar.set_label('Кратность иона')

    plt.tight_layout()
    plt.show()


def motion_equation(time, variables_vector, radius_vector):
    alpha = 1 / 137
    v_x, v_y = variables_vector
    reversed_gamma_factor = np.sqrt(1 - v_x ** 2 - v_y ** 2)
    field = pulse_field(time, radius_vector)

    v_x_eq = alpha * reversed_gamma_factor * field * (v_y - v_x * v_y)
    v_y_eq = alpha * reversed_gamma_factor * field * (1 - v_x - v_y ** 2)
    return np.array([v_x_eq, v_y_eq])


def solve_motion(t_start, radius_vector):
    t_stop = cut_time + delta_t if np.less(t_start, cut_time) else t_0

    t_eval = np.arange(t_start, t_stop - delta_t, delta_t)  # должно быть внутри t_span
    t_span = (t_start - delta_t, t_stop)

    sol = solve_ivp(
        lambda t, y: motion_equation(t, y, radius_vector),
        t_span,
        np.zeros(2),
        t_eval=t_eval,
        vectorized=True
    )
    return sol.y[0, -1], sol.y[1, -1]


def angle_calculation(y_comp, x_comp):
    if (y_comp > 0 and x_comp > 0) or (y_comp < 0 and x_comp > 0):
        return np.arctan(y_comp / x_comp)
    if y_comp > 0 and x_comp < 0:
        return np.pi + np.arctan(y_comp / x_comp)
    if y_comp < 0 and x_comp < 0:
        return np.arctan(y_comp / x_comp) - np.pi


# Генерация атомов
placed_cells = np.zeros((n, 7))
placed_cells[:, 3:] = ionization_order_array[0]  # начальные значения Z, l, m, g_|m|
positions = np.random.randint(-100, 101, (n, 3))  # от -w_0 до w_0
positions[:, 0] = np.where(positions[:, 0] == 0, 1, positions[:, 0])  # Избегаем x=0

positions[:, 0] = positions[:, 0] * (form[0] / 2)  # Настраиваем форму мишени
positions[:, 1] = positions[:, 1] * (form[1] / 2)
positions[:, 2] = positions[:, 2] * (form[2] / 2)

placed_cells[:, :3] = positions * step


# Основной цикл
all_p_x = []
all_p_y = []

all_v_x = []
all_v_y = []

all_energies = []

all_angles = []

time_array = np.arange(-t_0, t_0, delta_t)

random_values = np.random.random((len(time_array), n))

ionization_array = np.zeros((len(time_array), z_max))  # массив с данными о потенциалах ионизации электронов

fully_ionized_states_list = []

for i, current_moment in enumerate(time_array):
    # вероятность ионизации каждого атома в момент времени current_moment:
    probabilities = np.array([w_ppt(current_moment, atom) * delta_t for atom in placed_cells])
    mask = random_values[i] < probabilities
    true_indices_in_mask = np.where(mask)[0]  # массив индексов атомов/ионов, которые ПО ПОЛОЖЕНИЮ В ПРОСТРАНСТВЕ
    # подходят для ионизации. Возможно, часть индексов соответствует полностью ионизованным состояниям, которые
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
            print('full', j, len(fully_ionized_states_list))
            continue
        ionization_array[i, int(placed_cells[j, 3]) - 1] += 1  # инкрементируем число электронов в данный момент
        # времени для данного потенциала ионизации
        placed_cells[j, 3:] = ionization_order_array[int(placed_cells[j, 3])]  # переводим атом/ион в следующее
        # состояние

        vel_x, vel_y = solve_motion(current_moment, placed_cells[j, :3])  # расчет скоростей вылетающего электрона
        all_v_x.append(vel_x)
        all_v_y.append(vel_y)

        p_x = vel_x / np.sqrt(1 - vel_x ** 2 - vel_y ** 2)
        p_y = vel_y / np.sqrt(1 - vel_x ** 2 - vel_y ** 2)
        all_p_x.append(p_x)
        all_p_y.append(p_y)

        all_energies.append(np.sqrt(1 + p_x ** 2 + p_y ** 2))
        all_angles.append(angle_calculation(p_x, p_y))


number_of_electrons = len(all_p_x)
number_of_fully_ionized_states = len(fully_ionized_states_list)

print('number of electrons =', number_of_electrons)
print('number of full ionized states =', number_of_fully_ionized_states)

electrons_visualisation()

momenta_plotter()

electrons_angle_distribution()

ions_visualization()
