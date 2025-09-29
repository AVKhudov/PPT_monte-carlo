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
intensity = 1e20 * 1.5  # Интенсивность поля в фокусе в единицах Вт/см^2
focus_radius = 2  # Радиус фокусировки в мкм. Длина волны излучени фиксирована и равна 1 мкм
tau = 30  # Время включения поля
t_0 = 100  # Длительность моделирования
cut_time = 90  # Время отсечки, после которого электрон считается свободным
delta_t = 0.1  # Разрешение по времени
step = 0.01  # Разрешение в пространстве

wavelength = 1/focus_radius  # Длина волны в единицах радиуса перетяжки


def phi(z_coordinate):
    return np.arctan(wavelength * abs(z_coordinate) / np.pi)


def rho(z_coordinate):
    return z_coordinate * (1 + (np.pi / (wavelength * abs(z_coordinate))) ** 2)


def radius(z_coordinate):
    return np.sqrt(1 + (wavelength * abs(z_coordinate) / np.pi) ** 2)


amplitude = (intensity / 10 ** 16) ** (1/2) * 0.53


def pulse_field(time, rad_vec):
    return amplitude * np.cos(2 * np.pi * rad_vec[2] / wavelength - time - phi(rad_vec[2])
                  + np.pi / wavelength * (rad_vec[0] ** 2 + rad_vec[1] ** 2) / rho(rad_vec[2]) + np.pi/2) * \
           np.exp(-4 * (rad_vec[0] ** 2 + rad_vec[2] ** 2) / radius(rad_vec[2]) ** 2) * \
           np.exp(-(time - 2 * np.pi * rad_vec[2] / wavelength) ** 2 / tau ** 2) / radius(rad_vec[2])


def motion_equation(time, variables_vector, radius_vector):
    alpha = 1 / 137
    v_x, v_z = variables_vector
    reverse_gamma_factor = np.sqrt(1 - v_x ** 2 - v_z ** 2)
    field = pulse_field(time, radius_vector)

    v_x_eq = alpha * reverse_gamma_factor * (field - v_z * field - v_x ** 2 * field)
    v_z_eq = alpha * reverse_gamma_factor * (v_x * field - v_x * v_z * field)
    return np.array([v_x_eq, v_z_eq])


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


# Основной цикл
position = np.random.randint(-100, 101, 3) * step
time_array = np.arange(-t_0, t_0, delta_t)

p_x_list = []
p_z_list = []

for current_moment in time_array:

    vel_x, vel_z = solve_motion(current_moment, position)

    p_x = 137 * vel_x / np.sqrt(1 - vel_x ** 2 - vel_z ** 2)
    p_z = 137 * vel_z / np.sqrt(1 - vel_x ** 2 - vel_z ** 2)

    p_x_list.append(p_x)
    p_z_list.append(p_z)

p_x_array = np.array(p_x_list)
p_z_array = np.array(p_z_list)

print(position)

plt.plot(time_array, p_x_list, label='p_x')
plt.plot(time_array, p_z_list, label='p_z')

plt.legend()

plt.show()
