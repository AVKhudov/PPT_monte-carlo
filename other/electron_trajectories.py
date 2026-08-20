import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def lorenz_equation(time, variables_vector):
    x, y, z, momenta_x, momenta_y, momenta_z = variables_vector

    electric = np.array([0, 1, 0])
    magnetic = np.array([0, 0, 1])

    energy = np.sqrt(1 + momenta_x ** 2 + momenta_y ** 2 + momenta_z ** 2)

    cross_product = [momenta_y * magnetic[2] - momenta_z * magnetic[1],
                     momenta_z * magnetic[0] - momenta_x * magnetic[2],
                     momenta_x * magnetic[1] - momenta_y * magnetic[0]]

    x_eq = momenta_x / energy
    y_eq = momenta_y / energy
    z_eq = momenta_z / energy

    p_x_eq = electric[0] + cross_product[0] / energy
    p_y_eq = electric[1] + cross_product[1] / energy
    p_z_eq = electric[2] + cross_product[2] / energy

    return np.array([x_eq, y_eq, z_eq, p_x_eq, p_y_eq, p_z_eq])


def solve_lorenz_motion(t_start, t_stop, initial_r, initial_m):
    t_span = (t_start, t_stop)
    t_eval = np.linspace(t_start, t_stop, num=100)

    sol = solve_ivp(
        lorenz_equation,
        t_span,
        np.hstack((initial_r, initial_m)),
        t_eval=t_eval,
        vectorized=True
    )
    return sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4], sol.y[5], sol.t


def analytical_solution(r_start, momenta, p):
    momenta_squared = momenta[0] ** 2 + momenta[1] ** 2 + momenta[2] ** 2
    alpha = np.sqrt(1 + momenta_squared) - momenta[0]
    eps = np.sqrt(1 + momenta[2] ** 2)

    x_0, y_0, z_0 = r_start

    x_coord = 0.5 * p * (eps ** 2 / alpha ** 2 - 1) + p ** 3 / (6 * alpha ** 2) + x_0
    y_coord = p ** 2 / (2 * alpha) + y_0
    z_coord = p * momenta[2] / alpha + z_0
    time = 0.5 * p * (eps ** 2 / alpha ** 2 + 1) + p ** 3 / (6 * alpha ** 2)

    return np.array([x_coord, y_coord, z_coord, time, alpha])


initial_rad_vector = np.array([0., 0., 0.])
initial_momenta_vector = np.array([1., 0., 3.])

initial_t = 0.
final_t = 10000.

x_array, y_array, z_array, p_x_array, params_array, p_z_array, time_array = solve_lorenz_motion(initial_t, final_t,
                                                                                                initial_rad_vector,
                                                                                                initial_momenta_vector)

alpha_array = np.sqrt(1 + p_x_array ** 2 + params_array ** 2 + p_z_array ** 2) - p_x_array


analytical_x_array = np.array([analytical_solution(initial_rad_vector,
                                                   initial_momenta_vector, param)[0] for param in params_array])
analytical_y_array = np.array([analytical_solution(initial_rad_vector,
                                                   initial_momenta_vector, param)[1] for param in params_array])
analytical_z_array = np.array([analytical_solution(initial_rad_vector,
                                                   initial_momenta_vector, param)[2] for param in params_array])
analytical_time_array = np.array([analytical_solution(initial_rad_vector,
                                                      initial_momenta_vector, param)[3] for param in params_array])
analytical_alpha_array = np.array([analytical_solution(initial_rad_vector,
                                                       initial_momenta_vector, param)[4] for param in params_array])

plt.figure(figsize=(10, 6))
plt.plot(params_array, analytical_time_array, label='analytical')
plt.plot(params_array, time_array, 'o', label='numerical')
plt.legend()
plt.show()
