from scipy.special import gamma
import numpy as np


# Функции, считающая коэффициенты C и B в w_PPT
def c_n_l_squared(elem):
    return 2 ** (2 * elem[0] - 2) / (elem[0] * gamma(elem[0] + elem[1] + 1) * gamma(elem[0] - elem[1]))


def b_l_m(elem):
    return (2 * elem[1] + 1) * gamma(elem[1] + abs(elem[2]) + 1) / (2 ** abs(elem[2]) * gamma(abs(elem[2]) + 1)
                                                                  * gamma(elem[1] - abs(elem[2]) + 1))


def phi(x_coordinate, wave):
    return np.arctan(wave * abs(x_coordinate) / np.pi)


def rho(x_coordinate, wave):
    return x_coordinate * (1 + (np.pi / (wave * abs(x_coordinate))) ** 2)


def radius(x_coordinate, wave):
    return np.sqrt(1 + (wave * abs(x_coordinate) / np.pi) ** 2)


def pulse_field(time, rad_vec, wave, half_width, eps=0., right_or_left=+1):
    ellipticity = [1 / np.sqrt(1 + eps ** 2), eps / np.sqrt(1 + eps ** 2)]

    phase = 2 * np.pi * rad_vec[0] / wave - time - phi(rad_vec[0], wave) + np.pi / wave \
            * (rad_vec[1] ** 2 + rad_vec[2] ** 2) / rho(rad_vec[0], wave)

    spatial_factor = np.exp(-4 * (rad_vec[1] ** 2 + rad_vec[0] ** 2) / radius(rad_vec[0], wave) ** 2) / radius(rad_vec[0], wave)

    envelope = np.exp(-(time - 2 * np.pi * rad_vec[0] / wave) ** 2 / half_width ** 2)

    cos_comp = spatial_factor * envelope * np.cos(phase)

    sin_comp = spatial_factor * envelope * np.sin(phase)

    e_field = [0,
               cos_comp * ellipticity[0],
               sin_comp * ellipticity[1] * right_or_left]

    h_field = [0,
               -sin_comp * ellipticity[1] * right_or_left,
               cos_comp * ellipticity[0]]

    return [e_field, h_field]
