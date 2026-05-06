import matplotlib.pyplot as plt
import numpy as np
import ion_config as cfg

from matplotlib.ticker import MaxNLocator
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm, ListedColormap


def electrons_energy_distribution(energies, number_of_bins=100, save=False, title='energy_distribution.pdf',
                                  mask=False):
    electron_mass = 0.51099895069  # МэВ
    mev_energies = energies * electron_mass

    if mask:  # обрезаем возможные артефакты
        mev_energies_scratch = np.round(energies * electron_mass, 11)
        peak_mask = mev_energies_scratch > 0.54
        mev_energies = mev_energies_scratch[peak_mask]

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


def electrons_visualisation(t_array, ion_array, row_gap=1000, save=False, title='electron_visualisation.pdf'):

    # Создаем массив для столбцов
    n_segments = len(t_array) // row_gap
    bar_array = np.zeros((n_segments, cfg.z_max))
    x_array = t_array[::row_gap]

    # Вычисляем сумму ионизаций для каждого промежутка
    for i in range(0, len(t_array), row_gap):
        bar_array[i // row_gap] = np.sum(ion_array[i:i + row_gap], axis=0)

    # Рассчитываем пределы
    max_electrons = np.max(np.sum(bar_array, axis=1))
    y_upper_limit = max(max_electrons, np.exp(-(t_array / cfg.tau) ** 2).max() * max_electrons) * 1.1

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

    for k in range(cfg.z_max):
        ax.bar(x_array, bar_array[:, k], bottom=bottom, color=colors[k], width=width)
        bottom += bar_array[:, k]

    # Рисуем огибающую
    envelope = np.exp(-(t_array / cfg.tau) ** 2) * max_electrons
    ax.plot(t_array, envelope, 'k-', linewidth=2, label='Огибающая')

    bounds = np.arange(cfg.z_max + 1)
    norm = BoundaryNorm(bounds, len(colors))
    cbar = ColorbarBase(plt.axes([0.92, 0.15, 0.02, 0.7]),
                        cmap=ListedColormap(colors),
                        norm=norm,
                        ticks=np.arange(cfg.z_max) + 0.5)
    cbar.set_ticklabels(np.arange(1, cfg.z_max + 1))
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


def electron_gamma_factor_vs_time(sort_input, select_input, time_input, px_input, py_input, pz_input,
                                  save=False, title=f'гамма факторы электронов.jpg'):
    for every_sort in sort_input:
        sort_mask = select_input[:, 1].astype(int) == every_sort
        sort_indices = np.where(sort_mask)[0]

        random_idx = np.random.choice(sort_indices, size=1)[0]
        energy = np.sqrt(
            1
            + px_input[random_idx] ** 2
            + py_input[random_idx] ** 2
            + pz_input[random_idx] ** 2
        )
        time = time_input[random_idx]

        plt.plot(time, energy, label=f'{every_sort}')
    plt.title('Гамма-факторы для электронов разных п-лов ионизации')
    plt.legend()

    if save:
        plt.savefig(title)
    else:
        plt.show()


def angle_calculation(y_comp, x_comp):
    if (y_comp > 0 and x_comp > 0) or (y_comp < 0 < x_comp):
        return np.arctan(y_comp / x_comp)
    if y_comp > 0 > x_comp:
        return np.pi + np.arctan(y_comp / x_comp)
    if y_comp < 0 and x_comp < 0:
        return np.arctan(y_comp / x_comp) - np.pi


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
