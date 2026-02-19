import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm, ListedColormap
import os

path = r"C:\Users\Dns\Desktop\MC_Ion\19-02-2026_18-22-41"
os.chdir(path)

[n, tau, t_0, delta_t, step, z_max] = np.load('params.npy')

n = int(n)  # отсекаем дробные части целочисленных параметров
z_max = int(z_max)

placed_cells = np.load('placed_cells.npy')


def bar_plotter(save=False, title='ions_bar.pdf'):

    charge_numbers = placed_cells[:, 3].astype(int)
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


def electrons_visualisation(row_gap=100, save=False, title='electron_visualisation.pdf'):

    time_array = np.load('time_array.npy')
    ionization_array = np.load('ionization_array.npy')

    # Создаем массив для столбцов
    n_segments = len(time_array) // row_gap
    bar_array = np.zeros((n_segments, z_max))
    x_array = time_array[::row_gap]

    # Вычисляем сумму ионизаций для каждого промежутка
    for i in range(0, len(time_array), row_gap):
        bar_array[i // row_gap] = np.sum(ionization_array[i:i + row_gap], axis=0)

    # Рассчитываем пределы
    max_electrons = np.max(np.sum(bar_array, axis=1))
    y_upper_limit = max(max_electrons, np.exp(-(time_array / tau) ** 2).max() * max_electrons) * 1.1

    # Цвета
    colors = ['#440154', '#481567', '#482677', '#453781', '#404788',
              '#39558C', '#33638D', '#2D708E', '#287D8E', '#238A8D',
              '#1F968B', '#20A387', '#29AF7F', '#3CBB75', '#55C677',
              '#73D055', '#95D840', '#B8DE29']

    # Создаем график
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set(xlim=(time_array[0], time_array[-1]), ylim=(0, y_upper_limit))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Рисуем столбцы
    width = time_array[row_gap] - time_array[0]
    bottom = np.zeros_like(x_array)

    for k in range(z_max):
        ax.bar(x_array, bar_array[:, k], bottom=bottom, color=colors[k], width=width)
        bottom += bar_array[:, k]

    # Рисуем огибающую
    envelope = np.exp(-(time_array / tau) ** 2) * max_electrons
    ax.plot(time_array, envelope, 'k-', linewidth=2, label='Огибающая')

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


def momenta_plotter(save=False, title='импульсный спектр, интенсивность 1,5.pdf'):

    p_x = np.load('all_electron_p_x.npy')
    p_y = np.load('all_electron_p_y.npy')

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


def electrons_angle_distribution(save=False, title='angle_distribution.pdf'):

    angles = np.load('all_electron_angles.npy')

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


def electrons_energy_distribution(save=False, title='energy_distribution.pdf'):

    energies = np.load('all_electron_energies.npy')
    mev_energies = np.array([energy * 0.5 for energy in energies])

    # Прямое построение гистограммы с логарифмическими осями
    plt.figure(figsize=(10, 6))

    log_min = np.log10(np.min(mev_energies))
    log_max = np.log10(np.max(mev_energies))

    # Используем логарифмические бины
    log_bins = np.logspace(log_min,
                           log_max, 20)

    counts, bins = np.histogram(mev_energies, log_bins)
    counts = counts / np.size(mev_energies)

    print(np.size(mev_energies))

    plt.loglog(bins[:-1], counts)
    plt.xlabel('Энергия электронов, МэВ')
    plt.ylabel('dN/NdE')

    if save:
        plt.savefig(title)
    else:
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


ions_visualization()
electrons_visualisation()
momenta_plotter()
electrons_energy_distribution()
