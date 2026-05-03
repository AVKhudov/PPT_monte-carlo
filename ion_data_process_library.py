import matplotlib.pyplot as plt
import numpy as np
import ion_config as cfg
import os

from datetime import datetime
from matplotlib.ticker import MaxNLocator
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm, ListedColormap


def ions_momenta_distribution(ions_data_array, indices_array, axis=0, preferred_charge=-1, number_of_bins=100,
                              electron_scale=True, save=False, title='ions_momenta.jpg', path=r"C:\Users\Dns\Desktop"):
    scale_factor = 8e4  # считаем импульс в m_electron * c
    x_label_part = 'electron'
    if electron_scale is False:
        scale_factor = 1.  # считаем импульс в m_ion * c
        x_label_part = 'ion'

    local_axis = 3 + axis
    if preferred_charge == -1:
        ions_momenta = ions_data_array[:, local_axis] * scale_factor
    elif preferred_charge == 18:
        ions_momenta = ions_data_array[indices_array, local_axis] * scale_factor
    else:
        mask = (ions_data_array[:, 6] == preferred_charge + 1)
        ions_momenta = ions_data_array[mask, local_axis] * scale_factor

    momenta_bins = np.linspace(np.min(ions_momenta), np.max(ions_momenta), num=number_of_bins + 1)
    counts, bins = np.histogram(ions_momenta, momenta_bins)

    bin_width = bins[1] - bins[0]

    counts = counts / np.size(ions_momenta) / bin_width
    bin_edges = bins[:-1]

    plt.figure(figsize=(10, 6))
    plt.plot(bin_edges, counts)

    if preferred_charge == -1:
        plt.xlabel(f'Импульс всех ионов вдоль оси {axis}, m_{x_label_part} * c')
    else:
        plt.xlabel(f'Импульс ионов заряда {preferred_charge} вдоль оси {axis}, m_{x_label_part} * c')

    plt.ylabel('dN/Ndp')

    if save:
        file_path = os.path.join(path, title)
        plt.savefig(file_path)
    else:
        plt.show()


def electrons_energy_distribution(energies, number_of_bins=100, save=False, title='energy_distribution.pdf',
                                  mask=False):
    electron_mass = 0.51099895069
    mev_energies = energies * electron_mass

    if mask:
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


def ions_visualization(atoms):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = atoms[:, 0]
    y = atoms[:, 1]
    z = atoms[:, 2]

    color_data = atoms[:, 3]

    scatter = ax.scatter(x, y, z, c=color_data, s=50, alpha=0.8)
    color_bar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    color_bar.set_label('Кратность иона')

    plt.tight_layout()
    plt.show()


def bar_plotter(atoms, save=False, title='ions_bar.pdf'):

    charge_numbers = atoms[:, 3].astype(int)
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


def create_timestamp_folder(base_path=r'C:\Users\Dns\Desktop\MC_Ion'):
    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    folder_path = os.path.join(base_path, timestamp)  # создаем путь к папке
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_file(folder_path, filename, data):
    file_path = os.path.join(folder_path, filename)  # создаем путь к файлу
    np.save(file_path, data)
