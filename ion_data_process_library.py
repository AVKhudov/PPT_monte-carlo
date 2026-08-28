import matplotlib.pyplot as plt
import numpy as np

import os
from datetime import datetime


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


def final_ion_charge_bar(placed_cells, save=False, title='final_ion_charge_bar.png'):
    final_charges = placed_cells[:, 6].astype(int)
    unique_charges, counts = np.unique(final_charges, return_counts=True)

    plt.figure(figsize=(10, 6))
    plt.bar(unique_charges, counts)

    plt.xlabel('Сорт иона')
    plt.ylabel('Количество')
    plt.xticks(unique_charges)

    if save:
        plt.savefig(title)
    else:
        plt.show()


def create_timestamp_folder(base_path=r'D:\MC_Ion'):
    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    folder_path = os.path.join(base_path, timestamp)  # создаем путь к папке
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_file(folder_path, filename, data):
    file_path = os.path.join(folder_path, filename)  # создаем путь к файлу
    np.save(file_path, data)
