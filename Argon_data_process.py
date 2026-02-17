import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import MaxNLocator
import json


with open(r'C:\Users\Dns\Desktop\MC_Ion\charge_numbers.json', 'r') as file:
    charge_numbers = json.load(file)

with open(r'C:\Users\Dns\Desktop\MC_Ion\all_electron_p_x.json', 'r') as file:
    all_p_x = json.load(file)

with open(r'C:\Users\Dns\Desktop\MC_Ion\all_electron_p_y.json', 'r') as file:
    all_p_y = json.load(file)

with open(r'C:\Users\Dns\Desktop\MC_Ion\all_electron_p_z.json', 'r') as file:
    all_p_z = json.load(file)

with open(r'C:\Users\Dns\Desktop\MC_Ion\all_electron_energies.json', 'w') as file:
    all_energies = json.load(file)

with open(r'C:\Users\Dns\Desktop\MC_Ion\all_electron_angles.json', 'w') as file:
    all_angles = json.load(file)


def bar_plotter(save=False, title='ions_bar.pdf'):
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

    if save:
        plt.savefig(title)
    else:
        plt.show()


def momenta_plotter(save=False, title='импульсный спектр, интенсивность 1,5.pdf'):
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
        plt.savefig(title)
    else:
        plt.show()


def electrons_angle_distribution(save=False, title='angle_distribution.pdf'):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')

    ax.hist(np.array(all_angles), bins=36, alpha=0.7, color='red')

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.set_title('Углы вылета электронов', pad=20)

    if save:
        plt.savefig(title)
    else:
        plt.show()


def electrons_energy_distribution(save=False, title='energy_distribution.pdf'):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    all_energies_array = np.array(all_energies)

    energy_cut = 1.
    energy_mask = (all_energies_array > energy_cut)

    valid_energies = all_energies_array[energy_mask]

    counts, bin_edges = np.histogram(valid_energies, density=False, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax1.plot(bin_centers, counts)
    ax2.loglog(bin_centers, counts)

    print(counts, bin_edges)

    if save:
        plt.savefig(title)
    else:
        plt.show()


momenta_plotter()
