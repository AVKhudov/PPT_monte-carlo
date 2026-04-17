import ion_data_process_library as ion_proc
import os
import numpy as np

path = r"C:\Users\Dns\Desktop\MC_Ion\17-04-2026_16-31-21"
os.chdir(path)

n, tau, t_0, delta_t, z_max = np.load('params.npy')
n = int(n)  # отсекаем дробные части целочисленных параметров
z_max = int(z_max)

energies = np.load('all_electron_energies.npy')
ion_proc.electrons_energy_distribution(energies, number_of_bins=40)


time_array = np.load('time_array.npy')
ionization_array = np.load('ionization_array.npy')

ion_proc.electrons_visualisation(time_array, ionization_array)


placed_cells = np.load('placed_cells.npy')
indices = np.load('fully_ionized_indices.npy')

ion_proc.ions_momenta_distribution(placed_cells, indices, axis=1,
                                   preferred_charge=16, number_of_bins=60)
