import ion_library as ion
import os
import numpy as np

path = r"C:\Users\Dns\Desktop\MC_Ion\07-04-2026_01-04-16"
os.chdir(path)

n, tau, t_0, delta_t, z_max = np.load('params.npy')

n = int(n)  # отсекаем дробные части целочисленных параметров
z_max = int(z_max)

time_array = np.load('time_array.npy')
ionization_array = np.load('ionization_array.npy')
ion.electrons_visualisation(time_array, ionization_array)

energies = np.load('all_electron_energies.npy')
#ion.electrons_energy_distribution(energies)
