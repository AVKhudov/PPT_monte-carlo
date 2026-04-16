import ion_data_process_library as ion_data
import os
import numpy as np

path = r"C:\Users\Dns\Desktop\MC_Ion\13-04-2026_18-03-46"
os.chdir(path)

n, tau, t_0, delta_t, z_max = np.load('params.npy')
n = int(n)  # отсекаем дробные части целочисленных параметров
z_max = int(z_max)

ions_momenta = np.load('ion_momenta_array.npy')

ions_momenta_x = ions_momenta[:, 0]
ions_momenta_y = ions_momenta[:, 1]
ions_momenta_z = ions_momenta[:, 2]

ion_data.ions_momenta_distribution(ions_momenta_x, number_of_bins=40)
