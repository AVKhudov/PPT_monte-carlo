import ion_data_process_library as ion_data
import os
import numpy as np

path = r"C:\Users\Dns\Desktop\MC_Ion\10-04-2026_03-34-40"
os.chdir(path)

n, tau, t_0, delta_t, z_max = np.load('params.npy')

ions_momenta = np.load('ion_momenta_array.npy')

ions_momenta_x = ions_momenta[:, 0]
ions_momenta_y = ions_momenta[:, 1]
ions_momenta_z = ions_momenta[:, 2]

ion_data.ions_momenta_distribution(ions_momenta_z, number_of_bins=70)

n = int(n)  # отсекаем дробные части целочисленных параметров
z_max = int(z_max)
