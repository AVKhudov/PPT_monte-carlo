import numpy as np
import os
import electron_library as el_lib
import ion_data_process_library as ion_proc
import argon_05_09 as argon

from datetime import datetime

DATA_PATH = r"D:\MC_Ion\27-09-2026_23-50-54_C"

if __name__ == '__main__':

    electron_motion_input = np.load(os.path.join(DATA_PATH, 'electron_motion_input.npy'))

    sorts, initial_times, p_x_array, p_y_array, p_z_array = \
        el_lib.electron_parallel_simulation(electron_motion_input)

    energies_array = np.sqrt(1 + p_x_array * p_x_array + p_y_array * p_y_array + p_z_array * p_z_array)

    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    electron_path = os.path.join(DATA_PATH, rf'electron_motion_{timestamp}')
    os.makedirs(electron_path)

    ion_proc.save_file(electron_path, 'all_electron_sorts.npy', sorts)
    ion_proc.save_file(electron_path, 'all_initial_times.npy', initial_times)
    ion_proc.save_file(electron_path, 'all_electron_p_x.npy', p_x_array)
    ion_proc.save_file(electron_path, 'all_electron_p_y.npy', p_y_array)
    ion_proc.save_file(electron_path, 'all_electron_p_z.npy', p_z_array)
    ion_proc.save_file(electron_path, 'all_electron_energies.npy', energies_array)

    argon.save_simulation_parameters(electron_path)

    print('\n')
    print('output path:', electron_path)

