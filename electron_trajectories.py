import os
import numpy as np
import electron_library as el_lib
import ion_data_process_library as ion_proc

# Файл, в котором расчет траекторий электронов вынесен отдельно
# Работает с массивом electron_motion_input в качестве входных данных

# electron_motion_input = [[t, sort, x, y, z], [t, sort, x, y, z], ...]

DATA_PATH = r"D:\MC_Ion\21-08-2026_20-02-48_1e23"

if __name__ == '__main__':

    gamma_threshold = 200.
    sort = 18

    electron_motion_input = np.load(os.path.join(DATA_PATH, 'electron_motion_input.npy'))
    electron_energies = np.load(os.path.join(DATA_PATH, 'all_electron_energies.npy'))
    electron_sorts = np.load(os.path.join(DATA_PATH, 'all_electron_sorts.npy'))
    electron_p_x = np.load(os.path.join(DATA_PATH, 'all_electron_p_x.npy'))

    mask = True

    if mask:  # отсекаем электроны, летящие назад
        wrong_idx = np.where(electron_p_x < 0.)[0]
        print('electron_trajectories warning:')
        print(f'- p_x < 0. electrons indices: {wrong_idx}')
        print(f'- number of electrons with p_x < 0.: {len(wrong_idx)} of {len(electron_p_x)}')

        if np.size(wrong_idx) != 0:
            electron_motion_input = np.delete(electron_motion_input, wrong_idx, axis=0)
            electron_energies = np.delete(electron_energies, wrong_idx)
            electron_sorts = np.delete(electron_sorts, wrong_idx)
            electron_p_x = np.delete(electron_p_x, wrong_idx)

    above_threshold_idx = np.where(electron_energies > gamma_threshold)[0]
    sort_idx = np.where(electron_sorts == sort)[0]
    trajectory_idx = np.intersect1d(sort_idx, above_threshold_idx)

    stride = 10
    trajectory_idx_stride = trajectory_idx[::stride]

    trajectory_input = electron_motion_input[trajectory_idx_stride]

    electron_trajectories = el_lib.electron_trajectories_simulation(trajectory_input)
    trajectory_path = os.path.join(DATA_PATH,
                                   rf'trajectories_stride_{stride}\sort_{sort}_gamma_threshold_{gamma_threshold}')
    os.makedirs(trajectory_path)

    ion_proc.save_file(trajectory_path, 'trajectory_t', electron_trajectories["t"])
    ion_proc.save_file(trajectory_path, 'trajectory_x', electron_trajectories["x"])
    ion_proc.save_file(trajectory_path, 'trajectory_y', electron_trajectories["y"])
    ion_proc.save_file(trajectory_path, 'trajectory_z', electron_trajectories["z"])
    ion_proc.save_file(trajectory_path, 'trajectory_p_x', electron_trajectories["p_x"])
    ion_proc.save_file(trajectory_path, 'trajectory_p_y', electron_trajectories["p_y"])
    ion_proc.save_file(trajectory_path, 'trajectory_p_z', electron_trajectories["p_z"])
