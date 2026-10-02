import numpy as np
import os
import electron_library as el_lib
import argon_05_09 as argon
import ion_config as cfg

from datetime import datetime

DATA_PATH = r"D:\MC_Ion\18-08-2026_03-27-08_2e22\electron_motion_19-09-2026_19-55-02"

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

box = cfg.form / 4.

x_limit_box = box[0] / 2
y_limit_box = box[1] / 2
z_limit_box = box[2] / 2

coordinate_condition = (
    (-x_limit_box <= electron_motion_input[:, 2]) & (electron_motion_input[:, 2] <= x_limit_box) &
    (-y_limit_box <= electron_motion_input[:, 3]) & (electron_motion_input[:, 3] <= y_limit_box) &
    (-z_limit_box <= electron_motion_input[:, 4]) & (electron_motion_input[:, 4] <= z_limit_box)
)

coordinates_inside_box = np.unique(electron_motion_input[coordinate_condition, 2:], axis=0)

count_trajectories = True

if not count_trajectories:
    with open(os.path.join(DATA_PATH, 'coordinates_inside_box.txt'), 'a') as f:
        for i, row in enumerate(coordinates_inside_box):
            f.write(f'{i}: {row}\n')

if count_trajectories:
    atom_coordinates = coordinates_inside_box[464]
    coordinates_mask = np.all((electron_motion_input[:, 2:] == atom_coordinates), axis=1)
    coordinates_idx = np.where(coordinates_mask)[0]
    initial_conditions_array = electron_motion_input[coordinates_idx]

    base_path = r"D:\MC_Ion\one_atom_trajectories"

    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    trajectory_path = os.path.join(base_path, f'trajectory_Python_{timestamp}')
    os.makedirs(trajectory_path)

    argon.save_simulation_parameters(trajectory_path)

    with open(os.path.join(trajectory_path, 'atom_coordinates.txt'), 'a') as f:
        f.write(f'{atom_coordinates}')

    for i, initial_condition in enumerate(initial_conditions_array):
        electron_sort, t_arr, x_arr, y_arr, z_arr, p_x_arr, p_y_arr, p_z_arr = el_lib.single_electron_trajectory_process(
            initial_condition
        )

        start_sort = cfg.start_charge_number_of_particles + 1

        np.save(os.path.join(trajectory_path, f'x_{start_sort + i}_array.npy'), x_arr)
        np.save(os.path.join(trajectory_path, f'y_{start_sort + i}_array.npy'), y_arr)
        np.save(os.path.join(trajectory_path, 'atom_coordinates.npy'), atom_coordinates)
