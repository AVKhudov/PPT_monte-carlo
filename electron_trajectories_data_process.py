import numpy as np
import os
import pickle
import electron_library as el_lib
import argon_05_09 as argon

from datetime import datetime

DATA_PATH = r"D:\MC_Ion\18-08-2026_03-27-08_2e22\electron_motion_19-09-2026_19-55-02"

with open(os.path.join(DATA_PATH, 'all_params_dict.pkl'), 'rb') as f:
    params_dict = pickle.load(f)

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

form = params_dict['form']
box = form / 4.

x_limit_box = box[0] / 2
y_limit_box = box[1] / 2
z_limit_box = box[2] / 2

coordinate_condition = (
    (-x_limit_box <= electron_motion_input[:, 2]) & (electron_motion_input[:, 2] <= x_limit_box) &
    (-y_limit_box <= electron_motion_input[:, 3]) & (electron_motion_input[:, 3] <= y_limit_box) &
    (-z_limit_box <= electron_motion_input[:, 4]) & (electron_motion_input[:, 4] <= z_limit_box)
)

stride = 10
coordinates_inside_box = np.unique(electron_motion_input[coordinate_condition, 2:], axis=0)[::stride]

atom_coordinates = coordinates_inside_box[15]

coordinates_mask = np.all((electron_motion_input[:, 2:] == atom_coordinates), axis=1)
coordinates_idx = np.where(coordinates_mask)[0]
initial_conditions_array = electron_motion_input[coordinates_idx]

base_path = r"D:\MC_Ion\one_atom_trajectories"

timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
trajectory_path = os.path.join(base_path, f'trajectory_{timestamp}')
os.makedirs(trajectory_path)

argon.save_simulation_parameters(trajectory_path)
with open(os.path.join(trajectory_path, 'atom_coordinates.txt'), 'a') as f:
    f.write(f'{atom_coordinates}')

for i, initial_condition in enumerate(initial_conditions_array):
    electron_sort, t_arr, x_arr, y_arr, z_arr, p_x_arr, p_y_arr, p_z_arr = el_lib.single_electron_trajectory_process(
        initial_condition
    )

    np.save(os.path.join(trajectory_path, f'x_{9 + i}_array.npy'), x_arr)
    np.save(os.path.join(trajectory_path, f'y_{9 + i}_array.npy'), y_arr)
    np.save(os.path.join(trajectory_path, 'atom_coordinates.npy'), atom_coordinates)
