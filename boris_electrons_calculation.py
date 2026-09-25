import os
import numpy as np
import time as tm
import ion_data_process_library as ion_proc
import pickle

from datetime import datetime
from c_electrons_interface import run_boris


DATA_PATH = r"D:\MC_Ion\17-09-2026_17-01-59"

PICKLE_PATH = r"D:\MC_Ion\17-09-2026_17-01-59\all_params_dict.pkl"
with open(PICKLE_PATH, 'rb') as f:
    params_dict = pickle.load(f)

INPUT_FILE = os.path.join(
    DATA_PATH,
    "electron_motion_input.npy"
)

timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
BORIS_PATH = os.path.join(
    DATA_PATH,
    f"boris_electrons_C_motion_0_0025"
)

os.makedirs(BORIS_PATH, exist_ok=True)

electron_motion_input = np.load(INPUT_FILE)
print(f"Электронов: {len(electron_motion_input)}")

time_step = 0.0025
t_electron = params_dict['t_electron']
w_0 = params_dict['w_0']
tau = params_dict['tau']
a_0 = params_dict['a_0']
eps = params_dict['eps']
right_or_left = params_dict['right_or_left']

print("Boris-расчет электронов...")
start_boris_time = tm.perf_counter()

sorts_array, initial_t_array, p_x_arr, p_y_arr, p_z_arr = run_boris(
    electron_motion_input,
    t_electron,
    time_step,
    w_0,
    tau,
    a_0,
    eps,
    right_or_left,
    np.pi * w_0 * w_0
)

end_boris_time = tm.perf_counter()
print(f"Boris-расчет завершен, время: {end_boris_time - start_boris_time} с")
print(f"output path: {BORIS_PATH}")

energies_arr = np.sqrt(1 + p_x_arr ** 2 + p_y_arr ** 2 + p_z_arr ** 2)

ion_proc.save_file(BORIS_PATH, 'all_electron_sorts.npy', sorts_array)
ion_proc.save_file(BORIS_PATH, 'all_initial_times.npy', initial_t_array)
ion_proc.save_file(BORIS_PATH, 'all_electron_p_x.npy', p_x_arr)
ion_proc.save_file(BORIS_PATH, 'all_electron_p_y.npy', p_y_arr)
ion_proc.save_file(BORIS_PATH, 'all_electron_p_z.npy', p_z_arr)
ion_proc.save_file(BORIS_PATH, 'all_electron_energies.npy', energies_arr)
