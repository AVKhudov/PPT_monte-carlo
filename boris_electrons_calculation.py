import os
import numpy as np
import ion_config as cfg
import time as tm
import ion_data_process_library as ion_proc

from datetime import datetime
from boris_electrons_interface import run_boris


DATA_PATH = r"D:\MC_Ion\11-09-2026_22-36-51"

INPUT_FILE = os.path.join(
    DATA_PATH,
    "electron_motion_input.npy"
)

timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
BORIS_PATH = os.path.join(
    DATA_PATH,
    f"boris_electrons_{timestamp}"
)

os.makedirs(BORIS_PATH, exist_ok=True)

electron_motion_input = np.load(INPUT_FILE)
print(f"Электронов: {len(electron_motion_input)}")

print("Boris-расчет электронов...")
start_boris_time = tm.perf_counter()

sorts_array, initial_t_array, p_x_arr, p_y_arr, p_z_arr = run_boris(
    electron_motion_input,
    cfg.t_electron,
    cfg.delta_t,
    cfg.w_0,
    cfg.tau,
    cfg.a_0,
    cfg.eps,
    cfg.right_or_left,
    np.pi * cfg.w_0 * cfg.w_0
)

end_boris_time = tm.perf_counter()
print(f"Boris-расчет завершен, время: {end_boris_time - start_boris_time} сек")

energies_arr = np.sqrt(1 + p_x_arr ** 2 + p_y_arr ** 2 + p_z_arr ** 2)

ion_proc.save_file(BORIS_PATH, 'all_electron_sorts.npy', sorts_array)
ion_proc.save_file(BORIS_PATH, 'all_initial_times.npy', initial_t_array)
ion_proc.save_file(BORIS_PATH, 'all_electron_p_x.npy', p_x_arr)
ion_proc.save_file(BORIS_PATH, 'all_electron_p_y.npy', p_y_arr)
ion_proc.save_file(BORIS_PATH, 'all_electron_p_z.npy', p_z_arr)
ion_proc.save_file(BORIS_PATH, 'all_electron_energies.npy', energies_arr)
