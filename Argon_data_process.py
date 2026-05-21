import os
import numpy as np
import ion_config as cfg

import ion_data_process_library as ion_proc
import electron_data_process_library as el_proc


DATA_PATH = r"C:\Users\Dns\Desktop\MC_Ion\15-05-2026_22-54-09_5e21"


def main():
    electron_energies = np.load(os.path.join(DATA_PATH, 'all_electron_energies.npy'))
    electron_p_x = np.load(os.path.join(DATA_PATH, 'all_electron_p_x.npy'))
    electron_p_y = np.load(os.path.join(DATA_PATH, 'all_electron_p_y.npy'))
    electron_sorts = np.load(os.path.join(DATA_PATH, 'all_electron_sorts.npy'))

    # 3e22:
    # high_energies = (1000, 1600]
    # mid_energies = (200, 1000]
    # low_energies = (1, 200]

    # 1e22:
    # high_energies = (800, 1200]
    # mid_energies = (200, 800]
    # low_energies = (1, 200]

    # 3e21:
    # high_energies = (800, 1200]
    # mid_energies = (200, 800]
    # low_energies = (1, 200]

    # 5e21:
    # high_energies = (700, 900]
    # mid_energies = (200, 700]
    # low_energies = (1, 200]

    el_proc.all_gamma_factors_result(
        'all',
        electron_sorts,
        electron_energies,
        num_bins=40,
        log_n_scale=True,
        dn_n_d_gamma=False,
        save=True,
        path=os.path.join(DATA_PATH, 'pic')
    )

    el_proc.electrons_angle_distribution_energy_sorted_linear(
        electron_p_x,
        electron_p_y,
        electron_energies,
        num_bins=50,
        mask=True,
        left_energy_value=700.,
        right_energy_value=900.,
        angle_limit_deg_left=2.5,
        angle_limit_deg_right=4.,
        narrow=False,
        save=True,
        intensity_caption_vertical_position=0.05,
        intensity_caption_horizontal_position=0.98,
        path=os.path.join(DATA_PATH, 'pic')
    )

    #print(np.sort(electron_energies)[:10] * cfg.electron_mass)
    #el_proc.electrons_energy_distribution(np.sort(electron_energies)[2:], save=True,
    #                                      path=os.path.join(DATA_PATH, 'pic'))


if __name__ == '__main__':
    main()
