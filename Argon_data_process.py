import os
import numpy as np
import electron_data_process_library as el_proc


DATA_PATH = r"C:\Users\Dns\Desktop\MC_Ion\04-08-2026_19-59-13"


def main():
    electron_energies = np.load(os.path.join(DATA_PATH, 'all_electron_energies.npy'))
    electron_p_x = np.load(os.path.join(DATA_PATH, 'all_electron_p_x.npy'))
    electron_p_y = np.load(os.path.join(DATA_PATH, 'all_electron_p_y.npy'))
    electron_sorts = np.load(os.path.join(DATA_PATH, 'all_electron_sorts.npy'))

    # FORM IS [4, 2, 2] * w_0 BELOW

    # 5e21, wavelength=0.8:
    # low_energies = (1, 150]
    # mid_energies = (150, 600]
    # high_energies = (600, 900]

    # 1e22, wavelength=0.8:
    # low_energies = (1, 200]
    # mid_energies = (200, 800]
    # high_energies = (800, 1100]

    # 1e22, wavelength=1.0:
    # low_energies = (1, 200]
    # mid_energies = (200, 900]
    # high_energies = (900, 1200]

    # 2e22:
    # low_energies = (1, 300]
    # mid_energies = (300, 1000]
    # high_energies = (1000, 1400]

    # 3.5e22, wavelength=0.8:
    # low_energies = (1, 300]
    # mid_energies = (300, 1100]
    # high_energies = (1100, 1500]

    # 4e22:
    # low_energies = (1, 300]
    # mid_energies = (300, 1200]
    # high_energies = (1200, 1600]

    # 6e22:
    # low_energies = (1, 250]
    # mid_energies = (250, 1400]
    # high_energies = (1400, 1800]

    # 8e22:
    # low_energies = (1, 250]
    # mid_energies = (250, 1400]
    # high_energies = (1400, 1900]

    # 1e23:
    # low_energies = (1, 250]
    # mid_energies = (250, 1500]
    # high_energies = (1500, 2000]

    # 2e23:
    # low_energies = (1, 400]
    # mid_energies = (400, 1700]
    # high_energies = (1700, 2500]

    # 4e23:
    # low_energies = (1, 400]
    # mid_energies = (400, 1900]
    # high_energies = (1900, 2600]

    # 6e23:
    # low_energies = (1, 500]
    # mid_energies = (500, 1900]
    # high_energies = (1900, 2800]

    # 8e23:
    # low_energies = (1, 300]
    # mid_energies = (300, 1800]
    # high_energies = (1800, 2600]

    # 1e24:
    # low_energies = (1, 400]
    # mid_energies = (400, 2000]
    # high_energies = (2000, 2900]

    # FORM IS [8, 4, 4] * w_0 BELOW

    # 3.5e22, wavelength=0.8:
    # low_energies = (1, 300]
    # mid_energies = (300, 1000]
    # high_energies = (1000, 1400]

    el_proc.all_gamma_factors_result(
        'all',
        electron_sorts,
        electron_energies,
        num_bins=40,
        log_n_scale=True,
        dn_n_d_gamma=False,
        column_labels=False,
        save=True,
        path=os.path.join(DATA_PATH, 'pic'),
        form_caption=True,
        form_caption_vertical_position=0.95,
        intensity_caption_vertical_position=0.85
    )

    el_proc.electrons_angle_distribution_energy_sorted_linear(
        electron_p_x,
        electron_p_y,
        electron_energies,

        mask=True,
        num_bins=10,

        left_energy_value=1000.,
        right_energy_value=1400.,

        angle_limit_deg_left=2.,
        angle_limit_deg_right=3.,

        narrow=True,
        save=True,

        column_labels=False,
        intensity_caption_vertical_position=0.1,
        form_caption_vertical_position=0.2,
        path=os.path.join(DATA_PATH, 'pic'),
        form_caption=True
    )


if __name__ == '__main__':
    main()
