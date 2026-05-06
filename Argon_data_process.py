import os
import numpy as np

import ion_data_process_library as ion_proc
import electron_data_process_library as el_proc


DATA_PATH = r"C:\Users\Dns\Desktop\MC_Ion\06-05-2026_22-16-15"


def main():
    #energies = np.load(os.path.join(DATA_PATH, 'all_electron_energies.npy'))
    #el_proc.electrons_energy_distribution(energies, number_of_bins=110)

    #placed_cells = np.load(os.path.join(DATA_PATH, 'placed_cells.npy'))
    #ion_proc.final_ion_charge_bar(placed_cells)
    #fully_ionized_indices = np.load(os.path.join(DATA_PATH, 'fully_ionized_indices.npy'))

    selected_input = np.load(os.path.join(DATA_PATH, 'selected_trajectory_input.npy'))

    trajectory_t = np.load(os.path.join(DATA_PATH, 'trajectory_t.npy'))
    trajectory_p_x = np.load(os.path.join(DATA_PATH, 'trajectory_p_x.npy'))
    trajectory_p_y = np.load(os.path.join(DATA_PATH, 'trajectory_p_y.npy'))
    trajectory_p_z = np.load(os.path.join(DATA_PATH, 'trajectory_p_z.npy'))

    selected_sorts = [1, 2, 3]

    el_proc.electron_gamma_factor_vs_time(selected_sorts, selected_input, trajectory_t,
                                          trajectory_p_x, trajectory_p_y, trajectory_p_z)

    '''
    ion_proc.ions_momenta_distribution(
        placed_cells,
        fully_ionized_indices,
        axis=1,
        preferred_charge=-1,
        number_of_bins=70,
        save=False,
        title='ions_momenta_z_17.jpg',
        path=DATA_PATH
    )
    '''


if __name__ == '__main__':
    main()
