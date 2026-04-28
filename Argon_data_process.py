import os
import numpy as np

import ion_data_process_library as ion_proc


DATA_PATH = r"C:\Users\Dns\Desktop\MC_Ion\18-04-2026_07-45-14"


def main():
    energies = np.load(os.path.join(DATA_PATH, 'all_electron_energies.npy'))
    ion_proc.electrons_energy_distribution(energies, number_of_bins=110)

    placed_cells = np.load(os.path.join(DATA_PATH, 'placed_cells.npy'))
    fully_ionized_indices = np.load(os.path.join(DATA_PATH, 'fully_ionized_indices.npy'))

    ion_proc.ions_momenta_distribution(
        placed_cells,
        fully_ionized_indices,
        axis=0,
        preferred_charge=18,
        number_of_bins=33
    )
    ion_proc.ions_momenta_distribution(
        placed_cells,
        fully_ionized_indices,
        axis=1,
        preferred_charge=18,
        number_of_bins=33
    )


if __name__ == '__main__':
    main()
