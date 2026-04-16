import os
import numpy as np

import ion_config as cfg
import ion_library as ion
import ion_data_process_library as ion_proc


def run_argon_simulation():
    time_array = np.arange(-cfg.t_0, cfg.t_0, cfg.delta_t, dtype=float)

    initial_placed_cells = ion.placing_cells_with_ions_motion(
        cfg.n_atoms,
        cfg.start_charge_number_of_particles
    )

    placed_cells, ionization_array, electron_motion_input, number_of_electrons, \
        number_of_fully_ionized_states = ion.placed_cells_ionization_rk4(
            initial_placed_cells,
            time_array
        )

    all_p_x, all_p_y, all_p_z = ion.electron_parallel_simulation(electron_motion_input)
    all_energies = np.sqrt(1 + all_p_x ** 2 + all_p_y ** 2 + all_p_z ** 2)
    all_angles = np.vectorize(ion_proc.angle_calculation)(all_p_y, all_p_x)

    return {
        "time_array": time_array,
        "placed_cells": placed_cells,
        "ionization_array": ionization_array,
        "electron_motion_input": electron_motion_input,
        "number_of_electrons": number_of_electrons,
        "number_of_fully_ionized_states": number_of_fully_ionized_states,
        "all_p_x": all_p_x,
        "all_p_y": all_p_y,
        "all_p_z": all_p_z,
        "all_energies": all_energies,
        "all_angles": all_angles,
    }


def save_simulation_results(simulation_results, data_output_path):
    ion_momenta = simulation_results["placed_cells"][:, 3:6]

    ion_proc.save_file(data_output_path, 'ion_momenta_array.npy', ion_momenta)
    ion_proc.save_file(data_output_path, 'electron_motion_array.npy', simulation_results["electron_motion_input"])

    results_of_ionization = {
        "Число электронов": simulation_results["number_of_electrons"],
        "Число полностью ионизированных состояний": simulation_results["number_of_fully_ionized_states"],
        "Число атомов": cfg.n_atoms,
        "Сорт частиц мишени, Z - 1 (0 соотв. нейтральным атомам)": cfg.start_charge_number_of_particles
    }

    with open(os.path.join(data_output_path, 'results_of_simulations.txt'), 'a', encoding='utf-8') as f:
        for key, value in results_of_ionization.items():
            f.write(f"{key}: {value}\n")

    text_params = {
        "Число атомов": cfg.n_atoms,
        "Интенсивность поля в фокусе (Вт/см^2)": f"{cfg.intensity:.2e}",
        "Амплитуда поля (ат. ед.)": f"{cfg.atomic_field:.0f}",
        "Параметр a_0": f"{cfg.a_0:.0f}",
        "Эллиптичность поля": cfg.eps,
        "Частота поля (СИ)": f"{cfg.frequency:.2e}",
        "Радиус перетяжки в длинах волн": cfg.w_0,
        "Длительность импульса (omega * tau)": cfg.tau,
        "Длительность импульса (fs)": f"{cfg.tau / cfg.frequency * 1e15:.1f}",
        "Длительность симуляции (omega * t)": cfg.t_0,
        "Длительность симуляции (fs)": f"{cfg.t_0 / cfg.frequency * 1e15:.1f}",
        "Разрешение по времени (omega * delta_t)": cfg.delta_t,
        "Максимальное зарядовое число": cfg.z_max,
        "Сорт частиц мишени, Z - 1 (0 соотв. нейтральным атомам)": cfg.start_charge_number_of_particles,
        "Форма мишени в длинах волн": cfg.form,
    }

    with open(os.path.join(data_output_path, 'parameters_of_simulation.txt'), 'a', encoding='utf-8') as f:
        f.write("Параметры симуляции\n")
        for key, value in text_params.items():
            f.write(f"{key}: {value}\n")

    params = [cfg.n_atoms, cfg.tau, cfg.t_0, cfg.delta_t, cfg.z_max]
    ion_proc.save_file(data_output_path, 'params.npy', np.array(params))

    ion_proc.save_file(data_output_path, 'time_array.npy', simulation_results["time_array"])
    ion_proc.save_file(data_output_path, 'ionization_array.npy', simulation_results["ionization_array"])
    ion_proc.save_file(data_output_path, 'placed_cells.npy', simulation_results["placed_cells"])

    ion_proc.save_file(data_output_path, 'all_electron_p_x.npy', simulation_results["all_p_x"])
    ion_proc.save_file(data_output_path, 'all_electron_p_y.npy', simulation_results["all_p_y"])
    ion_proc.save_file(data_output_path, 'all_electron_p_z.npy', simulation_results["all_p_z"])
    ion_proc.save_file(data_output_path, 'all_electron_energies.npy', simulation_results["all_energies"])
    ion_proc.save_file(data_output_path, 'all_electron_angles.npy', simulation_results["all_angles"])


if __name__ == '__main__':  # НЕ УБИРАТЬ!!! ВАЖНО!!!
    results = run_argon_simulation()
    print('\n')

    output_path = ion_proc.create_timestamp_folder()
    print('output path:', output_path)

    save_simulation_results(results, output_path)
