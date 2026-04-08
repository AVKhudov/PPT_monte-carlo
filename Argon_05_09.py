import ion_library as ion
import os

from ion_config import *

if __name__ == '__main__':  # НЕ УБИРАТЬ!!! ВАЖНО!!!

    placed_cells = ion.placing_cells_with_ions_motion(n_atoms, start_charge_number_of_particles)
    motion_data, ion_data, number_of_electrons, number_of_fully_ionized_states \
        = ion.placed_cells_ionization_rk4(placed_cells)

    # Вывод данных в файлы:
    print('\n')
    output_path = ion.create_timestamp_folder()  # создаем папку с текущей датой и временем
    print('output path:', output_path)

    ion_momenta = ion_data[:, 3:6]

    ion.save_file(output_path, 'ion_momenta_array.npy', ion_momenta)  # импульс ионов
    ion.save_file(output_path, 'electron_motion_array.npy', motion_data)  # записываем нач. данные по эл-нам сразу

    results_of_ionization = {
        "Число электронов": number_of_electrons,
        "Число полностью ионизованных состояний": number_of_fully_ionized_states,
        "Число атомов": n_atoms,
        "Сорт частиц мишени, Z - 1 (0 соотв. нейтральным атомам)": start_charge_number_of_particles
    }

    with open(os.path.join(output_path, 'results_of_simulations.txt'), 'a', encoding='utf-8') as f:
        for key, value in results_of_ionization.items():
            f.write(f"{key}: {value}\n")

    all_p_x, all_p_y, all_p_z = ion.electron_parallel_simulation(motion_data)  # параллельные вычисления!
    all_energies = np.sqrt(1 + all_p_x ** 2 + all_p_y ** 2 + all_p_z ** 2)  # энергия в единицах mc^2
    all_angles = np.vectorize(ion.angle_calculation)(all_p_y, all_p_x)

    # Словарь с параметрами для вывода в файл
    text_params = {
        "Число атомов": n_atoms,
        "Интенсивность поля в фокусе (Вт/см^2)": f"{intensity:.2e}",
        "Амплитуда поля (ат. ед.)": f"{atomic_field:.0}",
        "Параметр a_0": f"{a_0:.0f}",
        "Эллиптичность поля": eps,
        "Частота поля (СИ)": f"{frequency:.2e}",
        "Радиус перетяжки в длинах волн": f"{w_0}",
        "Длительность импульса (omega * tau)": tau,
        "Длительность импульса (fs)": f"{tau / frequency * 1e15:.1f}",
        "Длительность симуляции (omega * t)": t_0,
        "Длительность симуляции (fs)": f"{t_0 / frequency * 1e15:.1f}",
        "Разрешение по времени (omega * delta_t)": delta_t,
        "Максимальное зарядовое число": z_max,
        "Сорт частиц мишени, Z - 1 (0 соотв. нейтральным атомам)": start_charge_number_of_particles,
        "Форма мишени в длинах волн": form,
    }

    with open(os.path.join(output_path, 'parameters_of_simulation.txt'), 'a', encoding='utf-8') as f:
        f.write("Параметры симуляции\n")
        for key, value in text_params.items():
            f.write(f"{key}: {value}\n")

    params = [n_atoms, tau, t_0, delta_t, z_max]  # параметры, необходимые для обработки данных
    ion.save_file(output_path, 'params.npy', np.array(params))

    ion.save_file(output_path, 'time_array.npy', time_array)
    ion.save_file(output_path, 'ionization_array.npy', ionization_array)

    ion.save_file(output_path, 'placed_cells.npy', placed_cells)

    ion.save_file(output_path, 'all_electron_p_x.npy', all_p_x)
    ion.save_file(output_path, 'all_electron_p_y.npy', all_p_y)
    ion.save_file(output_path, 'all_electron_p_z.npy', all_p_z)
    ion.save_file(output_path, 'all_electron_energies.npy', all_energies)
    ion.save_file(output_path, 'all_electron_angles.npy', all_angles)
