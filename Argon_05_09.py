import ion_library as ion
import os

from ion_config import *

if __name__ == '__main__':  # НЕ УБИРАТЬ!!! ВАЖНО!!!
    placed_cells = ion.placing_cells(n)

    # Основные циклы
    all_energies = []
    all_angles = []

    fully_ionized_states_list = []

    electron_motion_list = []

    # Цикл с ионизацией атомов, без вычисления движения электронов
    for i, current_moment in enumerate(time_array):

        progress = (i + 1) / len(time_array) * 100  # прогресс обратботки массива в процентах
        print(f'\rПрогресс: {i + 1}/{len(time_array)} ({progress:.1f}%)', end='')  # вывод значения прогресса

        # генерация случайных чисел для каждого атома для сравнения с w_ppt на каждом шаге по времени
        random_values = np.random.random(n)
        # вероятность ионизации каждого атома в момент времени current_moment:
        probabilities = ion.w_ppt_vectorized(current_moment, placed_cells)  # уже умножено на delta_t (см. ion_library)
        # работаем только с теми атомами, которые ионизовались на данном шаге по времени
        mask = random_values < probabilities
        true_indices_in_mask = np.where(mask)[0]  # массив индексов атомов/ионов, которые ПО ПОЛОЖЕНИЮ В ПРОСТРАНСТВЕ
        # подходят для ионизации. Часть индексов может соответствовать полностью ионизованным состояниям, которые
        # необходимо исключить из рассмотрения в цикле ниже.

        true_indices_in_mask_list = true_indices_in_mask.tolist()

        not_fully_ionized_states_list = [index for index in true_indices_in_mask_list
                                         if index not in fully_ionized_states_list]  # исключение
        # индексов полностью ионизованных состояний их списка индексов, где верна маска mask

        not_fully_ionized_states_array = np.array(not_fully_ionized_states_list)

        for j in not_fully_ionized_states_array:  # j пробегает значения тех индексов, где в массиве mask стоит True,
            # и при этом их нет в списке индексов, соответствующих полностью ионизованным состояниям.
            if placed_cells[j, 3] == z_max:
                fully_ionized_states_list.append(j)
                continue
            ionization_array[i, int(placed_cells[j, 3]) - 1] += 1  # инкрементируем число электронов в данный момент
            # времени для данного потенциала ионизации
            placed_cells[j, 3:] = ionization_order_array[int(placed_cells[j, 3])]  # переводим атом/ион в следующее
            # состояние
            electron_motion_list.append(np.append(current_moment, placed_cells[j, :3]))  # готовим список из начальных
            # положений и моментов ионизации электронов для последующих параллельных вычислений
            # при решении уравнений Лоренца

    motion_data = np.array(electron_motion_list)
    all_p_x, all_p_y, all_p_z = ion.parallel_simulation(motion_data)  # параллельные вычисления!
    all_energies = np.sqrt(1 + all_p_x ** 2 + all_p_y ** 2 + all_p_z ** 2)
    all_angles = np.vectorize(ion.angle_calculation)(all_p_y, all_p_x)

    number_of_electrons = len(all_p_x)
    number_of_fully_ionized_states = len(fully_ionized_states_list)

    print('\n')
    print('number of electrons =', number_of_electrons)
    print('number of full ionized states =', number_of_fully_ionized_states)

    # Вывод данных в файлы:

    output_path = ion.create_timestamp_folder()  # создаем папку с текущей датой и временем
    print('output path:', output_path)

    # Словарь с параметрами для вывода в файл
    text_params = {
        "Число атомов": n,
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
        "Разрешение в пространстве в длинах волн": step,
        "Максимальное зарядовое число": z_max,
        "Форма мишени в длинах волн": form
    }

    with open(os.path.join(output_path, 'parameters_of_simulation.txt'), 'a', encoding='utf-8') as f:
        f.write("Параметры симуляции\n")
        for key, value in text_params.items():
            f.write(f"{key}: {value}\n")

    params = [n, tau, t_0, delta_t, step, z_max]  # те параметры, которые необходимые для обработки данных
    ion.save_file(output_path, 'params.npy', np.array(params))

    ion.save_file(output_path, 'time_array.npy', time_array)
    ion.save_file(output_path, 'ionization_array.npy', ionization_array)

    ion.save_file(output_path, 'placed_cells.npy', placed_cells)

    ion.save_file(output_path, 'electron_motion_array.npy', motion_data)
    ion.save_file(output_path, 'all_electron_p_x.npy', all_p_x)
    ion.save_file(output_path, 'all_electron_p_y.npy', all_p_y)
    ion.save_file(output_path, 'all_electron_p_z.npy', all_p_z)
    ion.save_file(output_path, 'all_electron_energies.npy', all_energies)
    ion.save_file(output_path, 'all_electron_angles.npy', all_angles)
