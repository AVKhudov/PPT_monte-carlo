import matplotlib.pyplot as plt
import numpy as np
import ion_config as cfg
import os

from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm, ListedColormap


def electron_angle_energy_chart_sorted(
        p_x,
        p_y,
        energies,
        sorts,
        mask=True,

        left_energy_value=1.,
        right_energy_value=np.inf,

        custom_sorts=None,

        save=False,
        path=None,

        intensity_caption=True,
        intensity_caption_vertical_position=0.95,
        intensity_caption_horizontal_position=0.98,

        form_caption=False,
        form_caption_vertical_position=0.95,
        form_caption_horizontal_position=0.98
):
    if mask:  # отсекаем электроны, летящие назад
        wrong_idx = np.where(p_x < 0.)[0]
        print('el_proc.electron_angle_energy_chart_sorted warning:')
        print(f'- p_x < 0. electrons indices: {wrong_idx}')
        print(f'- number of electrons with p_x < 0.: {len(wrong_idx)} of {len(p_x)}')

        if np.size(wrong_idx) != 0:
            p_x = np.delete(p_x, wrong_idx)
            p_y = np.delete(p_y, wrong_idx)
            energies = np.delete(energies, wrong_idx)
            sorts = np.delete(sorts, wrong_idx)

    energies_idx = np.where((left_energy_value <= energies) & (energies < right_energy_value) & (p_y > 0.))[0]

    radian_angles = np.arctan2(p_y[energies_idx], p_x[energies_idx])
    angles_for_plot = np.rad2deg(radian_angles)

    energies_for_plot = energies[energies_idx]
    sorts_for_plot = sorts[energies_idx]

    if custom_sorts is None:
        sorts_collection = np.unique(sorts_for_plot)
    else:
        sorts_collection = np.array(custom_sorts)

    colors = ['#ADFF2F', '#68E042', '#32CD32', '#008000', '#556B2F', '#6A6B2F', '#9C953A', '#C4BB40', '#FFFF00',
              '#FFDD00', '#FFC300', '#FFAF00', '#FF8C00', '#FF4500', '#FF0000', '#D53015', '#8B0000', 'black']

    sort_colors = {
        sort: colors[sort - 1]
        for sort in range(1, len(colors) + 1)
    }

    available_sorts = []

    for i, single_sort in enumerate(sorts_collection):
        single_sort_idx = np.where(sorts_for_plot == single_sort)[0]

        if np.size(single_sort_idx) == 0:
            print(f'There is no any data for sort {single_sort}')
            continue

        available_sorts.append(single_sort)

        current_energies = energies_for_plot[single_sort_idx]
        current_angles = angles_for_plot[single_sort_idx]

        plt.scatter(
            current_angles,
            current_energies,
            s=8,
            c=sort_colors[single_sort]
        )

    legend_elements = [
        Patch(
            facecolor=sort_colors[single_sort],
            label=f'sort = {int(single_sort)}'
        )
        for single_sort in available_sorts
    ]

    ax = plt.gca()
    ax.set_yscale('log')

    plt.title(f'Sorted angle-energy chart of electrons with gamma factors {left_energy_value}--{right_energy_value}',
              fontsize=14)
    plt.xlabel('Angle, degree', fontsize=14)
    plt.ylabel('Energy, mc^2', fontsize=14)
    ax.tick_params(axis='both', labelsize=14)

    ax.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=12,
        title='Electron sort',
        title_fontsize=12
    )

    if intensity_caption:
        plt.text(
            intensity_caption_horizontal_position,  # default: 0.98
            intensity_caption_vertical_position,  # default: 0.95
            f'Intensity = {cfg.intensity:.1e} W/cm²',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=14,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
        )

    if form_caption:
        plt.text(
            form_caption_horizontal_position,  # default: 0.98
            form_caption_vertical_position,  # default: 0.95
            f'form = {cfg.form}',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=14,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
        )

    plt.tight_layout()

    if save:
        title = f'angle_energy_distribution_{left_energy_value}_{right_energy_value}_energies.jpg'
        if path is None:
            default_path = r"C:\Users\Dns\Desktop"
            default_file_path = os.path.join(default_path, title)
            plt.savefig(default_file_path, bbox_inches='tight')
            plt.close()
        else:
            file_path = os.path.join(path, title)
            plt.savefig(file_path, bbox_inches='tight')

        plt.close()
    else:
        plt.show()


def electrons_energy_distribution(energies, number_of_bins=100, save=False, path=None):
    mev_energies = energies * cfg.electron_mass

    print('max energy, MeV =', np.max(mev_energies))
    print('min energy, MeV =', np.min(mev_energies))

    plt.figure(figsize=(10, 6))

    log_min = np.log10(np.min(mev_energies))
    log_max = np.log10(np.max(mev_energies))

    log_bins = np.logspace(log_min, log_max, number_of_bins + 1)

    counts, bins = np.histogram(mev_energies, log_bins)

    bin_widths = np.diff(bins)
    counts = counts / (np.size(mev_energies) * bin_widths)

    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    plt.loglog(bin_centers, counts)

    plt.xlabel('Энергия электронов, МэВ')
    plt.ylabel('dN/NdE')
    plt.title('Энергии всех электронов')

    ax = plt.gca()
    plt.text(
        0.98, 0.95,
        f'I = {cfg.intensity:.1e} W/cm²',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )

    if save:
        title = 'electrons_energy_distribution.jpg'
        if path is None:
            default_path = r"C:\Users\Dns\Desktop"
            default_file_path = os.path.join(default_path, title)
            plt.savefig(default_file_path, bbox_inches='tight')
        else:
            file_path = os.path.join(path, title)
            plt.savefig(file_path, bbox_inches='tight')

        plt.close()
    else:
        plt.show()


def electrons_visualisation(t_array, ion_array, row_gap=1000, save=False, path=None):

    # Создаем массив для столбцов
    n_segments = len(t_array) // row_gap
    bar_array = np.zeros((n_segments, cfg.z_max))
    x_array = t_array[::row_gap]

    # Вычисляем сумму ионизаций для каждого промежутка
    for i in range(0, len(t_array), row_gap):
        bar_array[i // row_gap] = np.sum(ion_array[i:i + row_gap], axis=0)

    # Рассчитываем пределы
    max_electrons = np.max(np.sum(bar_array, axis=1))
    y_upper_limit = max(max_electrons, np.exp(-(t_array / cfg.tau) ** 2).max() * max_electrons) * 1.1

    # Цвета
    colors = ['#440154', '#481567', '#482677', '#453781', '#404788',
              '#39558C', '#33638D', '#2D708E', '#287D8E', '#238A8D',
              '#1F968B', '#20A387', '#29AF7F', '#3CBB75', '#55C677',
              '#73D055', '#95D840', '#B8DE29']

    # Создаем график
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set(xlim=(t_array[0], t_array[-1]), ylim=(0, y_upper_limit))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Рисуем столбцы
    width = t_array[row_gap] - t_array[0]
    bottom = np.zeros_like(x_array)

    for k in range(cfg.z_max):
        ax.bar(x_array, bar_array[:, k], bottom=bottom, color=colors[k], width=width)
        bottom += bar_array[:, k]

    # Рисуем огибающую
    envelope = np.exp(-(t_array / cfg.tau) ** 2) * max_electrons
    ax.plot(t_array, envelope, 'k-', linewidth=2, label='Огибающая')

    bounds = np.arange(cfg.z_max + 1)
    norm = BoundaryNorm(bounds, len(colors))
    cbar = ColorbarBase(plt.axes([0.92, 0.15, 0.02, 0.7]),
                        cmap=ListedColormap(colors),
                        norm=norm,
                        ticks=np.arange(cfg.z_max) + 0.5)
    cbar.set_ticklabels(np.arange(1, cfg.z_max + 1))
    cbar.set_label('Кратность ионизации', fontsize=12)

    # Подписи
    ax.set_xlabel('Время', fontsize=12)
    ax.set_ylabel('Количество электронов', fontsize=12)
    ax.legend()

    if save:
        title = 'electrons_visualisation.jpg'
        if path is None:
            default_path = r"C:\Users\Dns\Desktop"
            default_file_path = os.path.join(default_path, title)
            plt.savefig(default_file_path, bbox_inches='tight')
        else:
            file_path = os.path.join(path, title)
            plt.savefig(file_path, bbox_inches='tight')

        plt.close()
    else:
        plt.show()


def electrons_momenta_distribution(p_x, p_y, save=False, mask=False, path=None):
    if mask:
        wrong_idx = np.where(p_x < 0.)[0]
        print('Wrong behaviour electrons:', wrong_idx)
        p_x = np.delete(p_x, wrong_idx)
        p_y = np.delete(p_y, wrong_idx)

    plt.figure(figsize=(10, 10))
    plt.scatter(p_x, p_y, color='blue', s=10, alpha=0.5)

    plt.xlim(-max(np.abs(p_x)) * 1.1, max(np.abs(p_x)) * 1.1)
    plt.ylim(-max(np.abs(p_y)) * 1.1, max(np.abs(p_y)) * 1.1)

    plt.xlabel('p_x')
    plt.ylabel('p_y')

    plt.grid(True)

    plt.title('Electrons momenta distribution')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    if save:
        title = 'electrons_energy_distribution.jpg'
        if path is None:
            default_path = r"C:\Users\Dns\Desktop"
            default_file_path = os.path.join(default_path, title)
            plt.savefig(default_file_path, bbox_inches='tight')
        else:
            file_path = os.path.join(path, title)
            plt.savefig(file_path, bbox_inches='tight')

        plt.close()
    else:
        plt.show()


def electron_gamma_factor_vs_time(sort_input, select_input, time_input, px_input, py_input, pz_input,
                                  save=False, title='гамма факторы электронов.jpg', path=None):
    for every_sort in sort_input:
        sort_mask = select_input[:, 1].astype(int) == every_sort
        sort_indices = np.where(sort_mask)[0]

        random_idx = np.random.choice(sort_indices, size=1)[0]
        energy = np.sqrt(
            1
            + px_input[random_idx] ** 2
            + py_input[random_idx] ** 2
            + pz_input[random_idx] ** 2
        )
        time = time_input[random_idx]

        plt.plot(time, energy, label=f'{every_sort}')
    plt.title('Гамма-факторы для электронов разных п-лов ионизации')
    plt.legend()

    if save:
        if path is None:
            default_path = r"C:\Users\Dns\Desktop"
            default_file_path = os.path.join(default_path, title)
            plt.savefig(default_file_path)
        else:
            file_path = os.path.join(path, title)
            plt.savefig(file_path)

        plt.close()
    else:
        plt.show()


def electron_result_gamma_factors(single_sort, select_input, px_input, py_input, pz_input,
                                  save=False, path=None):
    single_sort_mask = select_input[:, 1].astype(int) == single_sort
    single_sort_indices = np.where(single_sort_mask)[0]

    single_sort_gamma_factors_list = []
    for every_idx in single_sort_indices:
        result_gamma_factor = np.sqrt(
            1
            + px_input[every_idx, -1] ** 2
            + py_input[every_idx, -1] ** 2
            + pz_input[every_idx, -1] ** 2
        )
        single_sort_gamma_factors_list.append(result_gamma_factor)

    single_sort_gamma_factors_array = np.array(single_sort_gamma_factors_list)

    plt.figure(figsize=(10, 6))

    counts, bins, patches = plt.hist(single_sort_gamma_factors_array)

    plt.title(f'Гамма-факторы для электронов {single_sort} сорта')
    plt.xlabel('Величина гамма-фактора')
    plt.ylabel('Количество электронов')

    # Подписи над столбцами: значение гамма-фактора и точное число электронов
    for i in range(len(counts)):
        if counts[i] == 0:
            continue

        gamma_center = 0.5 * (bins[i] + bins[i + 1])
        label = f'{gamma_center:.1f}, {int(counts[i])}'

        plt.text(
            gamma_center,
            counts[i],
            label,
            ha='center',
            va='bottom',
            fontsize=8
        )

    # Подпись с интенсивностью
    plt.text(
        0.98, 0.95,
        f'I = {cfg.intensity:.1e} W/cm²',
        transform=plt.gca().transAxes,
        ha='right',
        va='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )

    if save:
        title = f'Гамма-факторы для электронов {single_sort} сорта.jpg'
        if path is None:
            default_path = r"C:\Users\Dns\Desktop"
            default_file_path = os.path.join(default_path, title)
            plt.savefig(default_file_path, bbox_inches='tight')
        else:
            file_path = os.path.join(path, title)
            plt.savefig(file_path, bbox_inches='tight')

        plt.close()
    else:
        plt.show()


def all_gamma_factors_result(
        single_sort,
        sorts_array,
        energy_input,
        num_bins=10,
        log_n_scale=False,
        dn_n_d_gamma=False,
        column_labels=True,
        save=False,
        path=None,
        form_caption=False,
        form_caption_vertical_position=0.95,
        form_caption_horizontal_position=0.98,
        intensity_caption_vertical_position=0.95,
        intensity_caption_horizontal_position=0.98,

):
    if single_sort == 'all':
        single_sort_gamma_factors_array = energy_input
    else:
        single_sort_mask = sorts_array.astype(int) == single_sort
        single_sort_indices = np.where(single_sort_mask)[0]

        single_sort_gamma_factors_list = []
        for every_idx in single_sort_indices:
            single_sort_gamma_factors_list.append(energy_input[every_idx])

        single_sort_gamma_factors_array = np.array(single_sort_gamma_factors_list)

    plt.figure(figsize=(10, 6))

    bin_edges = np.linspace(
        np.min(single_sort_gamma_factors_array),
        np.max(single_sort_gamma_factors_array),
        num_bins + 1
    )

    counts, bins = np.histogram(single_sort_gamma_factors_array, bins=bin_edges)
    raw_counts = counts
    y_label = 'Number of electrons'

    bin_widths = np.diff(bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    if dn_n_d_gamma:
        n_electrons = np.size(single_sort_gamma_factors_array)
        counts = counts / (n_electrons * bin_widths)
        y_label = 'dN / N d_gamma'

    plt.bar(bin_centers, counts, width=bin_widths, align='center')

    ax = plt.gca()

    if log_n_scale:
        ax.set_yscale('log')

    if column_labels is True:
        plot_column_labels(ax, counts, raw_counts, bins, log_n_scale)

    plt.text(
        intensity_caption_horizontal_position,
        intensity_caption_vertical_position,
        f'Intensity = {cfg.intensity:.1e} W/cm²',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=14,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )

    if form_caption:
        plt.text(
            form_caption_horizontal_position,  # default: 0.98
            form_caption_vertical_position,  # default: 0.95
            f'form = {cfg.form}',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=14,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
        )

    plt.title(f'Gamma factors for all electrons', fontsize=16)
    plt.xlabel('Value of gamma factor', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    ax.tick_params(axis='both', labelsize=14)

    # Получаем текущие метки оси X
    x_ticks = ax.get_xticks()

    x_min = 0.  # чтобы включить в метки ноль, котрый потом заменяем на 1
    x_max = np.max(single_sort_gamma_factors_array)
    x_ticks = x_ticks[(x_ticks >= x_min) & (x_ticks <= x_max)]

    x_tick_labels = [str(int(tick)) if tick != 0 else '1' for tick in x_ticks]
    ax.set_xticks(x_ticks)  # <-- Явно устанавливаем позиции
    ax.set_xticklabels(x_tick_labels)

    plt.tight_layout()

    if save:
        title = f'all_gamma_{single_sort}_sort.jpg'
        if path is None:
            default_path = r"C:\Users\Dns\Desktop"
            default_file_path = os.path.join(default_path, title)
            plt.savefig(default_file_path, bbox_inches='tight')
            plt.close()
        else:
            file_path = os.path.join(path, title)
            plt.savefig(file_path, bbox_inches='tight')

        plt.close()
    else:
        plt.show()


def electrons_angle_distribution(p_x, p_y, mask=True, save=False, path=None):
    if mask:
        wrong_idx = np.where(p_x < 0.)[0]
        print('Wrong behaviour electrons:', wrong_idx)
        p_x = np.delete(p_x, wrong_idx)
        p_y = np.delete(p_y, wrong_idx)

    angles = np.arctan(p_y / p_x)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')

    counts, bins = np.histogram(np.array(angles), bins=36)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    ax.scatter(bin_centers, counts, color='red', s=30)

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)

    tick_positions = np.deg2rad([0, 5, 10, 15, 20, 25, 45, 90, 135, 180, 225, 270, 315, 335, 340, 345, 350, 355])
    tick_labels = ['0', '5', '10', '15', '20', '25', '45', '90', '135', '180', '135', '90', '45', '25',
                   '20', '15', '10', '5']

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_title('Углы вылета электронов высокой энергии', pad=20)

    if save:
        title = f'electrons_angle_distribution_high_en.jpg'
        if path is None:
            default_path = r"C:\Users\Dns\Desktop"
            default_file_path = os.path.join(default_path, title)
            plt.savefig(default_file_path, bbox_inches='tight')
        else:
            file_path = os.path.join(path, title)
            plt.savefig(file_path, bbox_inches='tight')

        plt.close()
    else:
        plt.show()


def electrons_angle_distribution_energy_sorted_polar(
        p_x,
        p_y,
        energies,
        left_value,
        right_value,
        num_bins=40,
        angle_limit_deg_left=-7,
        angle_limit_deg_right=7,
        angle_step=1,
        visual_angle_limit_deg=45,
        narrow=False,
        mask=True,
        save=False,
        path=None
):
    if mask:
        wrong_idx = np.where(p_x < 0.)[0]
        print('Wrong behaviour electrons:', wrong_idx, 'len of array:', len(wrong_idx))
        p_x = np.delete(p_x, wrong_idx)
        p_y = np.delete(p_y, wrong_idx)
        energies = np.delete(energies, wrong_idx)

    energies_idx = np.where((left_value < energies) & (energies <= right_value))[0]
    angles = np.arctan(p_y[energies_idx] / p_x[energies_idx])  # радианы

    ax_angles = angles

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)

    if narrow:
        angles_deg = np.rad2deg(angles)
        scale_factor_float = visual_angle_limit_deg / max(np.fabs(angle_limit_deg_right), np.fabs(angle_limit_deg_left))
        scale_factor = int(scale_factor_float)
        ax_angles = np.deg2rad(angles_deg * scale_factor)

        ax.set_thetamin(-visual_angle_limit_deg)
        ax.set_thetamax(visual_angle_limit_deg)

        tick_values_deg = np.arange(angle_limit_deg_left, angle_limit_deg_right + angle_step, angle_step)
        tick_positions_rad = np.deg2rad(tick_values_deg * scale_factor)
        tick_labels = [str(abs(int(x))) for x in tick_values_deg]

        ax.set_xticks(tick_positions_rad)
        ax.set_xticklabels(tick_labels)
    else:
        tick_values_deg = np.arange(0, 360, 45)
        tick_labels = ['0', '45', '90', '135', '180', '135', '90', '45']

        ax.set_xticks(np.deg2rad(tick_values_deg))
        ax.set_xticklabels(tick_labels)

    counts, bins = np.histogram(ax_angles, bins=num_bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ax.scatter(bin_centers, counts, color='red', s=10)

    ax.set_title(
        f'Углы вылета электронов, гамма-факторы {left_value}-{right_value}',
        pad=20
    )

    if save:
        title = f'electrons_angle_distribution_{left_value}_to_{right_value}.jpg'
        if path is None:
            default_path = r"C:\Users\Dns\Desktop"
            default_file_path = os.path.join(default_path, title)
            plt.savefig(default_file_path, bbox_inches='tight')
            plt.close()
        else:
            file_path = os.path.join(path, title)
            plt.savefig(file_path, bbox_inches='tight')

        plt.close()
    else:
        plt.show()


def electrons_angle_distribution_energy_sorted_linear(
        p_x,
        p_y,
        energies,
        left_energy_value=1.,
        right_energy_value=2000.,
        num_bins=40,
        narrow=False,
        angle_limit_deg_left=-10.,
        angle_limit_deg_right=10.,
        mask=True,
        column_labels=True,
        save=False,
        path=None,
        intensity_caption_vertical_position=0.95,
        intensity_caption_horizontal_position=0.90,
        form_caption=False,
        form_caption_vertical_position=0.95,
        form_caption_horizontal_position=0.90
):
    if mask:
        wrong_idx = np.where(p_x < 0.)[0]
        print('el_proc.electrons_angle_distribution_energy_sorted_linear warning:')
        print(f'- p_x < 0. electrons indices: {wrong_idx}')
        print(f'- number of electrons with p_x < 0.: {len(wrong_idx)} of {len(p_x)}')
        p_x = np.delete(p_x, wrong_idx)
        p_y = np.delete(p_y, wrong_idx)
        energies = np.delete(energies, wrong_idx)

    energies_idx = np.where((left_energy_value <= energies) & (energies < right_energy_value))[0]

    radian_angles = np.arctan2(p_y[energies_idx], p_x[energies_idx])
    deg_angles = np.rad2deg(radian_angles)
    angles_for_plot = deg_angles

    if narrow:
        angle_mask = (angle_limit_deg_left <= deg_angles) & (deg_angles < angle_limit_deg_right)
        angles_for_plot = deg_angles[angle_mask]

    bin_edges = np.linspace(
        np.min(angles_for_plot),
        np.max(angles_for_plot),
        num_bins + 1
    )

    counts, bins = np.histogram(angles_for_plot, bins=bin_edges)

    bin_widths = np.diff(bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, counts, width=bin_widths, align='center')

    ax = plt.gca()

    plt.title(f'Angle distribution of electrons with gamma factors {left_energy_value}--{right_energy_value}',
              fontsize=14)
    plt.xlabel('Angle, degree', fontsize=14)
    plt.ylabel('Number of electrons', fontsize=14)
    ax.tick_params(axis='both', labelsize=14)

    '''
    plt.title(f'Gamma factors for all electrons', fontsize=16)
    plt.xlabel('Value of gamma factor', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    '''

    y_max = ax.get_ylim()[1]

    outside_margin = 0.02 * y_max
    inside_margin = 0.05 * y_max
    inside_threshold = 0.85 * y_max

    if column_labels is True:
        for i in range(len(counts)):
            if counts[i] == 0:
                continue

            angle_center = 0.5 * (bins[i] + bins[i + 1])
            label = f'{angle_center:.1f}, {int(counts[i])}'

            if counts[i] >= inside_threshold:
                plt.text(
                    angle_center,
                    counts[i] - inside_margin,
                    label,
                    ha='center',
                    va='top',
                    fontsize=8,
                    rotation=270,
                    color='black'
                )
            else:
                plt.text(
                    angle_center,
                    counts[i] + outside_margin,
                    label,
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=270,
                    color='black'
                )

    plt.tight_layout()

    plt.text(
        intensity_caption_horizontal_position,  # default: 0.98
        intensity_caption_vertical_position,  # default: 0.95
        f'Intensity = {cfg.intensity:.1e} W/cm²',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=14,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )

    if form_caption:
        plt.text(
            form_caption_horizontal_position,  # default: 0.98
            form_caption_vertical_position,  # default: 0.95
            f'form = {cfg.form}',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=14,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
        )

    if save:
        if narrow:
            title = f'angle_distribution_narrow_{left_energy_value}_{right_energy_value}_energies.jpg'
        else:
            title = f'angle_distribution_{left_energy_value}_{right_energy_value}_energies.jpg'
        if path is None:
            default_path = r"C:\Users\Dns\Desktop"
            default_file_path = os.path.join(default_path, title)
            plt.savefig(default_file_path, bbox_inches='tight')
            plt.close()
        else:
            file_path = os.path.join(path, title)
            plt.savefig(file_path, bbox_inches='tight')

        plt.close()
    else:
        plt.show()


def plot_column_labels(axes, bar_heights, label_counts, bins_values, log_scale_status):
    y_maximum = axes.get_ylim()[1]

    outside_margin = 0.02 * y_maximum
    inside_margin = 0.05 * y_maximum
    inside_threshold = 0.85 * y_maximum

    outside_factor = 1.15
    inside_factor = 1.15

    if log_scale_status:
        inside_threshold = y_maximum / 10

    for i in range(len(bar_heights)):
        if label_counts[i] == 0:
            continue

        bin_center = 0.5 * (bins_values[i] + bins_values[i + 1])
        label = f'{bin_center:.1f}, {int(label_counts[i])}'

        text_position_inside = bar_heights[i] - inside_margin
        text_position_outside = bar_heights[i] + outside_margin

        if log_scale_status:
            text_position_inside = bar_heights[i] / inside_factor
            text_position_outside = bar_heights[i] * outside_factor

        if bar_heights[i] >= inside_threshold:
            plt.text(
                bin_center,
                text_position_inside,
                label,
                ha='center',
                va='top',
                fontsize=8,
                rotation=270,
                color='black'
            )
        else:
            plt.text(
                bin_center,
                text_position_outside,
                label,
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=270,
                color='black'
            )

    plt.tight_layout()
