#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include<omp.h>
#include<time.h>
#include "boris.h"

#define PI 3.14159265358979323846

static double cos_envelope(
    double moment_of_time,
    double x,
    double tau
)
{
     double env_phase = moment_of_time - 2.0 * PI * x;

     if (-tau <= env_phase && env_phase <= tau)
     {
         double cos_part_envelope = cos(PI * env_phase / tau / 2.0);
         double envelope = cos_part_envelope * cos_part_envelope;
         return envelope;
     }
     else
     {
         return 0.0;
     }
}

static double electric_field(
    double x,
    double y,
    double z,

    double moment_of_time,

    double beam_radius,
    double tau,
    double atomic_field,
    double ell_value,
    double x_r
)
{
    double electric_field_value;
    double envelope = cos_envelope(moment_of_time, x ,tau);

    if (envelope == 0.0)
    {
        electric_field_value = 0.0;
    }

    else
    {
        double r_squared = y * y + z * z;
        double x_xr = x / x_r;
        double xr_x = x_r / x;

        double phi = atan(x_xr);
        double rho = x * (1.0 + xr_x * xr_x);

        double phase = 2.0 * PI * x - moment_of_time - phi + PI * r_squared / rho;

        double radius = sqrt(1.0 + x_xr * x_xr);

        double w = beam_radius * radius;
        double spatial_part = 1.0 / radius * exp(-r_squared / (w * w));

        double cos_comp = spatial_part * envelope * cos(phase);
        double sin_comp = spatial_part * envelope * sin(phase);

        double ell_denominator = sqrt(1.0 + ell_value * ell_value);

        double ellipticity_0 = 1.0 / ell_denominator;
        double ellipticity_1 = ell_value / ell_denominator;

        electric_field_value = sqrt(
            (cos_comp * ellipticity_0) * (cos_comp * ellipticity_0)
            +
            (sin_comp * ellipticity_1) * (sin_comp * ellipticity_1)
        ) * atomic_field;
    }

    return electric_field_value;
}


static uint32_t randomuint(uint32_t * state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static double random_double(uint32_t *state)
{
    return (double)randomuint(state) / (double)UINT32_MAX;
}


static double single_atom_ionization_probability(
    double moment_of_time,
    double delta_t,

    double x,
    double y,
    double z,

    double beam_radius,
    double tau,
    double atomic_field,
    double ell_value,
    double x_r,

    double i_p,
    double c_value,
    double b_value,
    double n_star_value,

    double m_value,
    double g_m_value

)
{
    double electric_field_value = electric_field(
        x,
        y,
        z,

        moment_of_time,

        beam_radius,
        tau,
        atomic_field,
        ell_value,
        x_r
    );

    double field_char = pow((2.0 * i_p), 1.5);
    double field = electric_field_value / field_char;

    double field_power = 2 * n_star_value - fabs(m_value) - 1;

    double probability;

    probability = (
        4.0
        * c_value
        * b_value
        * i_p
        * pow((2.0 / field), field_power)
        * exp(-2.0 / (3.0 * field))
        * g_m_value
        * delta_t
    );

    return probability;
}


static void single_ion_process(
    double atom[7], // атом
    double delta_t,  // шаг по времени

    const int max_ion,  // максимальное зарядовое число и стартовый заряд частиц
    const int start_charge_number,

    double beam_radius,  // параметры поля
    double tau,
    double atomic_field,
    double ell_value,
    double x_r,

    const int (*ionization_order_array)[4], // массивы для одиночного процесса
    const double *ionization_potentals,
    const double *c_n_l_array,
    const double *b_l_m_array,
    const double *n_star_array,

    double (*tmp)[5],  // сюда пишем строки [t, sort, x, y, z]
    int atom_index,  // индекс атома в процессе
    int *ionization_counts_array  // сюда пишем число ионизаций для атома, которые случились в процессе
)
{
    double x = atom[0];  // распаковка координат атома
    double y = atom[1];
    double z = atom[2];

    double i_p;  // величины для вычисления вероятности
    double c_value;
    double b_value;
    double n_star_value;

    double m_value;  // распакуем в цикле while из атома
    double g_m_value;

    double probability;  // вероятность ионизации

    double start_time = 2.0 * PI * x - tau;  // работаем с атомом только в те моменты времени, когда в его точке есть поле
    double finish_time = 2.0 * PI * x + tau;

    double current_time = start_time;  // текущее время, которое будет обновляться в цикле

    int ionization_count = 0;  // пишем сюда число ионизаций, прошедших в цикле while ниже
    int offset = (max_ion - start_charge_number) * atom_index;  // сколько строк на один атом (равно числу электронов, которые атом или ион может отдать)

    uint32_t seed = 12345u + (uint32_t)atom_index;  // seed для рандома

    while (current_time <= finish_time)
    {
        int current_index = (int)atom[3] - 1;  // на элементе с каким индексом мы находимся в текущий момент

        i_p = ionization_potentals[current_index];
        c_value = c_n_l_array[current_index];
        b_value = b_l_m_array[current_index];
        n_star_value = n_star_array[current_index];

        m_value = atom[5];  // распаковали m и g_|m|
        g_m_value = atom[6];

        probability = single_atom_ionization_probability(
            current_time,
            delta_t,

            x,
            y,
            z,

            beam_radius,
            tau,
            atomic_field,
            ell_value,
            x_r,

            i_p,
            c_value,
            b_value,
            n_star_value,

            m_value,
            g_m_value
        );

        double random_value = random_double(&seed);

        if (random_value < probability)
        {
            tmp[offset + ionization_count][0] = current_time;
            tmp[offset + ionization_count][1] = (start_charge_number + 1) + ionization_count;

            tmp[offset + ionization_count][2] = x;
            tmp[offset + ionization_count][3] = y;
            tmp[offset + ionization_count][4] = z;

            ionization_count += 1;

            if (ionization_count == max_ion - start_charge_number)
            {
                break;
            }

            atom[3] = ionization_order_array[current_index + 1][0];  // обновляем состояние атома
            atom[4] = ionization_order_array[current_index + 1][1];
            atom[5] = ionization_order_array[current_index + 1][2];
            atom[6] = ionization_order_array[current_index + 1][3];
        }

        current_time += delta_t;
    }

    ionization_counts_array[atom_index] = ionization_count;
}


static void parallel_ionization(
    double (*placed_cells)[7],
    int number_of_atoms,

    double delta_t,

    int max_ion,
    int start_charge_number,

    double beam_radius,
    double tau,
    double atomic_field,
    double ell_value,
    double x_r,

    const int (*ionization_order_array)[4],
    const double *ionization_potentials,
    const double *c_n_l_array,
    const double *b_l_m_array,
    const double *n_star_array,

    double (*tmp)[5],  // сюда пишем начальные условия по электронам
    int *ionization_counts_array  // сюда пишем кол-ва ионизаций по атомам
)
{
    #pragma omp parallel for
    for (int atom_index = 0; atom_index < number_of_atoms; atom_index++)
    {
        single_ion_process(
            placed_cells[atom_index],

            delta_t,

            max_ion,
            start_charge_number,

            beam_radius,
            tau,
            atomic_field,
            ell_value,
            x_r,

            ionization_order_array,
            ionization_potentials,
            c_n_l_array,
            b_l_m_array,
            n_star_array,

            tmp,
            atom_index,
            ionization_counts_array
        );
    }
}


double *ionization_simulation(
    double (*placed_cells)[7],
    int number_of_atoms,

    double delta_t,

    int max_ion,
    int start_charge_number,

    double beam_radius,
    double tau,
    double atomic_field,
    double ell_value,
    double x_r,

    const int (*ionization_order_array)[4],
    const double *ionization_potentials,
    const double *c_n_l_array,
    const double *b_l_m_array,
    const double *n_star_array,

    int *number_of_results
)
{
    double (*tmp)[5] = malloc(
        number_of_atoms
        * (max_ion - start_charge_number)
        * sizeof(*tmp)  // sizeof(*tmp) - размер строки из 5 double
    );

    if (tmp == NULL)
        return NULL;

    int* ionization_counts_array = malloc(
        number_of_atoms * sizeof(*ionization_counts_array)
    );

    if (ionization_counts_array == NULL)
    {
        free(tmp);
        return NULL;
    }

    parallel_ionization(
        placed_cells,
        number_of_atoms,

        delta_t,

        max_ion,
        start_charge_number,

        beam_radius,
        tau,
        atomic_field,
        ell_value,
        x_r,

        ionization_order_array,
        ionization_potentials,
        c_n_l_array,
        b_l_m_array,
        n_star_array,

        tmp,
        ionization_counts_array
    );

    int number_of_electrons = 0;

    for (int atom_index = 0; atom_index < number_of_atoms; atom_index++)
    {
        number_of_electrons += ionization_counts_array[atom_index];
    }

    double (*result)[5] = malloc(
        number_of_electrons * sizeof(*result)  // result - будущий electron_motion_input
    );

    if (result == NULL)
    {
        free(tmp);
        free(ionization_counts_array);
        return NULL;
    }

    int result_index = 0;

    for (int atom_index = 0; atom_index < number_of_atoms; atom_index++)
    {
        int atom_count = ionization_counts_array[atom_index];
        int offset = (max_ion - start_charge_number) * atom_index;

        for (int i = 0; i < atom_count; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                result[result_index][j] = tmp[offset + i][j];
            }

            result_index++;
        }
    }

    free(tmp);
    free(ionization_counts_array);

    *number_of_results = number_of_electrons;

    return (double *)result;
}



void free_memory(double *ptr)
{
    free(ptr);
}


static void boris_field(
    double x,
    double y,
    double z,

    double moment_of_time,

    double beam_radius,
    double tau,
    double a_0_value,
    double ell_value,
    double r_or_l,
    double x_r,

    double electric_field[3],
    double magnetic_field[3]
)
{
    double envelope = cos_envelope(moment_of_time, x, tau);

    if (envelope == 0.0)
    {
        electric_field[0] = 0.0;
        electric_field[1] = 0.0;
        electric_field[2] = 0.0;

        magnetic_field[0] = 0.0;
        magnetic_field[1] = 0.0;
        magnetic_field[2] = 0.0;

        return;
    }

    double r_squared = y * y + z * z;
    double x_xr = x / x_r;
    double xr_x = x_r / x;

    double phi = atan(x_xr);
    double rho = x * (1.0 + xr_x * xr_x);

    double phase = 2.0 * PI * x - moment_of_time - phi + PI * r_squared / rho;

    double radius = sqrt(1.0 + x_xr * x_xr);

    double w = beam_radius * radius;
    double spatial_part = 1.0 / radius * exp(-r_squared / (w * w));

    double cos_comp = spatial_part * envelope * cos(phase);
    double sin_comp = spatial_part * envelope * sin(phase);

    double ell_denominator = sqrt(1.0 + ell_value * ell_value);

    double ellipticity_0 = 1.0 / ell_denominator;
    double ellipticity_1 = ell_value / ell_denominator;

    electric_field[0] = 0.0;
    electric_field[1] = a_0_value * cos_comp * ellipticity_0;
    electric_field[2] = a_0_value * sin_comp * ellipticity_1 * r_or_l;

    magnetic_field[0] = 0.0;
    magnetic_field[1] = -a_0_value * sin_comp * ellipticity_1 * r_or_l;
    magnetic_field[2] = a_0_value * cos_comp * ellipticity_0;
}


static void boris_relativistic_step(
    double time,
    const double radius_vector[3],
    const double p_half[3],

    double delta_t,

    double beam_radius,
    double tau,
    double a_0_value,
    double ell_value,
    double r_or_l,
    double x_r,

    double radius_vector_new[3],
    double p_new_half[3]
)
{
    double x = radius_vector[0];
    double y = radius_vector[1];
    double z = radius_vector[2];

    double electric_field[3];
    double magnetic_field[3];

    double charge = -1.0;

    double coefficient = 1.0 / (2.0 * PI);

    if (cos_envelope(time, x, tau) == 0)
    {
        double p_squared =
            p_half[0] * p_half[0] +
            p_half[1] * p_half[1] +
            p_half[2] * p_half[2];

        double gamma_half = sqrt(1.0 + p_squared);

        double v_half[3];

        v_half[0] = p_half[0] / gamma_half;
        v_half[1] = p_half[1] / gamma_half;
        v_half[2] = p_half[2] / gamma_half;

        radius_vector_new[0] =
            radius_vector[0] + coefficient * v_half[0] * delta_t;

        radius_vector_new[1] =
            radius_vector[1] + coefficient * v_half[1] * delta_t;

        radius_vector_new[2] =
            radius_vector[2] + coefficient * v_half[2] * delta_t;

        p_new_half[0] = p_half[0];
        p_new_half[1] = p_half[1];
        p_new_half[2] = p_half[2];

        return;
    }

    boris_field(
        x, y, z,
        time,
        beam_radius,
        tau,
        a_0_value,
        ell_value,
        r_or_l,
        x_r,
        electric_field,
        magnetic_field
    );

    electric_field[0] = electric_field[0] * charge;
    electric_field[1] = electric_field[1] * charge;
    electric_field[2] = electric_field[2] * charge;

    magnetic_field[0] = magnetic_field[0] * charge;
    magnetic_field[1] = magnetic_field[1] * charge;
    magnetic_field[2] = magnetic_field[2] * charge;

    /*
     * ---------------------------------------------------------
     * 1. Electric half-kick
     *
     * p_minus = p_half + 0.5 * E * delta_t
     * ---------------------------------------------------------
     */

    double p_minus[3];

    p_minus[0] =
        p_half[0] + 0.5 * electric_field[0] * delta_t;

    p_minus[1] =
        p_half[1] + 0.5 * electric_field[1] * delta_t;

    p_minus[2] =
        p_half[2] + 0.5 * electric_field[2] * delta_t;

    /*
     * gamma_minus = sqrt(1 + p_minus^2)
     */
    double p_minus_squared =
        p_minus[0] * p_minus[0] +
        p_minus[1] * p_minus[1] +
        p_minus[2] * p_minus[2];

    double gamma_minus =
        sqrt(1.0 + p_minus_squared);

    /*
     * ---------------------------------------------------------
     * 2. Relativistic magnetic rotation
     *
     * t = 0.5 * B * delta_t / gamma_minus
     * ---------------------------------------------------------
     */

    double t_vector[3];

    t_vector[0] = 0.5 * magnetic_field[0] * delta_t / gamma_minus;
    t_vector[1] = 0.5 * magnetic_field[1] * delta_t / gamma_minus;
    t_vector[2] = 0.5 * magnetic_field[2] * delta_t / gamma_minus;

    double t_squared =
        t_vector[0] * t_vector[0] +
        t_vector[1] * t_vector[1] +
        t_vector[2] * t_vector[2];

    /*
     * s = 2*t / (1 + t^2)
     */
    double s_vector[3];

    s_vector[0] = 2.0 * t_vector[0] / (1.0 + t_squared);
    s_vector[1] = 2.0 * t_vector[1] / (1.0 + t_squared);
    s_vector[2] = 2.0 * t_vector[2] / (1.0 + t_squared);

    /*
     * ---------------------------------------------------------
     * 3. First cross product
     *
     * p_prime = p_minus + p_minus × t
     * ---------------------------------------------------------
     */

    double cross_pt[3];

    cross_pt[0] =
        p_minus[1] * t_vector[2]
        - p_minus[2] * t_vector[1];

    cross_pt[1] =
        p_minus[2] * t_vector[0]
        - p_minus[0] * t_vector[2];

    cross_pt[2] =
        p_minus[0] * t_vector[1]
        - p_minus[1] * t_vector[0];

    double p_prime[3];

    p_prime[0] = p_minus[0] + cross_pt[0];
    p_prime[1] = p_minus[1] + cross_pt[1];
    p_prime[2] = p_minus[2] + cross_pt[2];

    /*
     * ---------------------------------------------------------
     * 4. Second cross product
     *
     * p_plus = p_minus + p_prime × s
     * ---------------------------------------------------------
     */

    double cross_ps[3];

    cross_ps[0] =
        p_prime[1] * s_vector[2]
        - p_prime[2] * s_vector[1];

    cross_ps[1] =
        p_prime[2] * s_vector[0]
        - p_prime[0] * s_vector[2];

    cross_ps[2] =
        p_prime[0] * s_vector[1]
        - p_prime[1] * s_vector[0];

    double p_plus[3];

    p_plus[0] = p_minus[0] + cross_ps[0];
    p_plus[1] = p_minus[1] + cross_ps[1];
    p_plus[2] = p_minus[2] + cross_ps[2];

    /*
     * ---------------------------------------------------------
     * 5. Second electric half-kick
     *
     * p_new_half = p_plus + 0.5 * E * delta_t
     * ---------------------------------------------------------
     */

    p_new_half[0] = p_plus[0] + 0.5 * electric_field[0] * delta_t;
    p_new_half[1] = p_plus[1] + 0.5 * electric_field[1] * delta_t;
    p_new_half[2] = p_plus[2] + 0.5 * electric_field[2] * delta_t;

    double p_new_squared =
        p_new_half[0] * p_new_half[0] +
        p_new_half[1] * p_new_half[1] +
        p_new_half[2] * p_new_half[2];

    double gamma_new = sqrt(1.0 + p_new_squared);

    double v_new_half[3];

    v_new_half[0] = p_new_half[0] / gamma_new;
    v_new_half[1] = p_new_half[1] / gamma_new;
    v_new_half[2] = p_new_half[2] / gamma_new;

    radius_vector_new[0]
        = radius_vector[0] + coefficient * v_new_half[0] * delta_t;

    radius_vector_new[1]
        = radius_vector[1] + coefficient * v_new_half[1] * delta_t;

    radius_vector_new[2]
        = radius_vector[2] + coefficient * v_new_half[2] * delta_t;
}

static void single_boris_process(
    const double initial_data[5],

    double el_sim_duration,
    double delta_t,

    double beam_radius,
    double tau,
    double a_0_value,
    double ell_value,
    double r_or_l,
    double x_r,

    double *sort,
    double *initial_t,
    double final_p[3]
)
{
    /*
     * Данные электрона при рождении.
     *
     * initial_data = [t, sort, x, y, z]
     */
    *initial_t = initial_data[0];
    *sort = initial_data[1];

    /*
     * Начальный момент времени.
     */
    double current_time = initial_data[0];

    /*
     * Начальная координата.
     */
    double current_r[3];

    current_r[0] = initial_data[2];
    current_r[1] = initial_data[3];
    current_r[2] = initial_data[4];

    /*
     * Начальный импульс.
     *
     * Электрон рождается с нулевым импульсом.
     */
    double current_p[3] = {
        0.0,
        0.0,
        0.0
    };

    /*
     * Основной цикл интегрирования.
     */
    while (current_time < el_sim_duration)
    {
        double new_r[3];
        double new_p[3];

        boris_relativistic_step(
            current_time,
            current_r,
            current_p,

            delta_t,

            beam_radius,
            tau,
            a_0_value,
            ell_value,
            r_or_l,
            x_r,

            new_r,
            new_p
        );

        current_r[0] = new_r[0];
        current_r[1] = new_r[1];
        current_r[2] = new_r[2];

        current_p[0] = new_p[0];
        current_p[1] = new_p[1];
        current_p[2] = new_p[2];

        current_time += delta_t;
    }

    final_p[0] = current_p[0];
    final_p[1] = current_p[1];
    final_p[2] = current_p[2];
}



void boris_parallel_simulation(
    const double *el_m_input,
    int number_of_electrons,

    double el_sim_duration,
    double delta_t,

    double beam_radius,
    double tau,
    double a_0_value,
    double ell_value,
    double r_or_l,
    double x_r,

    double *sorts_arr,
    double *initial_t_arr,
    double *p_x_arr,
    double *p_y_arr,
    double *p_z_arr
)
{
    #pragma omp parallel for
    for (int electron_idx = 0;
         electron_idx < number_of_electrons;
         electron_idx++)
    {
        const double *initial_data = &el_m_input[electron_idx * 5];

        double sort;
        double initial_t;
        double final_p[3];

        single_boris_process(
            initial_data,

            el_sim_duration,
            delta_t,

            beam_radius,
            tau,
            a_0_value,
            ell_value,
            r_or_l,
            x_r,

            &sort,
            &initial_t,
            final_p
        );

        sorts_arr[electron_idx] = sort;
        initial_t_arr[electron_idx] = initial_t;

        p_x_arr[electron_idx] = final_p[0];
        p_y_arr[electron_idx] = final_p[1];
        p_z_arr[electron_idx] = final_p[2];
    }
}
