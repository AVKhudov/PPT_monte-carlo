#include<math.h>
#include<omp.h>
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

    double spatial_part = 1.0 / radius * exp(-r_squared /
                                             pow(beam_radius * radius, 2.0));
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
    /*
     * Координаты частицы.
     */
    double x = radius_vector[0];
    double y = radius_vector[1];
    double z = radius_vector[2];

    /*
     * Электрическое и магнитное поля.
     */
    double electric_field[3];
    double magnetic_field[3];

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

    /*
     * Проверяем, равно ли поле нулю.
     *
     * В Python здесь:
     *
     * if np.all(electric_field == 0.) and np.all(magnetic_field == 0.):
     */
    if (electric_field[0] == 0.0 &&
        electric_field[1] == 0.0 &&
        electric_field[2] == 0.0 &&

        magnetic_field[0] == 0.0 &&
        magnetic_field[1] == 0.0 &&
        magnetic_field[2] == 0.0)
    {
        /*
         * При отсутствии поля импульс не изменяется.
         *
         * gamma = sqrt(1 + p^2)
         */
        double p_squared =
            p_half[0] * p_half[0] +
            p_half[1] * p_half[1] +
            p_half[2] * p_half[2];

        double gamma_half = sqrt(1.0 + p_squared);

        /*
         * v = p / gamma
         */
        double v_half[3];

        v_half[0] = p_half[0] / gamma_half;
        v_half[1] = p_half[1] / gamma_half;
        v_half[2] = p_half[2] / gamma_half;

        /*
         * r_new = r + v * delta_t
         */
        radius_vector_new[0] =
            radius_vector[0] + v_half[0] * delta_t;

        radius_vector_new[1] =
            radius_vector[1] + v_half[1] * delta_t;

        radius_vector_new[2] =
            radius_vector[2] + v_half[2] * delta_t;

        /*
         * p_new = p
         */
        p_new_half[0] = p_half[0];
        p_new_half[1] = p_half[1];
        p_new_half[2] = p_half[2];

        return;
    }

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

    t_vector[0] =
        0.5 * magnetic_field[0] * delta_t / gamma_minus;

    t_vector[1] =
        0.5 * magnetic_field[1] * delta_t / gamma_minus;

    t_vector[2] =
        0.5 * magnetic_field[2] * delta_t / gamma_minus;

    /*
     * t^2
     */
    double t_squared =
        t_vector[0] * t_vector[0] +
        t_vector[1] * t_vector[1] +
        t_vector[2] * t_vector[2];

    /*
     * s = 2*t / (1 + t^2)
     */
    double s_vector[3];

    s_vector[0] =
        2.0 * t_vector[0] / (1.0 + t_squared);

    s_vector[1] =
        2.0 * t_vector[1] / (1.0 + t_squared);

    s_vector[2] =
        2.0 * t_vector[2] / (1.0 + t_squared);

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

    p_new_half[0] =
        p_plus[0] + 0.5 * electric_field[0] * delta_t;

    p_new_half[1] =
        p_plus[1] + 0.5 * electric_field[1] * delta_t;

    p_new_half[2] =
        p_plus[2] + 0.5 * electric_field[2] * delta_t;

    /*
     * gamma_new
     */
    double p_new_squared =
        p_new_half[0] * p_new_half[0] +
        p_new_half[1] * p_new_half[1] +
        p_new_half[2] * p_new_half[2];

    double gamma_new =
        sqrt(1.0 + p_new_squared);

    /*
     * v_new_half = p_new_half / gamma_new
     */
    double v_new_half[3];

    v_new_half[0] =
        p_new_half[0] / gamma_new;

    v_new_half[1] =
        p_new_half[1] / gamma_new;

    v_new_half[2] =
        p_new_half[2] / gamma_new;

    /*
     * ---------------------------------------------------------
     * 6. Position update
     *
     * r_new = r + v_new_half * delta_t
     * ---------------------------------------------------------
     */

    radius_vector_new[0] =
        radius_vector[0] + v_new_half[0] * delta_t;

    radius_vector_new[1] =
        radius_vector[1] + v_new_half[1] * delta_t;

    radius_vector_new[2] =
        radius_vector[2] + v_new_half[2] * delta_t;
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

        /*
         * Обновляем координату.
         */
        current_r[0] = new_r[0];
        current_r[1] = new_r[1];
        current_r[2] = new_r[2];

        /*
         * Обновляем импульс.
         */
        current_p[0] = new_p[0];
        current_p[1] = new_p[1];
        current_p[2] = new_p[2];

        /*
         * Переходим к следующему моменту времени.
         */
        current_time += delta_t;
    }

    /*
     * Конечный импульс.
     */
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
        const double *initial_data =
            &el_m_input[electron_idx * 5];

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

static void func(
    double x,
    double y,
    double z
)
{

}
