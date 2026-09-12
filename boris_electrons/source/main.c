#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "boris.h"

#define ATOMIC_FREQUENCY_UNIT 4.1e16
#define ATOMIC_INTENSITY_UNIT 3.5e16
#define SPEED_OF_LIGHT 2.997925e8
#define PI 3.14159265358979323846


int main(void)
{
    /* -------------------------------------------------
       1. Открываем бинарный файл с входными данными
       ------------------------------------------------- */

    FILE *file;

    file = fopen("electron_motion_input.bin", "rb");

    if (file == NULL)
    {
        printf("Cannot open input file\n");
        return 1;
    }


    /* -------------------------------------------------
       2. Определяем размер файла
       ------------------------------------------------- */

    fseek(file, 0, SEEK_END);

    long file_size = ftell(file);

    rewind(file);


    /* -------------------------------------------------
       3. Определяем количество электронов

       Один электрон = 5 double:
       [t, sort, x, y, z]
       ------------------------------------------------- */

    int number_of_electrons =
        file_size / (5 * sizeof(double));

    printf("Number of electrons: %d\n",
           number_of_electrons);


    /* -------------------------------------------------
       4. Выделяем память под входные данные
       ------------------------------------------------- */

    double *el_m_input =
        malloc(number_of_electrons * 5 * sizeof(double));

    if (el_m_input == NULL)
    {
        printf("Memory allocation failed\n");
        fclose(file);
        return 1;
    }


    /* -------------------------------------------------
       5. Читаем входной бинарный файл
       ------------------------------------------------- */

    size_t elements_read = fread(
        el_m_input,
        sizeof(double),
        number_of_electrons * 5,
        file
    );

    if (elements_read !=
        (size_t)(number_of_electrons * 5))
    {
        printf("Error reading input file\n");

        free(el_m_input);
        fclose(file);

        return 1;
    }

    fclose(file);


    /* -------------------------------------------------
       6. Выделяем память под результаты
       ------------------------------------------------- */

    double *sorts_arr =
        malloc(number_of_electrons * sizeof(double));

    double *initial_t_arr =
        malloc(number_of_electrons * sizeof(double));

    double *p_x_arr =
        malloc(number_of_electrons * sizeof(double));

    double *p_y_arr =
        malloc(number_of_electrons * sizeof(double));

    double *p_z_arr =
        malloc(number_of_electrons * sizeof(double));


    if (sorts_arr == NULL ||
        initial_t_arr == NULL ||
        p_x_arr == NULL ||
        p_y_arr == NULL ||
        p_z_arr == NULL)
    {
        printf("Output memory allocation failed\n");

        free(el_m_input);

        free(sorts_arr);
        free(initial_t_arr);
        free(p_x_arr);
        free(p_y_arr);
        free(p_z_arr);

        return 1;
    }


    /* -------------------------------------------------
       7. Параметры расчёта

       Пока задаём их непосредственно здесь.
       Позже можно вынести в отдельный config.
       ------------------------------------------------- */

    double intensity = 2.15e22;
    double w_0 = 2.654;
    double tau = 44.0;
    double t_electron = 1.0e4;
    double delta_t = 0.001;

    double eps = 0.;
    double right_or_left = 1.0;
    double wavelength = 0.8;
    double x_r = PI * w_0 * w_0;


    double frequency = 2 * PI * SPEED_OF_LIGHT / (wavelength * 1.0e-6);
    double atomic_frequency = frequency / ATOMIC_FREQUENCY_UNIT;

    double atomic_field = sqrt(intensity /
                               (ATOMIC_INTENSITY_UNIT * (1.0 + eps * eps)));

    double a_0 = atomic_field / (137.0 * atomic_frequency);

    /* -------------------------------------------------
       8. Запускаем расчёт
       ------------------------------------------------- */

    printf("Starting Boris simulation...\n");

    boris_parallel_simulation(
        el_m_input,
        number_of_electrons,

        t_electron,
        delta_t,

        w_0,
        tau,
        a_0,
        eps,
        right_or_left,
        x_r,

        sorts_arr,
        initial_t_arr,
        p_x_arr,
        p_y_arr,
        p_z_arr
    );

    printf("Simulation finished.\n");


    /* -------------------------------------------------
       9. Открываем файл для результатов
       ------------------------------------------------- */

    FILE *output_file;

    output_file =
        fopen("electron_motion_output.bin", "wb");

    if (output_file == NULL)
    {
        printf("Cannot open output file\n");

        free(el_m_input);

        free(sorts_arr);
        free(initial_t_arr);
        free(p_x_arr);
        free(p_y_arr);
        free(p_z_arr);

        return 1;
    }


    /* -------------------------------------------------
       10. Записываем пять массивов последовательно

       Формат файла:

       [sort_0 ... sort_N]
       [t_0    ... t_N]
       [px_0   ... px_N]
       [py_0   ... py_N]
       [pz_0   ... pz_N]
       ------------------------------------------------- */

    fwrite(
        sorts_arr,
        sizeof(double),
        number_of_electrons,
        output_file
    );

    fwrite(
        initial_t_arr,
        sizeof(double),
        number_of_electrons,
        output_file
    );

    fwrite(
        p_x_arr,
        sizeof(double),
        number_of_electrons,
        output_file
    );

    fwrite(
        p_y_arr,
        sizeof(double),
        number_of_electrons,
        output_file
    );

    fwrite(
        p_z_arr,
        sizeof(double),
        number_of_electrons,
        output_file
    );


    fclose(output_file);


    /* -------------------------------------------------
       11. Освобождаем память
       ------------------------------------------------- */

    free(el_m_input);

    free(sorts_arr);
    free(initial_t_arr);
    free(p_x_arr);
    free(p_y_arr);
    free(p_z_arr);


    printf("Results saved to electron_motion_output.bin\n");

    return 0;
}
