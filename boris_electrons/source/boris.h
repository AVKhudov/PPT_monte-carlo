#ifndef BORIS_H
#define BORIS_H

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
);

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
);

void free_memory(double *ptr);

#endif
