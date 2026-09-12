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

#endif
