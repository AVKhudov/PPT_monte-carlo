import ctypes
import numpy as np

DLL_PATH = r"C:\Users\Dns\Desktop\НИРС 4 сезон\монте карла\boris_electrons\boris.dll"

boris = ctypes.CDLL(DLL_PATH)

boris.boris_parallel_simulation.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,

    ctypes.c_double,
    ctypes.c_double,

    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,

    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double)
]


def run_boris(
        el_m_input,

        el_sim_duration,
        delta_t,

        beam_radius,
        tau,
        a_0_value,
        ell_value,
        r_or_l,
        x_r
):
    el_m_input = np.ascontiguousarray(
        el_m_input,
        dtype=np.float64
    )

    number_of_electrons = len(el_m_input)

    sorts_array = np.empty(
        number_of_electrons,
        dtype=np.float64
    )

    initial_times_array = np.empty(
        number_of_electrons,
        dtype=np.float64
    )

    p_x_array = np.empty(
        number_of_electrons,
        dtype=np.float64
    )

    p_y_array = np.empty(
        number_of_electrons,
        dtype=np.float64
    )

    p_z_array = np.empty(
        number_of_electrons,
        dtype=np.float64
    )

    boris.boris_parallel_simulation(
        el_m_input.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        ),

        number_of_electrons,

        el_sim_duration,
        delta_t,
        beam_radius,
        tau,
        a_0_value,
        ell_value,
        r_or_l,
        x_r,

        sorts_array.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        ),

        initial_times_array.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        ),

        p_x_array.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        ),

        p_y_array.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        ),

        p_z_array.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        ),
    )

    return (
        sorts_array,
        initial_times_array,
        p_x_array,
        p_y_array,
        p_z_array
    )
