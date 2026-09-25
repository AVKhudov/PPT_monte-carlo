import ctypes
import numpy as np

DLL_PATH = r"C:\Users\Dns\Desktop\НИРС 4 сезон\монте карла\boris_electrons\boris.dll"

dll = ctypes.CDLL(DLL_PATH)

dll.ionization_simulation.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    ctypes.c_int,

    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,

    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,

    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),

    ctypes.POINTER(ctypes.c_int)
]

dll.ionization_simulation.restype = ctypes.POINTER(ctypes.c_double)


dll.free_memory.argtypes = [ctypes.POINTER(ctypes.c_double)]
dll.free_memory.restype = None


def ionization_simulation(
    placed_cells,
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
    n_star_array
):
    placed_cells = np.ascontiguousarray(
        placed_cells, dtype=np.float64
    )

    ionization_order_array = np.ascontiguousarray(
        ionization_order_array, dtype=np.int32
    )

    ionization_potentials = np.ascontiguousarray(
        ionization_potentials, dtype=np.float64
    )

    c_n_l_array = np.ascontiguousarray(
        c_n_l_array, dtype=np.float64
    )

    b_l_m_array = np.ascontiguousarray(
        b_l_m_array, dtype=np.float64
    )

    n_star_array = np.ascontiguousarray(
        n_star_array, dtype=np.float64
    )

    number_of_atoms = placed_cells.shape[0]

    number_of_results = ctypes.c_int()

    result_ptr = dll.ionization_simulation(
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

        ctypes.byref(number_of_results)
    )

    if not result_ptr:
        raise MemoryError("C: malloc failed")

    number_of_results = number_of_results.value

    result = np.ctypeslib.as_array(
        result_ptr,
        shape=(number_of_results * 5,)
    ).reshape(number_of_results, 5).copy()

    dll.free_memory(result_ptr)

    return result


dll.boris_parallel_simulation.argtypes = [
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

    dll.boris_parallel_simulation(
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
