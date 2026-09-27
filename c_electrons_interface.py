import ctypes
import time as tm
import numpy as np

DLL_PATH = r"C:\Users\Dns\Desktop\НИРС 4 сезон\монте карла\boris_electrons\boris_ext_ion_time.dll"

dll = ctypes.CDLL(DLL_PATH)

dll.ionization_simulation.argtypes = [
    # placed_cells
    np.ctypeslib.ndpointer(
        dtype=np.float64,
        ndim=2,
        flags="C_CONTIGUOUS"
    ),

    # number_of_atoms
    ctypes.c_int,

    # t_0, delta_t
    ctypes.c_double,
    ctypes.c_double,

    # max_ion, start_charge_number
    ctypes.c_int,
    ctypes.c_int,

    # beam_radius, tau, atomic_field, ell_value, x_r
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,

    # ionization_order_array
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"),

    # ionization_potentials
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),

    # c_n_l_array, b_l_m_array, n_start_array
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),

    # number_of_results
    ctypes.POINTER(ctypes.c_int),

    # fully_ionized_states_indices
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),

    # number_of_fully_ionized_states
    ctypes.POINTER(ctypes.c_int),

    # ion_times
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),

    # initial_seed
    ctypes.c_uint32
]

dll.ionization_simulation.restype = ctypes.POINTER(ctypes.c_double)

dll.free_memory.argtypes = [ctypes.POINTER(ctypes.c_double)]
dll.free_memory.restype = None

dll.free_int_memory.argtypes = [ctypes.POINTER(ctypes.c_int)]
dll.free_int_memory.restype = None


def ionization_simulation(
    placed_cells,

    t_0,
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
    seed = int(tm.time_ns() & 0xFFFFFFFF)  # сид для рандома

    number_of_atoms = placed_cells.shape[0]

    number_of_electrons = ctypes.c_int()
    number_of_fully_ionized_states = ctypes.c_int()

    fully_ionized_states_indices_ptr = ctypes.POINTER(ctypes.c_int)()

    ion_times_ptr = ctypes.POINTER(ctypes.c_double)()

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

    print('Ионизация...')
    t_start = tm.perf_counter()

    result_ptr = dll.ionization_simulation(
        placed_cells,
        number_of_atoms,

        t_0,
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

        ctypes.byref(number_of_electrons),
        ctypes.byref(fully_ionized_states_indices_ptr),
        ctypes.byref(number_of_fully_ionized_states),
        ctypes.byref(ion_times_ptr),

        seed
    )

    if not result_ptr:
        raise MemoryError("C: malloc failed")

    t_final = tm.perf_counter()
    print(f'Ионизация завершена, время: {t_final - t_start} с')

    number_of_electrons_value = number_of_electrons.value

    electron_motion_input = np.ctypeslib.as_array(
        result_ptr,
        shape=(number_of_electrons_value * 5,)
    ).reshape(number_of_electrons_value, 5).copy()

    dll.free_memory(result_ptr)

    number_of_fully_ionized_states_value = (
        number_of_fully_ionized_states.value
    )

    if number_of_fully_ionized_states_value > 0:
        fully_ionized_states_indices = np.ctypeslib.as_array(
            fully_ionized_states_indices_ptr,
            shape=(number_of_fully_ionized_states_value,)
        ).copy()

        dll.free_int_memory(
            fully_ionized_states_indices_ptr
        )
    else:
        fully_ionized_states_indices = np.empty(
            0,
            dtype=np.int32
        )

    number_of_ion_times = (number_of_atoms * (max_ion - start_charge_number))

    ion_times = np.ctypeslib.as_array(
        ion_times_ptr,
        shape=(number_of_ion_times,)
    ).reshape(number_of_atoms, max_ion - start_charge_number).copy()

    dll.free_memory(ion_times_ptr)

    return (
        electron_motion_input,
        number_of_electrons_value,
        number_of_fully_ionized_states_value,
        fully_ionized_states_indices,
        ion_times
    )


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
