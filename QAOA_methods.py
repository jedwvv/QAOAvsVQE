import multiprocessing as mp
import numpy as np
from qiskit import QuantumCircuit
from generate_qubos import solve_classically, arr_to_str
import QAOAEx

def CustomQAOA(operator, quantum_instance, optimizer, reps, **kwargs):
    
    initial_state = None if "initial_state" not in kwargs else kwargs["initial_state"]
    mixer = None if "mixer" not in kwargs else kwargs["mixer"]
    construct_circ = False if "construct_circ" not in kwargs else kwargs["construct_circ"]
    fourier_parametrise = False if "fourier_parametrise" not in kwargs else kwargs["fourier_parametrise"]
    qubo = None if "fourier_parametrise" not in kwargs else kwargs["qubo"]

    qaoa_instance = QAOAEx.QAOACustom(quantum_instance = quantum_instance,
                                        reps = reps,
                                        force_shots = False,
                                        optimizer = optimizer,
                                        qaoa_name = "example_qaoa",
                                        initial_state = initial_state,
                                        mixer = mixer,
                                        include_custom = False,
                                        max_evals_grouped = 1
                                        )
    if fourier_parametrise:
        qaoa_instance.set_parameterise_point_for_energy_evaluation(QAOAEx.convert_from_fourier_point)
    bounds = [(-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2)]*reps
    
    if "list_points" in kwargs:
        list_points = kwargs["list_points"]
        list_results = []

        for point in list_points:
            result = qaoa_instance.solve(operator, point, bounds=bounds)
            qc = qaoa_instance.get_optimal_circuit() if construct_circ else None
            list_results.append( (result, qc) )

        qaoa_results, qc = min(list_results, key=lambda x: x[0].eigenvalue)
    else:
        initial_point = kwargs["initial_point"] if "initial_points" in kwargs\
                                                else [ np.pi * (np.random.rand() - 0.5) for _ in range(2*reps) ]

        qaoa_results = qaoa_instance.solve(operator, initial_point, bounds=bounds)
        qc = qaoa_instance.get_optimal_circuit() if construct_circ else None
    
    if fourier_parametrise and qubo:
        optimal_point = qaoa_results.optimal_point
        state = qaoa_instance.calculate_statevector_at_point(operator = operator, point = QAOAEx.convert_from_fourier_point(optimal_point, len(optimal_point)))
        qaoa_results.eigenstate = qaoa_instance.eigenvector_to_solutions(state, quadratic_program=qubo)


    return qaoa_results, qc

def generate_points(point, no_perturb, penalty):
    points = [point.copy() for _ in range(no_perturb+1)]
    for r in range(1,no_perturb+1):
        for i in range(len(point)):
            random_sign = np.random.choice([-1,1])
            if point[i] == 0:
                points[r][i] += penalty*0.05*random_sign
            else:
                noise = np.random.normal(0, (point[i]/np.pi)**2)
                points[r][i] += penalty*noise*random_sign
            if points[r][i] < -np.pi/2:
                points[r][i] = -np.pi/2+0.01
            if points[r][i] > np.pi/2:
                points[r][i] = np.pi/2-0.01

    return points

def eval_cost(x_str_arr, qubo_problem, no_qubits):
    if isinstance(x_str_arr, np.ndarray):
        cost = qubo_problem.objective.evaluate([int(x_str_arr[i]) for i in range(no_qubits)])
    else:
        cost = qubo_problem.objective.evaluate([int(x_str_arr[no_qubits-1-i]) \
                                                for i in range(no_qubits)]
                                            )
    return x_str_arr, cost

def get_costs(qubo):
    """[summary]

    Args:
        qubo ([type]): [description]
        no_qubits ([type]): [description]

    Returns:
        [type]: [description]
    """
    no_qubits = qubo.get_num_binary_vars()
    cpus = mp.cpu_count()
    pool = mp.Pool(cpus-1)
    params = ( ('0'*(no_qubits-len('{0:b}'.format(k)))+'{0:b}'.format(k),
            qubo,
            no_qubits
            ) for k in range(2**no_qubits))
    values = pool.starmap(eval_cost, params)
    values = dict(values)
    pool.close()
    pool.join()
    sort_values = sorted(values.items(), key=lambda x: x[1])
    return sort_values

def find_all_ground_states(qubo):
    classical_result = solve_classically(qubo)
    x_arr = classical_result.x
    opt_value = classical_result.fval
    x_str = arr_to_str(x_arr)
    sort_values = get_costs(qubo)
    ground_energy = sort_values[0][1]
    x_s = [x_str]
    for i in range(1,10):
        if np.round(sort_values[i][1], 4) == np.round(ground_energy, 4):
            print("Other ground state(s) found: '{}'".format(sort_values[i][0]))
            x_s.append(sort_values[i][0])
    
    return x_s, opt_value, classical_result


def count_coupling_terms(operator):
    """[summary]

    Args:
        operator ([type]): [description]
    """
    count = 0
    for op_i in operator.to_pauli_op().oplist:
        op_str = str(op_i.primitive)
        if op_str.count('Z') == 2:
            count += 1
    return count

def interp_point(optimal_point):
    """Method to interpolate to next point from the optimal point found from the previous layer   
    
    Args:
        optimal_point (np.array): Optimal point from previous layer
    
    Returns:
        point (list): the informed next point
    """
    optimal_point = list(optimal_point)
    p = int(len(optimal_point)/2)
    gammas = [0]+optimal_point[0:p]+[0]
    betas = [0]+optimal_point[p:2*p]+[0]
    interp_gammas = [0]+gammas
    interp_betas = [0]+betas    
    for i in range(1,p+2):
        interp_gammas[i] = gammas[i-1]*(i-1)/p+gammas[i]*(p+1-i)/p
        interp_betas[i] = betas[i-1]*(i-1)/p+betas[i]*(p+1-i)/p
        if interp_gammas[i] < -np.pi/2:
            interp_gammas[i] = -np.pi/2+0.01
        if interp_betas[i] < -np.pi/2:
            interp_betas[i] = -np.pi/2+0.01
        if interp_gammas[i] > np.pi/2:
            interp_gammas[i] = np.pi/2-0.01
        if interp_betas[i] > np.pi/2:
            interp_betas[i] = np.pi/2-0.01
    
    point = interp_gammas[1:p+2] + interp_betas[1:p+2]

    return point

def construct_initial_state(no_routes: int, no_cars: int):
    """[summary]

    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    R = no_routes
    N = no_cars
    w_one = QuantumCircuit(R)
    for r in range(0,R-1):
        if r == 0:
            w_one.ry(2*np.arccos(1/np.sqrt(R-r)),r)
        elif r != R-1:
            w_one.cry(2*np.arccos(1/np.sqrt(R-r)), r-1, r)
    for r in range(1,R):
        w_one.cx(R-r-1,R-r)
    w_one.x(0)

    #Tensor product for N cars
    initial_state = w_one.copy()
    for _ in range(0,N-1):
        initial_state = initial_state.tensor(w_one)

    initial_state.name = "U_init"

    return initial_state

def n_qbit_mixer(initial_state: QuantumCircuit):
    from qiskit.circuit.parameter import Parameter
    no_qubits = initial_state.num_qubits
    t = Parameter('t')
    # print(initial_state.draw())
    mixer = QuantumCircuit(no_qubits)
    # print(initial_state.inverse().draw())
    mixer.append(initial_state.inverse(), range(no_qubits))
    # mixer.barrier()
    mixer.rz(2*t, range(no_qubits))
    # mixer.barrier()
    mixer.append(initial_state, range(no_qubits))
    # from qiskit.compiler import transpile
    # test = transpile(mixer, basis_gates=["id", "rx", "ry", "rz", "cx"])
    # print(test.draw())
    return mixer