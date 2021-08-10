"""[summary]

Raises:
    TypeError: [description]

Returns:
    [type]: [description]
"""
from time import time
import warnings
import multiprocessing as mp
import numpy as np
import pickle as pkl
from qiskit import Aer, QuantumCircuit
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.quantum_info import Statevector
import QAOAEx
from parser_all import parse
from generate_qubos import solve_classically, arr_to_str, visualise_solution
from generate_problem import import_map
from classical_optimizers import NLOPT_Optimizer

warnings.filterwarnings('ignore')

def main(args = None):
    """[summary]

    Args:
        raw_args ([type], optional): [description]. Defaults to None.
    """
    start = time()
    if args == None:
        args = parse()
    
    prob_s_s = []
    qubo_no = args["no_samples"]
    print("__"*50, "\nQUBO NO: {}\n".format(qubo_no), "__"*50)
    
    #Load generated qubo_no
    with open('qubos_{}_car_{}_routes/qubo_{}.pkl'.format(args["no_cars"], args["no_routes"], qubo_no), 'rb') as f:
        qubo, max_coeff, operator, offset, routes = pkl.load(f)
    print(operator)

    classical_result = solve_classically(qubo)
    print(classical_result)
    x_arr = classical_result.x
    x_str = arr_to_str(x_arr)
    no_qubits = len(x_arr)

    sort_values = get_costs(qubo, no_qubits)

    print("_"*50)
    up_to = 27
    print("{} lowest states:".format(up_to))
    avg = 0
    for i in range(up_to):
        print(sort_values[i])
        avg += sort_values[i][1]
    print("_"*50)
    print("Avg: {}".format(avg/up_to))
    
    ground_energy = sort_values[0][1]
    x_s = [x_str]
    for i in range(1,10):
        if np.round(sort_values[i][1], 4) == np.round(ground_energy, 4):
            print("Other ground state(s) found: '{}'".format(sort_values[i][0]))
            x_s.append(sort_values[i][0])

    # Visualise
    if args["visual"]:
        graph = import_map('melbourne.pkl')
        visualise_solution(graph, routes)

    # Solve QAOA from QUBO with valid solution
    no_couplings = count_coupling_terms(operator)
    print("Number of couplings: {}".format(no_couplings))
    print("Solving with QAOA...")
    no_shots = 10000
    backend = Aer.get_backend('statevector_simulator')
    quantum_instance = QuantumInstance(backend, shots = no_shots)
    optimizer_method = "LN_SBPLX"
    optimizer = NLOPT_Optimizer(method = optimizer_method)
    print("_"*50,"\n"+optimizer.__class__.__name__)
    print("_"*50)


    quantum_instance = QuantumInstance(backend)
    prob_s_s = []
    initial_state = construct_initial_state(args["no_routes"], args["no_cars"])
    mixer = n_qbit_mixer(initial_state)
    next_fourier_point, next_fourier_point_B = [0,0], [0,0] #Not used for p=1 then gets updated for p>1.
    for p in range(1, args["p_max"]+1):
        print("p = {}".format(p))
        if p==1:
            points = [[0.75,0]] \
                # + [[ np.pi*(2*np.random.rand() - 1) for _ in range(2) ] for _ in range(args["no_restarts"])]
            draw_circuit = True
        else:
            penalty = 0.6
            points = [0.75*np.append([], [[(i+1)/p, 1-(i+1)/p] for i in range(p)])]
            print(points) \
                # + generate_points(next_fourier_point_B, 10, penalty)
            draw_circuit = False
        #empty lists to save following results to choose best result
        results = []
        exp_vals = []
        for r in range(len(points)):
            point = points[r]
            if np.amax(np.abs(point)) < np.pi/2: 
                qaoa_results, optimal_circ = Fourier_QAOA(operator,
                                                            quantum_instance,
                                                            optimizer,
                                                            reps = p,
                                                            initial_fourier_point=points[r],
                                                            initial_state = initial_state,
                                                            mixer = mixer,
                                                            construct_circ=draw_circuit
                                                            )
                if r == 0:
                    next_fourier_point = np.array(qaoa_results.optimal_point)
                    next_fourier_point = QAOAEx.convert_from_fourier_point(next_fourier_point, 2*p+2)
                    next_fourier_point = QAOAEx.convert_to_fourier_point(next_fourier_point, 2*p+2)           
                exp_val = qaoa_results.eigenvalue * max_coeff + offset
                exp_vals.append(exp_val)
                prob_s = 0
                for string in x_s:
                    prob_s += qaoa_results.eigenstate[string] if string in qaoa_results.eigenstate else 0
                results.append((qaoa_results, optimal_circ, prob_s))
                print("Point_no: {}, Exp_val: {}, Prob_s: {}".format(r, exp_val, prob_s))
            else:
                print("Point_no: {}, was skipped because it is outside of bounds".format(r))
        minim_index = np.argmin(exp_vals)
        optimal_qaoa_result, optimal_circ, optimal_prob_s = results[minim_index]
        # if draw_circuit:
        #     print(optimal_circ.draw())
        minim_exp_val = exp_vals[minim_index]
        print("Minimum: {}, prob_s: {}".format(minim_exp_val, optimal_prob_s))
        prob_s_s.append(optimal_prob_s)
        next_fourier_point_B = np.array(optimal_qaoa_result.optimal_point)
        print("Optimal_point: {}".format(next_fourier_point_B))
        next_fourier_point_B = QAOAEx.convert_from_fourier_point(next_fourier_point_B, 2*p+2)
        next_fourier_point_B = QAOAEx.convert_to_fourier_point(next_fourier_point_B, 2*p+2)

    print(prob_s_s)

    with open('results/{}cars{}routes_qubo{}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"]), 'w') as f:
        np.savetxt(f, prob_s_s, delimiter=',')
    finish = time()
    print("Time Taken: {}".format(finish - start))


def Fourier_QAOA(operator, quantum_instance, optimizer, reps, initial_fourier_point, initial_state=None, mixer = None, construct_circ = False):
    qaoa_instance = QAOAEx.QAOACustom(quantum_instance = quantum_instance,
                                        reps = reps,
                                        force_shots = False,
                                        optimizer = optimizer,
                                        qaoa_name = "example_qaoa",
                                        initial_state = initial_state,
                                        mixer = mixer,
                                        include_custom = False,
                                        max_evals_grouped = 2
                                        )
    # qaoa_instance.set_parameterise_point_for_energy_evaluation(QAOAEx.convert_from_fourier_point)
    # bounds = [None, (-np.pi/2)]*len(initial_fourier_point)
    bounds = [(-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2)]*reps
    qaoa_results = qaoa_instance.solve(operator, initial_fourier_point)
    if not isinstance(qaoa_results.eigenstate, dict):
        qaoa_results.eigenstate = Statevector(qaoa_results.eigenstate).probabilities_dict()
    if construct_circ:
        qc = qaoa_instance.get_optimal_circuit()
    else:
        qc = None
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
    return points

def eval_cost(x_str_arr, qubo_problem, no_qubits):
    """[summary]

    Args:
        x ([type]): [description]
        qubo_problem ([type]): [description]
        no_qubits ([type]): [description]
        max_coeff ([type]): [description]
        offset ([type]): [description]

    Returns:
        [type]: [description]
    """
    if isinstance(x_str_arr, np.ndarray):
        cost = qubo_problem.objective.evaluate([int(x_str_arr[i]) for i in range(no_qubits)])
    else:
        cost = qubo_problem.objective.evaluate([int(x_str_arr[no_qubits-1-i]) \
                                                for i in range(no_qubits)]
                                            )
    return x_str_arr, cost

def get_costs(qubo, no_qubits):
    """[summary]

    Args:
        qubo ([type]): [description]
        no_qubits ([type]): [description]

    Returns:
        [type]: [description]
    """
    # values = ['{}'.format()]
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

main()