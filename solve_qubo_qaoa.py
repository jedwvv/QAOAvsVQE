"""[summary]

Raises:
    TypeError: [description]

Returns:
    [type]: [description]
"""
from sys import float_info
from time import time
import warnings
import argparse
import multiprocessing as mp
import numpy as np
import osmnx as ox
import pickle as pkl
from qiskit import Aer
from qiskit.algorithms.minimum_eigen_solvers import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import (
                                        ADAM,
                                        CG,
                                        COBYLA,
                                        L_BFGS_B,
                                        NELDER_MEAD,
                                        NFT,
                                        POWELL,
                                        SLSQP,
                                        SPSA,
                                        TNC,
                                        P_BFGS,
                                        BOBYQA
                                        )
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.utils.quantum_instance import QuantumInstance
# from qiskit.opflow.primitive_ops import PauliSumOp
from generate_problem import import_map
from generate_qubo import main as generate_qubo
import QAOAEx


warnings.filterwarnings('ignore')

def main(raw_args = None):
    """[summary]

    Args:
        raw_args ([type], optional): [description]. Defaults to None.
    """
    start = time()
    args = parse(raw_args)
    qubo_prob_s_s = []
    for qubo_no in range(args["no_samples"]):
        print("__"*50, "\nQUBO NO: {}\n".format(qubo_no), "__"*50)
        # #Generate and save valid qubos
        # qubo, max_coeff, operator, offset, routes = generate_valid_qubo(args)
        # with open('qubos/qubo_{}.pkl'.format(qubo_no), 'wb') as f:
        #     pkl.dump([qubo, max_coeff, operator, offset, routes],f)
        
        #Load generated qubos
        with open('qubos/qubo_{}.pkl'.format(qubo_no), 'rb') as f:
            qubo, max_coeff, operator, offset, routes = pkl.load(f)

        classical_result = solve_classically(qubo)   
        x_arr = classical_result.x
        x_str = arr_to_str(x_arr)
        no_qubits = len(x_arr)

        sort_values = get_costs(qubo, no_qubits)

        print("_"*50)
        print("10 lowest states:")
        for i in range(10):
            print(sort_values[i])
        print("_"*50)
        ground_energy = sort_values[0][1]
        x_s = [x_str]
        for i in range(1,10):
            if np.round(sort_values[i][1], 4) == np.round(ground_energy, 4):
                print("Other ground state(s) found: '{}'".format(sort_values[i][0]))
                x_s.append(sort_values[i][0])

        #Visualise
        if args["visual"]:
            graph = import_map('melbourne.pkl')
            visualise_solution(graph, routes)

        #Solve QAOA from QUBO with valid solution
        no_couplings = count_coupling_terms(operator)
        print("Number of couplings: {}".format(no_couplings))
        print("Solving with QAOA...")
        no_shots = 10000
        backend = Aer.get_backend('qasm_simulator', shots = no_shots)
        
        #Optimizers available from QISKIT - as chosen from parsed argument.
        optimizers = {"ADAM":ADAM(),
        "CG":CG(),
        "COBYLA":COBYLA(),
        "L_BFGS_B":L_BFGS_B(),
        "NELDER_MEAD":NELDER_MEAD(),
        "NFT":NFT(),
        "POWELL":POWELL(),
        "SLSQP":SLSQP(),
        "SPSA":SPSA(),
        "TNC":TNC(),
        "P_BFGS": P_BFGS(),
        "BOBYQA": BOBYQA()}
        optimizer = optimizers[args["optimizer"]]
        print("_"*50,"\n"+optimizer.__class__.__name__)
        print("_"*50)

        quantum_instance = QuantumInstance(backend, shots=no_shots)


        # for p in range(1, args["p_max"]+1):
            # print("p = {}".format(p))
        minim_exp_val = float_info.max
        prob_s_s = []
        #P = 1
        p=1
        print("p=1")
        for r in range(args["no_restarts"]):
            print("Restart No: {}".format(r))
            initial_fourier_point = [ 1.99*np.pi*np.random.rand() - np.pi for _ in range(2) ]
            qaoa_results, _ = Fourier_QAOA(operator, quantum_instance, optimizer, p, initial_fourier_point)
            exp_val = qaoa_results.eigenvalue*max_coeff + offset
            print("Exp_val: {}".format(exp_val))
            prob_s = 0
            for string in x_s:
                prob_s += qaoa_results.eigenstate[string] if string in qaoa_results.eigenstate else 0
            print("Prob_s: {} \n".format(prob_s))
            # print("_"*50)
            if exp_val < minim_exp_val:
                minim_exp_val = exp_val
                optimal_qaoa_result = qaoa_results
                optimal_prob_s = prob_s
        print("Minimum Exp Val: ", minim_exp_val)
        print("Prob_s for this: ", optimal_prob_s)
        prob_s_s.append(optimal_prob_s)

        # print("RESULT: ", optimal_qaoa_result)
        optimal_fourier_point = np.array(optimal_qaoa_result.optimal_point)
        optimal_fourier_point_B = optimal_fourier_point.copy()

        for p in range(2, args["p_max"]+1):
            print("p={}".format(p))
            if args["optimizer"] == 'SLSQP':
                optimizer.set_options(maxiter = 100*p)
            elif args["optimizer"] == 'L_BFGS_B' or 'P_BFGS':
                optimizer.set_options(maxfun = 1000*p)
            print("Optimizer options: {}".format(optimizer.setting))
            next_fourier_point = np.zeros(shape = (2*p,))
            next_fourier_point_B = np.zeros(shape = (2*p,))

            next_fourier_point[0:p-1] = optimal_fourier_point[0:p-1]
            next_fourier_point[p:2*p-1] = optimal_fourier_point[p-1:2*p-2]

            next_fourier_point_B[0:p-1] = optimal_fourier_point_B[0:p-1]
            next_fourier_point_B[p:2*p-1] = optimal_fourier_point_B[p-1:2*p-2]

            perturbed_points = generate_points(next_fourier_point_B, 10) if p<= 5 \
                                else generate_points(next_fourier_point_B, 20)


            qaoa_results, _ = Fourier_QAOA(operator, quantum_instance, optimizer, p, next_fourier_point)
            optimal_fourier_point = np.array(qaoa_results.optimal_point)
            optimal_fourier_point_B = optimal_fourier_point.copy()

            minim_exp_val = qaoa_results.eigenvalue*max_coeff + offset
            minim_exp_val_B = minim_exp_val

            optimal_prob_s = 0
            for string in x_s:
                optimal_prob_s += qaoa_results.eigenstate[string] if string in qaoa_results.eigenstate else 0
            print("Minim_exp_val_L: {}, prob_s_L: {}".format(minim_exp_val, optimal_prob_s))
            optimal_prob_s_B = optimal_prob_s

            t1 = time()
            for point in perturbed_points:
                qaoa_results_B, _ = Fourier_QAOA(operator, quantum_instance, optimizer, p, point)
                exp_val = qaoa_results_B.eigenvalue*max_coeff + offset
                prob_s = 0
                for string in x_s:
                    prob_s += qaoa_results_B.eigenstate[string] if string in qaoa_results_B.eigenstate else 0
                if exp_val < minim_exp_val_B:
                    minim_exp_val_B = exp_val
                    optimal_prob_s_B = prob_s
                    optimal_fourier_point_B = np.array(qaoa_results_B.optimal_point)
            t2 = time()
            print("Time for intialising with 10 perturbed values: {}".format(t2-t1))
            print("Minim_exp_val_B: {}, prob_s_B: {}".format(minim_exp_val_B, optimal_prob_s_B))
            if minim_exp_val <= minim_exp_val_B:
                prob_s_s.append(optimal_prob_s)
            else:
                prob_s_s.append(optimal_prob_s_B)
        qubo_prob_s_s.append(prob_s_s)

    print(qubo_prob_s_s)

    with open('{}cars{}routes_{}.csv'.format(args["no_cars"], args["no_routes"], args["optimizer"]), 'w') as f:
        np.savetxt(f, qubo_prob_s_s, delimiter=',')
    finish = time()
    print("Time Taken: {}".format(finish - start))


def Fourier_QAOA(operator, quantum_instance, optimizer, p, initial_fourier_point):
    qaoa_instance = QAOAEx.QAOACustom(quantum_instance = quantum_instance,
                                        reps = p,
                                        force_shots = False,
                                        optimizer = optimizer,
                                        qaoa_name = "example_qaoa"
                                        )
    # qaoa_instance.set_parameterise_point_for_energy_evaluation(QAOAEx.convert_from_fourier_point)
    bounds = [(-np.pi, np.pi)]*len(initial_fourier_point)
    qaoa_results = qaoa_instance.solve(operator, initial_fourier_point, bounds)
    optimal_parameterised_point = qaoa_instance.latest_parameterised_point
    return qaoa_results, optimal_parameterised_point

def generate_points(point, no_perturb):
    alpha = 0.6
    points = [point.copy() for _ in range(no_perturb+1)]
    for r in range(1,no_perturb+1):
        for i in range(len(point)):
            if point[i] == 0:
                continue
            else:
                noise = np.random.normal(0, point[i]**2)
                points[r][i] += alpha*noise
    return points

def parse(raw_args):
    """Parse inputs for no_cars (number of cars),
                        no_routes (number of routes)

    Args:
        raw_args (list): list of strings describing the inputs e.g. ["-N 3", "-R 3"]

    Returns:
        dict: A dictionary of the inputs as values, and the name of variables as the keys
    """
    optimizer_choices = ["ADAM", "CG", "COBYLA", "L_BFGS_B",\
                        "NELDER_MEAD", "NFT", "POWELL",\
                         "SLSQP", "SPSA", "TNC", "P_BFGS", "BOBYQA"]
    parser = argparse.ArgumentParser()
    required_named = parser.add_argument_group('Required arguments')
    required_named.add_argument("--no_cars", "-N",
                                required = True,
                                help="Set the number of cars",
                                type = int
                                )
    required_named.add_argument("--no_routes", "-R",
                                required = True,
                                help="Set the number of routes for each car",
                                type = int
                                )
    required_named.add_argument("--penalty_multiplier", "-P",
                                required = True,
                                help="Set the penalty multiplier for QUBO constraint",
                                type = float
                                )
    required_named.add_argument("--optimizer", "-O",
                                required = True,
                                help = "Choose from Qiskit's optimizers",
                                type = str,
                                choices=optimizer_choices
                                )
    required_named.add_argument("--p_max", "-M",
                                required = True,
                                help = "Set maximum number of layers for QAOA",
                                type = int
                                )
    required_named.add_argument("--no_restarts", "-T", required = True,
                                help = "Set number of restarts for Interp QAOA",
                                type = int
                                )
    required_named.add_argument("--no_samples", "-S", required = True,
                                help = "Set number of samples/qubos with given no_cars no_routes",
                                type=int
                                )
    parser.add_argument("--visual",
                        "-V", default = False,
                        help="Activate routes visualisation with '-V' ",
                        action="store_true"
                        )
    args = parser.parse_args(raw_args)
    args = vars(args)
    return args

def solve_classically(qubo):
    """[summary]

    Args:
        qubo ([type]): [description]

    Returns:
        [type]: [description]
    """
    exact_mes = NumPyMinimumEigensolver()
    exact = MinimumEigenOptimizer(exact_mes)
    exact_result = exact.solve(qubo)
    print(exact_result)
    return exact_result

def arr_to_str(x_arr):
    """[summary]

    Args:
        x_arr ([type]): [description]

    Returns:
        [type]: [description]
    """
    x_arr = x_arr[::-1]
    string = ''.join(str(int(x_i)) for x_i in x_arr)
    return string


def filter_solutions(results, qaoa_dict, no_cars):
    """[summary]

    Args:
        results ([type]): [description]
        qaoa_dict ([type]): [description]
        no_cars ([type]): [description]

    Returns:
        [type]: [description]
    """
    routes = []
    variables = []
    for var_route, var_value in zip( results.items(), qaoa_dict.items() ):
        if var_route[0] == var_value[0] and int(var_value[1]) == 1 \
            and var_route[0] not in variables:
            routes.append(var_route[1])
            variables.append(var_route[0])
        elif var_route[0] != var_value[0]:
            print("Error, function returned non-matching variables")
            return None
        elif var_route[0] in variables:
            print("Error: Solution found two routes for one car")
            return None
    if len(variables) != no_cars:
        print("Error: At least one car did not have a valid route")
        return None
    return routes

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

def visualise_solution(graph, routes):
    """[summary]

    Args:
        graph ([type]): [description]
        routes ([type]): [description]

    Returns:
        [type]: [description]
    """
    colors = np.array(["r", "y", "b", "g", "w"]*10)[0:len(routes)]
    fig, axes = ox.plot_graph_routes(graph,
                                    routes,
                                    route_colors=colors,
                                    route_linewidth=4,
                                    node_size=10
                                    )
    return fig, axes

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
    pool = mp.Pool(cpus)
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

def check_solutions(routes):
    """[summary]

    Args:
        routes ([type]): [description]

    Raises:
        TypeError: [description]
    """
    try:
        len(routes)
        print("This is a valid solution")
    except TypeError as typ:
        raise TypeError("This is not a valid solution") from typ

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

def generate_valid_qubo(args):
    while True: #Generate valid QUBO
        try:
            #Generate a QUBO
            if args["visual"] is True:
                qubo, max_coeff, operator, offset, _, _, results = \
                    generate_qubo( [
                                    "-N "+str(args["no_cars"]),
                                    "-R "+str(args["no_routes"]),
                                    "-P "+str(args["penalty_multiplier"]),
                                    "-V"
                                    ]
                                )
            else:
                qubo, max_coeff, operator, offset, linear, quadratic, results = \
                    generate_qubo( [
                                    "-N "+str(args["no_cars"]),
                                    "-R "+str(args["no_routes"]),
                                    "-P "+str(args["penalty_multiplier"])
                                    ]
                                )
            #Classically solve generated QUBO
            classical_result = solve_classically(qubo)

            #Check solution
            variables_dict = classical_result.variables_dict
            routes = filter_solutions(results, variables_dict, args["no_cars"])
            check_solutions(routes)

            #If valid solution, save the solution.
            x_arr = classical_result.x
            x_str = arr_to_str(x_arr)

            break #break loop if valid solution, else increase penalty.
    
        except TypeError:
            #QUBO has invalid solution, start again with increased penalty.
            print("Starting with new qubo")
            args["penalty_multiplier"] += 0.05
    return qubo, max_coeff, operator, offset, routes

main(["-N 3", "-R 3", "-P 1.5", "-M 10", "-T 10", "-S 100"])


# operator, offset = quadratic_program.to_ising()

# initial_point = [0.40784, 0.73974, -0.53411, -0.28296]
# print()
# print("Solving QAOA...")
# qaoa_results = qaoa_instance.solve(operator, initial_point)

# qaoa_results_eigenstate = qaoa_results.eigenstate
# print("optimal_value:", qaoa_results.optimal_value)
# print("optimal_parameters:", qaoa_results.optimal_parameters)
# print("optimal_point:", qaoa_results.optimal_point)
# print("optimizer_evals:", qaoa_results.optimizer_evals)

# solutions = qaoa_instance.get_optimal_solutions_from_statevector(qaoa_results_eigenstate, quadratic_program)
# print_qaoa_solutions(solutions)
