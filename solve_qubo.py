import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from qiskit_optimization.problems import QuadraticProgram
from qiskit.providers.aer import QasmSimulator
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.minimum_eigen_solvers import NumPyMinimumEigensolver
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import ADAM, CG, COBYLA, L_BFGS_B, NELDER_MEAD, NFT, POWELL, SLSQP, SPSA, TNC
from qiskit import Aer, execute
from qiskit.aqua import QuantumInstance, aqua_globals
from generate_problem import import_map
from generate_qubo import main as generate_qubo
import multiprocessing as mp
from time import time
import argparse
import os

def main(raw_args = None):
    args = parse(raw_args)
    G = import_map('unimelb_graph.pkl')
    avg_avg_prob_success = []
    for qubo_no in range(args["no_samples"]):
        while True: #Generate valid QUBO
            try:
                #Generate a QUBO
                qubo, max_coeff, op, offset, linear, quadratic, results = ( generate_qubo( ["-N "+str(args["no_cars"]), "-R "+str(args["no_routes"]), "-P "+str(args["penalty_multiplier"]), "-V"] ) 
                        if args["visual"] == True
                        else generate_qubo(["-N "+str(args["no_cars"]), "-R "+str(args["no_routes"]), "-P "+str(args["penalty_multiplier"]) ] ) )
                del(linear, quadratic)
                
                #Classically solve generated QUBO
                classical_result = solve_classically(qubo)
                
                #Check solution
                variables_dict = classical_result.variables_dict
                routes = filter_solutions(results, variables_dict, args["no_cars"])
                check_solutions(routes)

                #If valid solution, save the solution.
                x = classical_result.x
                x = arr_to_str(x)

                break #break loop if valid solution, else increase penalty.

            except TypeError:
                #QUBO has invalid solution, start again with increased penalty.
                print("Starting with new qubo")
                args["penalty_multiplier"] += 0.05

        #Solve QAOA from QUBO with valid solution
        print("Solving with QAOA...")
        no_shots = 10000
        backend = Aer.get_backend('qasm_simulator', shots = no_shots)
        
        #Optimizers available from QISKIT - as chosen from parsed argument.
        optimizers = {"ADAM":ADAM(), "CG":CG(), "COBYLA":COBYLA(), "L_BFGS_B":L_BFGS_B(), "NELDER_MEAD":NELDER_MEAD(), "NFT":NFT(), "POWELL":POWELL(), "SLSQP":SLSQP(), "SPSA":SPSA(), "TNC":TNC()}
        optimizer = optimizers[args["optimizer"]]
        print("__"*50, "\n", optimizer.__class__.__name__, "\n", "__"*50)

        #Solve with interp QAOA up to args["p_max"] number of layers, restarting after p_max is reached.
        p_max = args["p_max"]
        no_restarts = args["no_restarts"]
        points = [ [2*np.pi*np.random.rand() for _ in range(2)] for _ in range(no_restarts) ]
        avg_optimal_values = []
        for i in range(no_restarts):
            point = points[i]
            qaoa_result, optimal_values = interp_qaoa(p_max, point, op, backend, optimizer, x, max_coeff, offset)
            del(qaoa_result)
            avg_optimal_values.append(optimal_values)
        avg_prob_success = np.array(avg_optimal_values)
        avg_prob_success = avg_prob_success.mean(axis = 0)
        print("Average over {} restarts for probability of success per layer of QUBO number {}: {}".format(no_restarts, qubo_no, avg_prob_success))
        avg_avg_prob_success.append(avg_prob_success)
        #Now start again with new QUBO of same number of qubits
        
    avg_avg_prob_success = np.array(avg_avg_prob_success)
    avg_avg_prob_success = avg_avg_prob_success.mean(axis = 0)
    print("Average over {} QUBOs for probability of success per layer: {}".format(args["no_samples"], avg_avg_prob_success))

    #Visualise solution
    if args["visual"]:
        visualise_solution(G, routes)

    return avg_avg_prob_success, optimizer.__class__.__name__, args["no_restarts"], args["no_samples"], p_max

def parse(raw_args):
    """Parse inputs for no_cars (number of cars),
                        no_routes (number of routes)               

    Args:
        raw_args (list): list of strings describing the inputs e.g. ["-N 3", "-R 3"]

    Returns:
        dict: A dictionary of the inputs as values, and the name of variables as the keys
    """
    optimizer_choices = ["ADAM", "CG", "COBYLA", "L_BFGS_B", "NELDER_MEAD", "NFT", "POWELL", "SLSQP", "SPSA", "TNC"]
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('Required arguments')
    requiredNamed.add_argument("--no_cars", "-N", required = True, help="Set the number of cars", type = int)
    requiredNamed.add_argument("--no_routes", "-R", required = True, help="Set the number of routes for each car", type = int)
    requiredNamed.add_argument("--penalty_multiplier", "-P", required = True, help="Set the penalty multiplier for QUBO constraint", type = float)
    requiredNamed.add_argument("--optimizer", "-O", required = True, help = "Choose from Qiskit's optimizers", type = str, choices=optimizer_choices)
    requiredNamed.add_argument("--p_max", "-M", required = True, help = "Set maximum number of layers for QAOA", type = int)
    requiredNamed.add_argument("--no_restarts", "-T", required = True, help = "Set number of restarts for Interp QAOA", type = int)
    requiredNamed.add_argument("--no_samples", "-S", required = True, help = "Set number of samples/qubos with given no_cars no_routes", type=int)
    parser.add_argument("--visual", "-V", default = False, help="Activate routes visualisation with '-V' ", action="store_true")
    args = parser.parse_args(raw_args)
    args = vars(args)
    return args

def arr_to_str(x):
    x = x[::-1]
    string = ''.join(str(int(x_i)) for x_i in x)
    return string

def interp_qaoa(p_max, point, op, backend, optimizer, x, max_coeff, offset):
    prob_s = []
    for p in range(1, p_max+1):
        print("p: {}".format(p))
        point = interp_point(optimal_point) if p != 1 else point
        qaoa_result = solve_qubo_qaoa(op, p, point, backend, optimizer)
        optimal_value = qaoa_result['optimal_value']*max_coeff + offset
        optimal_point = qaoa_result['optimal_point']
        optimizer_evals = qaoa_result['optimizer_evals']
        estate = qaoa_result['eigenstate']
        print("Optimizer_evals: {}".format(optimizer_evals))
        print("Optimal_point: {} \nOptimal_value: {:0.4f}".format(optimal_point, optimal_value))
        prob_x = estate[x] if x in estate else 0
        print("Prob of state {}: {:0.2%}\n".format(x, prob_x))
        prob_s.append(prob_x)
    return qaoa_result, prob_s

def solve_classically(qubo):
    exact_mes = NumPyMinimumEigensolver()
    exact = MinimumEigenOptimizer(exact_mes)
    exact_result = exact.solve(qubo)
    print(exact_result)
    return exact_result

def solve_qubo_qaoa(operator, p, point, backend, optimizer):
    qaoa = QAOA(optimizer = optimizer, reps=p,  initial_point=point, include_custom=True, quantum_instance=backend)
    qaoa_result = qaoa.compute_minimum_eigenvalue(operator=operator)
    qaoa_result = ({"optimizer_evals": qaoa_result.optimizer_evals,
                    "optimizer_time": qaoa_result.optimizer_time,
                    "optimal_value": qaoa_result.eigenvalue,
                    "optimal_point": qaoa_result.optimal_point,
                    "eigenstate": qaoa_result.eigenstate,
                    })
    return qaoa_result

def print_return_string(string):
    print(string)
    return string

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

def get_cost(qubo, no_qubits):
    # values = ['{}'.format()]
    cpus = mp.cpu_count()
    pool = mp.Pool(cpus)
    params = (('0'*(no_qubits-len('{0:b}'.format(k)))+'{0:b}'.format(k), qubo, no_qubits) for k in range(2**no_qubits))
    values = pool.starmap_async(eval_cost, params)
    values = values.get()
    # values = dict(values)
    pool.close()
    pool.join()
    sort_values = sorted(values, key=lambda x: x[1])
    return sort_values

def check_solutions(routes):
    try:
        len(routes)
        print("This is a valid solution")
    except TypeError:
        raise TypeError("This is not a valid solution")

def filter_solutions(results, qaoa_dict, no_cars):
    routes = []
    variables = []
    for var_route, var_value in zip( results.items(), qaoa_dict.items() ):
        if var_route[0] == var_value[0] and int(var_value[1]) == 1 and var_route[0] not in variables:
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

def eval_cost(x, qubo_problem, no_qubits, max_coeff, offset):
    cost = qubo_problem.objective.evaluate([int(x[i]) for i in range(no_qubits)]) if isinstance(x, np.ndarray) else qubo_problem.objective.evaluate([int(x[no_qubits-1-i]) for i in range(no_qubits)]) 
    cost /= max_coeff
    cost -= offset
    return x, cost

def visualise_solution(G, routes):
    colors = np.array(["r", "y", "b", "g", "w"]*10)[0:len(routes)]
    fig, ax = ox.plot_graph_routes(G, routes, route_colors=colors, route_linewidth=4, node_size=10)
    return fig, ax

if __name__ == "__main__":
    start = time()
    avg_avg_prob_success, optimizer_name, no_restarts, no_samples, p_max = main()
    finish = time()

    #Save results
    with open('{}.txt'.format(optimizer_name), 'w') as f:
        print("No_restarts: " + str(no_restarts) + "\nNo_samples: " + str(no_samples) + "\nMax number of layers: " + str(p_max), file = f)

    with open('{}.csv'.format(optimizer_name), 'w') as f:
        np.savetxt(f, avg_avg_prob_success, delimiter=',')

    print("Time Taken: {}s".format(finish-start))