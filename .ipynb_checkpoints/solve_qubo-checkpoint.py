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
    start = time()
    args = parse(raw_args)
    output_txt = open('{}cars{}routes_{}_output.txt'.format(args["no_cars"], args["no_routes"], args["optimizer"]), 'w')
    G = import_map('unimelb_graph.pkl')
    avg_avg_prob_success = []
    for qubo_no in range(args["no_samples"]):
        output_txt = open('{}cars{}routes_{}_output.txt'.format(args["no_cars"], args["no_routes"], args["optimizer"]), 'a')
        print("__"*50, "\nQUBO NO: {}\n".format(qubo_no), "__"*50, file = output_txt)
        output_txt.close()
        while True: #Generate valid QUBO
            try:
                #Generate a QUBO
                qubo, max_coeff, op, offset, linear, quadratic, results = ( generate_qubo( ["-N "+str(args["no_cars"]), "-R "+str(args["no_routes"]), "-P "+str(args["penalty_multiplier"]), "-V"] ) 
                        if args["visual"] == True
                        else generate_qubo(["-N "+str(args["no_cars"]), "-R "+str(args["no_routes"]), "-P "+str(args["penalty_multiplier"]) ] ) )
                del(linear, quadratic)
                
                #Classically solve generated QUBO
                output_txt = open('{}cars{}routes_{}_output.txt'.format(args["no_cars"], args["no_routes"], args["optimizer"]), 'a')
                classical_result = solve_classically(qubo, output_txt)
                output_txt.close()
                #Check solution
                variables_dict = classical_result.variables_dict
                output_txt = open('{}cars{}routes_{}_output.txt'.format(args["no_cars"], args["no_routes"], args["optimizer"]), 'a')
                routes = filter_solutions(results, variables_dict, args["no_cars"], output_txt)
                check_solutions(routes, output_txt)
                output_txt.close()
                #If valid solution, save the solution.
                x = classical_result.x
                x = arr_to_str(x)

                break #break loop if valid solution, else increase penalty.

            except TypeError:
                #QUBO has invalid solution, start again with increased penalty.
                output_txt = open('{}cars{}routes_{}_output.txt'.format(args["no_cars"], args["no_routes"], args["optimizer"]), 'a')
                print("Starting with new qubo", file = output_txt)
                output_txt.close()
                args["penalty_multiplier"] += 0.05

        #Solve QAOA from QUBO with valid solution
        output_txt = open('{}cars{}routes_{}_output.txt'.format(args["no_cars"], args["no_routes"], args["optimizer"]), 'a')
        print("Solving with QAOA...", file = output_txt)
        output_txt.close()
        no_shots = 10000
        backend = Aer.get_backend('qasm_simulator', shots = no_shots)
        
        #Optimizers available from QISKIT - as chosen from parsed argument.
        optimizers = {"ADAM":ADAM(), "CG":CG(), "COBYLA":COBYLA(), "L_BFGS_B":L_BFGS_B(), "NELDER_MEAD":NELDER_MEAD(), "NFT":NFT(), "POWELL":POWELL(), "SLSQP":SLSQP(), "SPSA":SPSA(), "TNC":TNC()}
        optimizer = optimizers[args["optimizer"]]
        output_txt = open('{}cars{}routes_{}_output.txt'.format(args["no_cars"], args["no_routes"], args["optimizer"]), 'a')
        print("__"*50, "\n"+optimizer.__class__.__name__+"\n", "__"*50, file = output_txt)
        output_txt.close()

        #Solve with interp QAOA up to args["p_max"] number of layers, restarting after p_max is reached.
        p_max = args["p_max"]
        no_restarts = args["no_restarts"]
        points = [ [2*np.pi*np.random.rand() for _ in range(2)] for _ in range(no_restarts) ]
        all_prob_s = []
        all_optimal_values = []
        for i in range(no_restarts):
            point = points[i]
            output_txt = open('{}cars{}routes_{}_output.txt'.format(args["no_cars"], args["no_routes"], args["optimizer"]), 'a')
            qaoa_result, optimal_value, final_prob_s = interp_qaoa(p_max, point, op, backend, optimizer, x, max_coeff, offset, output_txt)
            output_txt.close()
            all_optimal_values.append(optimal_value)
            all_prob_s.append(final_prob_s)
            del(qaoa_result)
        i = np.where(all_prob_s==np.amax(all_prob_s))[0][0]
        avg_avg_prob_success.append(all_optimal_values[i])

    avg_avg_prob_success = np.array(avg_avg_prob_success)
    avg_avg_prob_success = avg_avg_prob_success.mean(axis = 0)
    output_txt = open('{}cars{}routes_{}_output.txt'.format(args["no_cars"], args["no_routes"], args["optimizer"]), 'a')
    print("Average over {} QUBOs for probability of success per layer: {}".format(args["no_samples"], avg_avg_prob_success), file=output_txt)
    output_txt.close()

    #Visualise solution
    if args["visual"]:
        visualise_solution(G, routes)
    finish = time()
    output_txt = open('{}cars{}routes_{}_output.txt'.format(args["no_cars"], args["no_routes"], args["optimizer"]), 'a')
    print("Time Taken: {}s".format(finish-start), file = output_txt)
    output_txt.close()

    return avg_avg_prob_success, args
 
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

def interp_qaoa(p_max, point, op, backend, optimizer, x, max_coeff, offset, output_file):
    prob_s = []
    for p in range(1, p_max+1):
        print("p: {}".format(p), file = output_file)
        point = interp_point(optimal_point) if p != 1 else point
        qaoa_result = solve_qubo_qaoa(op, p, point, backend, optimizer)
        optimal_point = qaoa_result['optimal_point']
        optimizer_evals = qaoa_result['optimizer_evals']
        estate = qaoa_result['eigenstate']
        prob_x = estate[x] if x in estate else 0
        print("Prob of state {}: {:0.2%}".format(x, prob_x), file = output_file)
        print("Optimizer Evals: {}".format(optimizer_evals), file = output_file)
        prob_s.append(prob_x)
    final_prob_s = prob_x
    return qaoa_result, prob_s, final_prob_s

def solve_classically(qubo, output_txt):
    exact_mes = NumPyMinimumEigensolver()
    exact = MinimumEigenOptimizer(exact_mes)
    exact_result = exact.solve(qubo)
    print(exact_result, file = output_txt)
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

def print_return_string(string, output_file):
    print(string, file = output_file)
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

def get_costs(qubo, no_qubits):
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

def check_solutions(routes, output_file):
    try:
        len(routes)
        print("This is a valid solution", file = output_file)
    except TypeError:
        raise TypeError("This is not a valid solution", file = output_file)

def filter_solutions(results, qaoa_dict, no_cars, output_txt):
    routes = []
    variables = []
    for var_route, var_value in zip( results.items(), qaoa_dict.items() ):
        if var_route[0] == var_value[0] and int(var_value[1]) == 1 and var_route[0] not in variables:
            routes.append(var_route[1])
            variables.append(var_route[0])
        elif var_route[0] != var_value[0]:
            print("Error, function returned non-matching variables", file = output_txt)
            return None
        elif var_route[0] in variables:
            print("Error: Solution found two routes for one car", file = output_txt)
            return None
    if len(variables) != no_cars:
        print("Error: At least one car did not have a valid route", file = output_txt)
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
    avg_avg_prob_success, args = main()

    #Save results
    optimizer_name = args["optimizer"]
    no_cars = args["no_cars"]
    no_routes = args["no_routes"]
    no_restarts = args["no_restarts"]
    no_samples = args["no_samples"]
    p_max = args["p_max"]
    with open('{}cars{}routes_{}.txt'.format(no_cars, no_routes, optimizer_name), 'w') as f:
        print("No_restarts: " + str(no_restarts) + "\nNo_samples: " + str(no_samples) + "\nMax number of layers: " + str(p_max), file = f)
    with open('{}cars{}routes_{}.csv'.format(no_cars, no_routes, optimizer_name), 'w') as f:
        np.savetxt(f, avg_avg_prob_success, delimiter=',')

    