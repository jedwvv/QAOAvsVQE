from time import time
import numpy as np
import pickle as pkl
from parser_all import parse
from qiskit import Aer
from qiskit.utils.quantum_instance import QuantumInstance
from generate_qubos import solve_classically, arr_to_str
# from qiskit.algorithms.optimizers import COBYLA, SLSQP
from classical_optimizers import NLOPT_Optimizer
from QAOAEx import QAOACustom
from QAOA_methods import (CustomQAOA,
                         generate_points,
                         get_costs,
                         find_all_ground_states,
                         count_coupling_terms,
                         interp_point,
                         construct_initial_state,
                         n_qbit_mixer)
from QAOAEx import convert_from_fourier_point, convert_to_fourier_point
from qiskit_optimization import QuadraticProgram

# warnings.filterwarnings('ignore')

def main(args = None):
    """[summary]

    Args:
        raw_args ([type], optional): [description]. Defaults to None.
    """
    start = time()
    if args == None:
        args = parse()

    qubo_no = args["no_samples"]
    print_to_file("-"*50)
    print_to_file("QUBO_{}".format(qubo_no))
    #Load generated qubo_no
    with open('qubos_{}_car_{}_routes/qubo_{}.pkl'.format(args["no_cars"], args["no_routes"], qubo_no), 'rb') as f:
        qubo, max_coeff, operator, offset, routes = pkl.load(f)
    
    qubo = QuadraticProgram()
    qubo.from_ising(operator)

    x_s, opt_value, classical_result = find_all_ground_states(qubo)
    print_to_file(classical_result)
    
    #Set optimizer method
    method = args["method"]
    optimizer = NLOPT_Optimizer(method = method, result_message=False)
    # optimizer = COBYLA()
    backend = Aer.get_backend("statevector_simulator")
    quantum_instance = QuantumInstance(backend = backend)

    approx_ratios = []
    prob_s_s = []
    p_max = args["p_max"]
    no_routes, no_cars = (args["no_routes"], args["no_cars"])

    custom = True
    if custom:
        initial_state = construct_initial_state(no_routes = no_routes, no_cars = no_cars)
        mixer = n_qbit_mixer(initial_state)
    else:
        initial_state, mixer = (None, None)

    fourier_parametrise = args["fourier"]
    print_to_file("-"*50)
    print_to_file("Now solving with TQA_QAOA... Fourier Parametrisation: {}".format(fourier_parametrise))
#     maxeval = 125
    for p in range(1, p_max+1):
        construct_circ = False
        deltas = np.arange(0.45, 0.91, 0.05)
        point = np.append( [ (i+1)/p for i in range(p) ] , [ 1-(i+1)/p for i in range(p) ] )
        points = [delta*point for delta in deltas]
        print_to_file("-"*50)
        print_to_file("    "+"p={}".format(p))
        if fourier_parametrise:
            points = [ convert_to_fourier_point(point, len(point)) for point in points ]
#         maxeval *= 2 #Double max_allowed evals for optimizer
#         optimizer.set_options(maxeval = maxeval)
        optimizer.set_options(maxeval = 1000*p)
        qaoa_results, optimal_circ = CustomQAOA(operator,
                                                    quantum_instance,
                                                    optimizer,
                                                    reps = p,
                                                    initial_state = initial_state,
                                                    mixer = mixer,
                                                    construct_circ= construct_circ,
                                                    fourier_parametrise = fourier_parametrise,
                                                    list_points = points,
                                                    qubo = qubo
                                                    )
        exp_val = qaoa_results.eigenvalue * max_coeff
        state_solutions = { item[0][::-1]: item[1:] for item in qaoa_results.eigenstate }
        for item in sorted(state_solutions.items(), key = lambda x: x[1][1], reverse = True)[0:5]:
            print_to_file( item )
        prob_s = 0
        for string in x_s:
            prob_s += state_solutions[string][1] if string in state_solutions else 0
        prob_s /= len(x_s) #normalise
        optimal_point = qaoa_results.optimal_point
        if fourier_parametrise:
            optimal_point = convert_from_fourier_point(optimal_point, len(optimal_point))
        approx_ratio = 1 - np.abs( (opt_value - exp_val) / opt_value )
        nfev = qaoa_results.cost_function_evals
        print_to_file("    "+"Optimal_point: {}, Nfev: {}".format(optimal_point, nfev))
        print_to_file("    "+"Exp_val: {}, Prob_s: {}, approx_ratio: {}".format(exp_val, prob_s, approx_ratio))
        approx_ratios.append(approx_ratio)
        prob_s_s.append(prob_s)
    print_to_file("-"*50)
    print_to_file("QAOA terminated")
    print_to_file("-"*50)
    print_to_file("Approximation ratios per layer: {}".format(approx_ratios))
    print_to_file("Prob_success per layer: {}".format(prob_s_s))
    save_results = np.append(approx_ratios, prob_s_s)
    if fourier_parametrise:
        with open('results_{}cars{}routes/TQA_F_{}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"]), 'w') as f:
            np.savetxt(f, save_results, delimiter=',')
        print_to_file("Results saved in results_{}cars{}routes/TQA_F_{}.csv".format(args["no_cars"], args["no_routes"], args["no_samples"]))
    else:
        with open('results_{}cars{}routes/TQA_NF_{}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"]), 'w') as f:
            np.savetxt(f, save_results, delimiter=',')
        print_to_file("Results saved in results_{}cars{}routes/TQA_NF_{}.csv".format(args["no_cars"], args["no_routes"], args["no_samples"]))
    finish = time()
    print_to_file("Time Taken: {}".format(finish - start))

def print_to_file(string, filepath = "TQA_output.txt"):
    print(string)
    with open(filepath, 'a') as f:
        print(string, file=f)

if __name__ == "__main__":
    main()

def main(params):
    """[summary]

    Args:
        params ([type]): [description]

    Returns:
        [type]: [description]
    """
    return value