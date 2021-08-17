from time import time
import warnings
import numpy as np
import pickle as pkl
from parser_all import parse
from qiskit import Aer
from qiskit.utils.quantum_instance import QuantumInstance
from generate_qubos import solve_classically, arr_to_str
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
    print("-"*50)
    print("QUBO_{}".format(qubo_no))
    #Load generated qubo_no
    with open('qubos_{}_car_{}_routes/qubo_{}.pkl'.format(args["no_cars"], args["no_routes"], qubo_no), 'rb') as f:
        qubo, max_coeff, operator, offset, routes = pkl.load(f)

    x_s, opt_value = find_all_ground_states(qubo)

    #Set optimizer method
    method = args["method"]
    optimizer = NLOPT_Optimizer(method = method, result_message=False)
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
    print("-"*50)
    print("Now solving with TQA_QAOA... Fourier Parametrisation: {}".format(fourier_parametrise))
    for p in range(1, p_max+1):
        construct_circ = False
        deltas = np.arange(0.45, 0.91, 0.05)
        point = np.append( [ (i+1)/p for i in range(p) ] , [ 1-(i+1)/p for i in range(p) ] )
        points = [delta*point for delta in deltas]
        print("-"*50)
        print("    "+"p={}".format(p))
        if fourier_parametrise:
            points = [ convert_to_fourier_point(point, len(point)) for point in points ]
        qaoa_results, optimal_circ = CustomQAOA(operator,
                                                    quantum_instance,
                                                    optimizer,
                                                    reps = p,
                                                    initial_state = initial_state,
                                                    mixer = mixer,
                                                    construct_circ= construct_circ,
                                                    fourier_parametrise = fourier_parametrise,
                                                    list_points = points
                                                    )
        exp_val = qaoa_results.eigenvalue * max_coeff + offset
        prob_s = 0
        for string in x_s:
            prob_s += qaoa_results.eigenstate[string] if string in qaoa_results.eigenstate else 0
        optimal_point = qaoa_results.optimal_point
        if fourier_parametrise:
           optimal_point = convert_from_fourier_point(optimal_point, len(optimal_point))
        approx_ratio = 1 - np.abs( (opt_value - exp_val) / opt_value )
        print("    "+"Optimal_point: {}".format(optimal_point))
        print("    "+"Exp_val: {}, Prob_s: {}, approx_ratio: {}".format(exp_val, prob_s, approx_ratio))
        approx_ratios.append(approx_ratio)
        prob_s_s.append(prob_s)
    print("-"*50)
    print("QAOA terminated")
    print("-"*50)
    print("Approximation ratios per layer", approx_ratios)
    print("Prob_success per layer", prob_s_s)
    save_results = np.append(approx_ratios, prob_s_s)
    if fourier_parametrise:
        with open('results_{}cars{}routes/TQA_F_{}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"]), 'w') as f:
            np.savetxt(f, save_results, delimiter=',')
        print("Results saved in results_{}cars{}routes/TQA_F_{}.csv".format(args["no_cars"], args["no_routes"], args["no_samples"]))
    else:
        with open('results_{}cars{}routes/TQA_NF_{}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"]), 'w') as f:
            np.savetxt(f, save_results, delimiter=',')
        print("Results saved in results_{}cars{}routes/TQA_NF_{}.csv".format(args["no_cars"], args["no_routes"], args["no_samples"]))
    finish = time()
    print("Time Taken: {}".format(finish - start))


if __name__ == "__main__":
    main()