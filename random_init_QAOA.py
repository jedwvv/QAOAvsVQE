from time import time
import warnings
import numpy as np
import pickle as pkl
from parser_all import parse
from qiskit import Aer
from qiskit.utils.quantum_instance import QuantumInstance
from generate_qubos import solve_classically, arr_to_str
from classical_optimizers import NLOPT_Optimizer
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
    print("Fourier Parametrisation: {}".format(fourier_parametrise))
    for p in range(1, p_max+1):

        if p == 1:
            points = [[0,0]] + [ np.random.uniform(low = -np.pi/2+0.01, high = np.pi/2-0.01, size = 2*p) for _ in range(2**p)]
            next_point = []
        else:
            penalty = 0.6
            points = [next_point_l] + generate_points(next_point, no_perturb=2**p-1, penalty=penalty)
        construct_circ = False

        #empty lists to save following results to choose best result
        results = []
        exp_vals = []
        print("-"*50)
        print("p={}".format(p))
        for r, point in enumerate(points):
            qaoa_results, optimal_circ = CustomQAOA(operator,
                                                        quantum_instance,
                                                        optimizer,
                                                        reps = p,
                                                        initial_fourier_point= point,
                                                        initial_state = initial_state,
                                                        mixer = mixer,
                                                        construct_circ= construct_circ,
                                                        fourier_parametrise = fourier_parametrise
                                                        )
            if r==0:
                if fourier_parametrise:
                    next_point_l = np.array(qaoa_results.optimal_point)
                    next_point_l = convert_from_fourier_point(next_point_l, 2*p+2)
                    next_point_l = convert_to_fourier_point(next_point_l, 2*p+2)
                else:
                    next_point_l = interp_point(qaoa_results.optimal_point)
            exp_val = qaoa_results.eigenvalue * max_coeff + offset
            exp_vals.append(exp_val)
            prob_s = 0
            for string in x_s:
                prob_s += qaoa_results.eigenstate[string] if string in qaoa_results.eigenstate else 0
            results.append((qaoa_results, optimal_circ, prob_s))
            print("Point_{}, Exp_val: {}, Prob_s: {}".format(p, r, exp_val, prob_s))
        minim_index = np.argmin(exp_vals)
        optimal_qaoa_result, optimal_circ, optimal_prob_s = results[minim_index]
        if fourier_parametrise:
            next_point = np.array(optimal_qaoa_result.optimal_point)
            next_point = convert_from_fourier_point(next_point, 2*p+2)
            next_point = convert_to_fourier_point(next_point, 2*p+2)
        else:
            next_point = interp_point(optimal_qaoa_result.optimal_point)
        if construct_circ:
            print(optimal_circ.draw(fold=150))
        minim_exp_val = exp_vals[minim_index]
        approx_ratio = 1.0 - np.abs( opt_value - minim_exp_val ) / opt_value
        print("Minimum: {}, prob_s: {}, approx_ratio {}".format(minim_exp_val, optimal_prob_s, approx_ratio))
        approx_ratios.append(approx_ratio)
        prob_s_s.append(optimal_prob_s)
    print("-"*50)
    print("QAOA terminated")
    print("-"*50)
    print("Approximation ratios per layer", approx_ratios)
    print("Prob_success per layer", prob_s_s)
    save_results = np.append(approx_ratios, prob_s_s)
    if fourier_parametrise:
        with open('results_{}cars{}routes/RI_F_{}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"]), 'w') as f:
            np.savetxt(f, save_results, delimiter=',')
        print("Results saved in results_{}cars{}routes/RI_F_{}.csv".format(args["no_cars"], args["no_routes"], args["no_samples"]))
    else:
        with open('results_{}cars{}routes/RI_NF_{}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"]), 'w') as f:
            np.savetxt(f, save_results, delimiter=',')
        print("Results saved in results_{}cars{}routes/RI_NF_{}.csv".format(args["no_cars"], args["no_routes"], args["no_samples"]))
    finish = time()
    print("Time Taken: {}".format(finish - start))


if __name__ == "__main__":
    main()