"""[summary]

Raises:
    TypeError: [description]

Returns:
    [type]: [description]
"""
from time import time
import warnings
import numpy as np
import pickle as pkl
from qiskit import Aer
from qiskit.utils.quantum_instance import QuantumInstance
import QAOAEx
from parser_all import parse
from generate_qubos import visualise_solution
from generate_problem import import_map
from classical_optimizers import NLOPT_Optimizer
from QAOA_methods import (CustomQAOA,
                         generate_points,
                         get_costs,
                         find_all_ground_states,
                         count_coupling_terms,
                         construct_initial_state,
                         n_qbit_mixer)

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

    x_s = find_all_ground_states(qubo)

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
            points = generate_points(next_fourier_point, no_perturb=10, penalty=0.6)
            print(points) \
                # + generate_points(next_fourier_point_B, 10, penalty)
            draw_circuit = False
        #empty lists to save following results to choose best result
        results = []
        exp_vals = []
        for r in range(len(points)):
            point = points[r]
            if np.amax(np.abs(point)) < np.pi/2: 
                qaoa_results, optimal_circ = CustomQAOA(operator,
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

if __name__ == "__main__":
    main()