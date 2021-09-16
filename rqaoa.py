from time import time
start = time()

import numpy as np
import pickle as pkl
from copy import deepcopy
from qiskit import Aer, QuantumCircuit, QuantumRegister
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.quantum_info import Statevector
import QAOAEx
from parser_all import parse
from generate_qubos import solve_classically, arr_to_str, str_to_arr, visualise_solution
from generate_problem import import_map
from classical_optimizers import NLOPT_Optimizer
from qiskit_optimization import QuadraticProgram
from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops import PauliOp
from QAOA_methods import CustomQAOA, find_all_ground_states
from pprint import pprint

def main(args=None):
    start = time()
    if args == None:
        args = parse()
    with open('qubos_{}_car_{}_routes/qubo_{}.pkl'.format(args["no_cars"], args["no_routes"], args["no_samples"]), 'rb') as f:
        load_data = pkl.load(f)
        if len(load_data) == 5:
            qubo, max_coeff, operator, offset, routes = load_data
        else:
            qubo, max_coeff, operator, offset, routes, classical_result = load_data
    rqaoa = RQAOA(qubo, args["no_cars"], args["no_routes"])
    rqaoa.solve_classically()
    print(rqaoa.result)
    print("First round of TQA-QAOA. Results below:")
    qaoa_results = rqaoa.solve_qaoa(2, tqa=True)
    rqaoa.perform_substitution_from_qaoa_results(qaoa_results, biased = args["bias"])
    print("Performed variable substition(s) and constructed new initial state and mixer.")
    num_vars = rqaoa.qubo.get_num_vars()
    print("Remaining variables: {}".format(num_vars))
    t = 1
    while num_vars > 1:
        print("-"*50)
        t+=1
        qaoa_results = rqaoa.solve_qaoa(2, tqa=True)
        print( "Round {} of TQA-QAOA. Results below:".format(t) )
        rqaoa.perform_substitution_from_qaoa_results(qaoa_results, biased = args["bias"])
        print("Performed variable substition(s) and constructed new initial state and mixer.")
        num_vars = rqaoa.qubo.get_num_vars()
        print("Remaining variables: {}".format(num_vars))
    print("-"*50)
    p=2
    qaoa_results = rqaoa.solve_qaoa( p, point = [0]* (2*p) ) #qaoa with tqa = False by default here
    print( "Final round of QAOA Done. Eigenstate below:" )
    pprint(qaoa_results.eigenstate)
    print( rqaoa.prob_s )
    print( rqaoa.approx_s )
    max_state = sorted(qaoa_results.eigenstate, key = lambda x: x[2], reverse=True)[0]
    var_last = rqaoa.qubo.variables[0].name
    rqaoa.var_values[var_last] = int(max_state[0])
    var_values = rqaoa.var_values
    replacements = rqaoa.replacements
    while True:
        for var, replacement in replacements.items():
            if replacement == None:
                continue
            elif replacement[0] in var_values and var not in var_values and replacement[1] != None:
                var_values[var] = var_values[ replacement[0] ] if replacement[1] == 1 else 1 - var_values[ replacement[0] ]
        if len(var_values.keys()) == args["no_cars"]*args["no_routes"]:
            break
    print(rqaoa.result)
    print(var_values)
    list_values = list(var_values.values())
    cost = rqaoa.original_qubo.objective.evaluate(var_values)
    
    print("{}, Cost: {}".format(list_values, cost))
    save_results = np.array( [rqaoa.prob_s, rqaoa.approx_s] )
    
    if args["bias"]:
        with open('results_{}cars{}routes_mps/Biased_RQAOA_{}_tan.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"]), 'w') as f:
            np.savetxt(f, save_results, delimiter=',')
            print("Results saved in results_{}cars{}routes_mps/Biased_RQAOA_{}_tan.csv".format(args["no_cars"], args["no_routes"], args["no_samples"]))
    else:
        with open('results_{}cars{}routes_mps/Regular_RQAOA_{}_new.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"]), 'w') as f:
            np.savetxt(f, save_results, delimiter=',')
            print("Results saved in results_{}cars{}routes_mps/Regular_RQAOA_{}_new.csv".format(args["no_cars"], args["no_routes"], args["no_samples"]))

class RQAOA:
    def __init__(self, qubo, no_cars, no_routes):
        var_list = qubo.variables
        opt_str = "LN_SBPLX"
        print("Optimizer: {}".format(opt_str))
        self.optimizer = NLOPT_Optimizer(opt_str)
        self.original_qubo = qubo
        self.qubo = qubo
        op, offset = qubo.to_ising()
        self.operator = op
        self.offset = offset
        self.quantum_instance = QuantumInstance(backend = Aer.get_backend("aer_simulator_matrix_product_state"), shots = 1024)
        self.replacements = {var.name:None for var in var_list}
        self.no_cars = no_cars
        self.no_routes = no_routes
        self.car_blocks = np.empty(shape = (no_cars,), dtype=object)
        for car_no in range(no_cars):
            self.car_blocks[car_no] = ["X_{}_{}".format(car_no, route_no) for route_no in range(no_routes)]
        self.qaoa_result = None
        self.benchmark_energy = None
        self.var_values = {}
        self.construct_initial_state()
        self.construct_mixer()
        self.get_random_energy()
        self.get_benchmark_energy()
        self.prob_s = []
        self.approx_s = []
        self.optimal_point = None

    
    def construct_initial_state(self):
        qc = QuantumCircuit()
        for car_no in range(self.no_cars):
            car_block = self.car_blocks[car_no]
            R = len(car_block)
            if R != 0 and R != 1:
                num_qubits = qc.num_qubits
                q_regs = QuantumRegister(R, 'car_{}'.format((car_no)))
                qc.add_register(q_regs)
                w_one = QuantumCircuit(q_regs)
                for r in range(0,R-1):
                    if r == 0:
                        w_one.ry(2*np.arccos(1/np.sqrt(R-r)),r)
                    elif r != R-1:
                        w_one.cry(2*np.arccos(1/np.sqrt(R-r)), r-1, r)
                for r in range(1,R):
                    w_one.cx(R-r-1,R-r)
                w_one.x(0)
                qc.append(w_one, range(num_qubits, num_qubits+R))
            elif R == 1:
                num_qubits = qc.num_qubits
                q_regs = QuantumRegister(R, 'car_{}'.format((car_no)))
                qc.add_register(q_regs)
                w_one = QuantumCircuit(q_regs)
                w_one.h(0)
                qc.append(w_one, range(num_qubits, num_qubits+R))
            else:
                continue
        qc = qc.decompose()
        self.initial_state = qc
    
    def construct_mixer(self):
        from qiskit.circuit.parameter import Parameter
        initial_state = self.initial_state 
        no_qubits = initial_state.num_qubits
        t = Parameter('t')
        mixer = QuantumCircuit(no_qubits)
        mixer.append(initial_state.inverse(), range(no_qubits))
        mixer.rz(2*t, range(no_qubits))
        mixer.append(initial_state, range(no_qubits))
        self.mixer = mixer
    
    def solve_classically(self):
        x_s, opt_value, classical_result, _ = find_all_ground_states(self.qubo)
        self.result = classical_result
        x_arr = classical_result.x
        self.x_s =  [ x_str[::-1] for x_str in x_s ]
        self.opt_value = opt_value
    
    def get_random_energy(self):
        #Get random benchmark energy for 0 layer QAOA (achieved by using layer 1 QAOA with [0,0] angles)
        random_energy = CustomQAOA(operator = self.operator,
                    quantum_instance = self.quantum_instance,
                    optimizer = self.optimizer,
                    reps = 1,
                    initial_state = self.initial_state,
                    mixer = self.mixer,
                    solve = False,
                    )
        temp = random_energy
        self.random_energy = temp + self.offset
        print("random energy: {}".format(self.random_energy))
        
        
    
    def get_benchmark_energy(self):
        #Get benchmark energy with 0-layer QAOA (just as random_energy)
        benchmark_energy = CustomQAOA(operator = self.operator,
                    quantum_instance = self.quantum_instance,
                    optimizer = self.optimizer,
                    reps = 1,
                    initial_state = self.initial_state,
                    mixer = self.mixer,
                    solve = False,
                    )
        temp = benchmark_energy + self.offset
        #Choose minimum of benchmark_energy if there already exists self.benchmark_energy
        self.benchmark_energy = min(self.benchmark_energy, temp) if self.benchmark_energy else temp
        return self.benchmark_energy
    
    def solve_qaoa(self, p, **kwargs):
        
        if self.optimal_point and 'point' not in kwargs:
            point = self.optimal_point
        elif 'point' in kwargs:
            point = kwargs['point']

        fourier_parametrise = True
        self.optimizer.set_options(maxeval = 1000)
        
        tqa = False if 'tqa' not in kwargs else kwargs['tqa']
        if tqa:
            deltas = np.arange(0.45, 0.91, 0.05)
            point = np.append( [ (i+1)/p for i in range(p) ] , [ 1-(i+1)/p for i in range(p) ] )
            points = [delta*point for delta in deltas]
            fourier_parametrise = True
            if fourier_parametrise:
                points = [ QAOAEx.convert_to_fourier_point(point, len(point)) for point in points ]
            self.optimizer.set_options(maxeval = 1000)
            qaoa_results, _, _ = CustomQAOA( self.operator,
                                                        self.quantum_instance,
                                                        self.optimizer,
                                                        reps = p,
                                                        initial_state = self.initial_state,
                                                        mixer = self.mixer,
                                                        fourier_parametrise = fourier_parametrise,
                                                        list_points = points,
                                                        qubo = self.qubo
                                                        )
            
        else:
            point = point if point else [0]*(2*p)
            qaoa_results, _, _ = CustomQAOA( self.operator,
                                        self.quantum_instance,
                                        self.optimizer,
                                        reps = p,
                                        initial_state = self.initial_state,
                                        initial_point = point,
                                        mixer = self.mixer,
                                        fourier_parametrise = fourier_parametrise,
                                        qubo = self.qubo
                                        )
        point = qaoa_results.optimal_point
        qaoa_results.eigenvalue = sum( [ x[1] * x[2] for x in qaoa_results.eigenstate ] )
        self.optimal_point = QAOAEx.convert_to_fourier_point(point, len(point)) if fourier_parametrise else point
        self.qaoa_result = qaoa_results
              
        print("-"*50)
        pprint(qaoa_results.eigenstate)
        print("Eigenvalue: {}".format(qaoa_results.eigenvalue))
        print("Optimal point: {}".format(qaoa_results.optimal_point))
        print("Optimizer Evals: {}".format(qaoa_results.optimizer_evals))
              
        sorted_eigenstate_by_energy = sorted(qaoa_results.eigenstate, key = lambda x: x[1])
        sorted_eigenstate_by_prob = sorted(qaoa_results.eigenstate, key = lambda x: x[2], reverse = True)
        scale = self.random_energy - self.result.fval
        approx_quality = (self.random_energy - sorted_eigenstate_by_energy[0][1])/ scale 
        energy_prob = {}
        for x in qaoa_results.eigenstate:
            energy_prob[ int(x[1]) ] = energy_prob.get(int(x[1]), 0) + x[2]
        print("energy_prob: {}".format(energy_prob))
        prob_s = energy_prob.get(int(self.result.fval), 0)
        self.prob_s.append( prob_s )
        self.approx_s.append( approx_quality )
        print( "QAOA lowest energy solution: {}".format(sorted_eigenstate_by_energy[0]) )
        print( "Approx_quality: {}".format(approx_quality) )
        print( "QAOA most probable solution: {}".format(sorted_eigenstate_by_prob[0]) )
        print( "Approx_quality: {}".format((self.random_energy - sorted_eigenstate_by_prob[0][1])/ scale) )

        return qaoa_results
    
    def perform_substitution_from_qaoa_results(self, qaoa_results, update_benchmark_energy=True, biased = True):
        
        correlations = self.get_biased_correlations(qaoa_results.eigenstate) if biased else self.get_correlations(qaoa_results.eigenstate)
        i, j = self.find_strongest_correlation(correlations)
        new_qubo = deepcopy(self.qubo)
        x_i, x_j = new_qubo.variables[i].name, new_qubo.variables[j].name
        print( "\nCorrelation: < {} {} > = {}".format(x_i, x_j, correlations[i, j])) 
        
        car_block = int(x_i[2])
        #If same car_block and x_i = x_j, then both must be 0 since only one 1 in a car block
        if x_i[2] == x_j[2] and correlations[i, j] > 0 and len(self.car_blocks[car_block]) > 2: 
            # set x_i = x_j = 0
            new_qubo = new_qubo.substitute_variables({x_i: 0, x_j:0})
            if new_qubo.status == QuadraticProgram.Status.INFEASIBLE:
                raise QiskitOptimizationError('Infeasible due to variable substitution {} = {} = 0'.format(x_i, x_j))
            self.var_values[x_i] = 0
            self.var_values[x_j] = 0
            self.car_blocks[car_block].remove(x_i)
            self.car_blocks[car_block].remove(x_j)
            print("Two variable substitutions were performed due to extra information from constraints.")
            if len(self.car_blocks[car_block]) == 1: #If only one remaining variable
                x_r = self.car_blocks[car_block][0] #remaining variable
                #Check if all other variables are 0 (then their sum should be 0) -> so x_r must be 1
                check = sum( [self.var_values.get("X_{}_{}".format(car_block, route_no), 0) for route_no in range(self.no_routes)] )
                if check == 0:
                    new_qubo = new_qubo.substitute_variables({x_r: 1})
                    if new_qubo.status == QuadraticProgram.Status.INFEASIBLE:
                        raise QiskitOptimizationError('Infeasible due to variable substitution {} = 1'.format(x_r))
                    self.car_blocks[car_block].remove(x_r)
                    print("{} = 1 can also be determined from all other variables being 0 for car_{}".format(x_r, car_block))
        elif x_i[2] != x_j[2] and correlations[i, j] > 0: 
            # set x_i = x_j
            new_qubo = new_qubo.substitute_variables(variables={i: (j, 1)})
            if new_qubo.status == QuadraticProgram.Status.INFEASIBLE:
                raise QiskitOptimizationError('Infeasible due to variable substitution {} = {}'.format(x_i, x_j))            
            self.replacements[x_i] = (x_j, 1)
            self.car_blocks[car_block].remove(x_i)
        else:
            # set x_i = 1 - x_j, this is done in two steps:
            # 1. set x_i = 1 + x_i
            # 2. set x_i = -x_j

            # 1a. get additional offset from the 1 on the RHS of (xi -> 1+xi)
            constant = new_qubo.objective.constant
            constant += new_qubo.objective.linear[i]
            constant += new_qubo.objective.quadratic[i, i]
            new_qubo.objective.constant = constant

            #1b get additional linear part from quadratic terms becoming linear due to the same 1 in the 1+xi as above
            for k in range(new_qubo.get_num_vars()):
                coeff = new_qubo.objective.linear[k]
                if k == i:
                    coeff += 2*new_qubo.objective.quadratic[i, k]
                else:
                    coeff += new_qubo.objective.quadratic[i, k]

                # set new coefficient if not too small
                if np.abs(coeff) > 1e-10:
                    new_qubo.objective.linear[k] = coeff
                else:
                    new_qubo.objective.linear[k] = 0

            #2 set xi = -xj
            new_qubo = new_qubo.substitute_variables(variables={i: (j, -1)})
            if new_qubo.status == QuadraticProgram.Status.INFEASIBLE:
                raise QiskitOptimizationError('Infeasible due to variable substitution {} = -{}'.format(x_i, x_j))      
            self.replacements[x_i] = (x_j, -1)
            self.car_blocks[car_block].remove(x_i)
        self.qubo = new_qubo
        op, offset = new_qubo.to_ising()
        self.operator = op
        self.offset = offset
        self.construct_initial_state()
        self.construct_mixer()
        if update_benchmark_energy:
            temp = self.get_benchmark_energy()
        
        
    def get_correlations(self, state) -> np.ndarray:
        """
        Get <Zi x Zj> correlation matrix from the eigenstate(state: Dict).

        Returns:
            A correlation matrix.
        """
        states = state
        # print(states)
        x, _, prob = states[0]
        n = len(x)
        correlations = np.zeros((n, n))
        for x, cost, prob in states:
            for i in range(n):
                for j in range(i):
                    if x[i] == x[j]:
                        correlations[i, j] += prob
                    else:
                        correlations[i, j] -= prob
        return correlations
    
              
    def get_biased_correlations(self, state) -> np.ndarray:
        """
        Get <Zi x Zj> correlation matrix from the eigenstate(state: Dict).

        Returns:
            A correlation matrix.
        """
        states = state
        # print(states)
        x, _, prob = states[0]
        n = len(x)
        correlations = np.zeros((n, n))
        for x, cost, prob in states:
                scaled_approx_quality = np.arctan(self.benchmark_energy - cost)/np.pi + 0.5
                for i in range(n):
                    for j in range(i):
                        if x[i] == x[j]:
                            correlations[i, j] += scaled_approx_quality * prob
                        else:
                            correlations[i, j] -= scaled_approx_quality * prob
        return correlations

    def find_strongest_correlation(self, correlations):

        # get absolute values and set diagonal to -1 to make sure maximum is always on off-diagonal
        abs_correlations = np.abs(correlations)
        diagonal = np.diag( np.ones(len(correlations)) )
        abs_correlations = abs_correlations - diagonal


        # get index of maximum (by construction on off-diagonal)
        m_max = np.argmax(abs_correlations.flatten())

        # translate back to indices
        i = int(m_max // len(correlations))
        j = int(m_max - i*len(correlations))

        return (i, j)

if __name__ == '__main__':
    main()

finish = time()
print("Time taken: {} s".format(finish - start))