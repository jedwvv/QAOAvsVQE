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
from QAOA_methods import CustomQAOA

class RQAOA:
    def __init__(self, qubo, no_cars, no_routes):
        var_list = qubo.variables
        self.optimizer = NLOPT_Optimizer("LN_BOBYQA")
        self.original_qubo = qubo
        self.qubo = qubo
        op, offset = qubo.to_ising()
        self.operator = op
        self.offset = offset
        self.quantum_instance = QuantumInstance(backend = Aer.get_backend("aer_simulator_matrix_product_state"), shots = 1024)
        # self.variables_solutions = {var.name:None for var in var_list}
        # self.variables_qubits = {var.name:None for var in var_list}
        self.replacements = {var.name:None for var in var_list}
        self.no_cars = no_cars
        self.no_routes = no_routes
        self.car_blocks = np.empty(shape = (no_cars,), dtype=object)
        for car_no in range(no_cars):
            self.car_blocks[car_no] = ["X_{}_{}".format(car_no, route_no) for route_no in range(no_routes)]
    
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
        print(self.initial_state.draw())
    
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
        classical_result, _ = solve_classically(self.qubo)
        self.result = classical_result
    
    def get_random_energy(self):
        random_energy = CustomQAOA(operator = self.operator,
                    quantum_instance = self.quantum_instance,
                    optimizer = self.optimizer,
                    reps = 1,
                    initial_state = self.initial_state,
                    mixer = self.mixer,
                    solve = False,
                    )
        self.random_energy = random_energy + self.offset
        return self.random_energy
    
    def solve_tqa_qaoa(self, p):
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
        return qaoa_results

    def perform_substitution_from_qaoa_results(self, qaoa_results):
        sorted_eigenstate = sorted(qaoa_results.eigenstate, key = lambda x: x[2], reverse = True)
        print(sorted_eigenstate[0])
        print(  "Approx_quality: {}".format((self.random_energy - sorted_eigenstate[0][1])/ (self.random_energy - self.result.fval)) )
        correlations = self.get_correlations(qaoa_results.eigenstate)
        print(correlations)
        i, j = self.find_strongest_correlation(correlations)
        new_qubo = deepcopy(self.qubo)
        x_i, x_j = new_qubo.variables[i].name, new_qubo.variables[j].name
        if correlations[i, j] > 0:
                # set x_i = x_j
                new_qubo = new_qubo.substitute_variables(variables={i: (j, 1)})
                self.replacements[x_i] = (x_j, 1)
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
                self.replacements[x_i] = (x_j, -1)
        car_block = int(x_i[2])
        self.car_blocks[car_block].remove(x_i)
        self.qubo = new_qubo
        op, offset = new_qubo.to_ising()
        self.operator = op
        self.offset = offset
        # print(self.car_blocks)
        # print(self.replacements)
        
        
        
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
        for x, _, prob in states:
            for i in range(n):
                for j in range(i):
                    if x[i] == x[j]:
                        correlations[i, j] += prob
                    else:
                        correlations[i, j] -= prob
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
    
    # def solve_qaoa(self):


        # if self.

    # def solve_QAOA(self, reps):
        
    
    # def solve_qaoa()

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
    rqaoa.construct_initial_state()
    rqaoa.construct_mixer()
    random_energy = rqaoa.get_random_energy()
    qaoa_results = rqaoa.solve_tqa_qaoa(1)
    rqaoa.perform_substitution_from_qaoa_results(qaoa_results)
    num_vars = args["no_cars"]*args["no_routes"] - 1
    while num_vars > 1:
        rqaoa.construct_initial_state()
        rqaoa.construct_mixer()
        qaoa_results = rqaoa.solve_tqa_qaoa(1)
        print("Done")
        rqaoa.perform_substitution_from_qaoa_results(qaoa_results)
        print("Done_2")
        num_vars = rqaoa.qubo.get_num_vars()

    rqaoa.construct_initial_state()
    rqaoa.construct_mixer()
    qaoa_results = rqaoa.solve_tqa_qaoa(2)
    print(qaoa_results)
    max_state = sorted(qaoa_results.eigenstate, key = lambda x: x[2], reverse=True)[0]
    var_values = {}
    var_last = rqaoa.qubo.variables[0].name
    var_values[var_last] = int(max_state[0])

    while True:
        for var, replacement in rqaoa.replacements.items():
            if replacement == None:
                continue
            elif replacement[0] in var_values and var not in var_values and replacement[1] != None:
                var_values[var] = var_values[ replacement[0] ] if replacement[1] == 1 else 1 - var_values[ replacement[0] ]

        if len(var_values.keys()) == args["no_cars"]*args["no_routes"]:
            break
    print(rqaoa.replacements)
    print(rqaoa.result)
    print(var_values)
    print(list(var_values.values()))

if __name__ == '__main__':
    main()

finish = time()
print("Time taken: {} s".format(finish - start))