from time import time

start = time()

import numpy as np
import pickle as pkl
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
        self.qubo = qubo
        self.variables_solutions = {var.name:None for var in var_list}
        self.variables_qubits = {var.name:None for var in var_list}
        self.equalities = {var.name:None for var in var_list}
        self.no_cars = no_cars
        self.no_routes = no_routes
        self.car_blocks = np.empty(shape = (no_cars,), dtype=object)
        for car_no in range(no_cars):
            temp = ["X_{}_{}".format(car_no, route_no) for route_no in range(no_routes)]
            self.car_blocks[car_no] = temp
        print(self.car_blocks)
    
    def construct_initial_state(self):
        qc = QuantumCircuit()
        for car_no in range(self.no_cars):
            car_block = self.car_blocks[car_no]
            R = len(car_block)
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
        classical_result, _ = solve_classically(self.qubo)
        print(classical_result)
    
    def get_random_result(self):
        random_energy = CustomQAOA(operator,
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
    
    print(qubo.to_ising()[0])
    print(operator)
    # print()
    rqaoa = RQAOA(qubo, args["no_cars"], args["no_routes"])
    rqaoa.construct_initial_state()
    rqaoa.construct_mixer()
    rqaoa.solve_classically()


if __name__ == '__main__':
    main()
finish = time()
print("Time taken: {} s".format(finish - start))