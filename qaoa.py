from QAOA_methods import CustomQAOA, construct_initial_state, n_qbit_mixer, find_all_ground_states
from QAOAEx import convert_to_fourier_point, convert_from_fourier_point
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit import Aer, QuantumCircuit, QuantumRegister
from classical_optimizers import NLOPT_Optimizer
from qiskit_optimization import QuadraticProgram, QiskitOptimizationError
from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops import PauliOp
from parser_all import parse
import numpy as np
import pickle as pkl
from copy import deepcopy
from time import time
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import depolarizing_error
import json

def main(args=None):
    start = time()
    if args == None:
        args = parse()
    
    #Load QUBO and reduce, then check existing classical solution
    with open('qubos_{}_car_{}_routes/qubo_{}.pkl'.format(args["no_cars"], args["no_routes"], args["no_samples"]), 'rb') as f:
        load_data = pkl.load(f)
        if len(load_data) == 5:
            qubo, max_coeff, operator, offset, routes = load_data
            classical_result = None
        else:
            qubo, max_coeff, operator, offset, routes, classical_result = load_data
    qubo, normalize_factor = reduce_qubo(qubo)
    if classical_result:
        classical_result._fval /= normalize_factor #Also normalize classical result
    
    #Noise
    if args["noisy"]:
        multiplier = args["multiplier"]
        print("Simulating with noise...Error mulitplier: {}".format(multiplier))
        with open('average_gate_errors.json', 'r') as f:
            noise_rates = json.load(f)
        noise_model = build_noise_model(noise_rates, multiplier)
    else:
        print("Simulating without noise...")
        noise_model = None

    #Initialize QAOA object
    qaoa = QAOA_Base(qubo = qubo,
                  no_cars = args["no_cars"],
                  no_routes = args["no_routes"],
                  symmetrise = args["symmetrise"],
                  customise = args["customise"],
                  classical_result = classical_result,
                  simulator = args["simulator"],
                  noise_model = noise_model,
                  opt_str = "LN_BOBYQA"
                 )
    p_max = args["p_max"]

    for p in range(1, p_max+1):
        p_start = time()
        print( "\nQAOA p={}. Results below:".format(p) )
        qaoa.optimizer.set_options(maxeval=100*(2**p))
        if p==1:
            qaoa.solve_qaoa(p)
        else:
            qaoa.solve_qaoa(p, tqa=True)
        p_end = time()
        print("Time taken (for this iteration): {}s".format(p_end - p_start))
        
    
    print( "\nProbabilities: {}".format(qaoa.prob_s) )
    print( "Eigenvalue at each p: {}".format(qaoa.eval_s) )
    print( "Approx Qualities of most_probable_state: {}\n".format(qaoa.approx_s) )
    if args["noisy"]:
        if args["symmetrise"]:
            filedir = 'results_{}cars{}routes/Noisy_QAOA_Symm_{}_Cust_p={}_error{}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"], args["multiplier"])
        else:
            filedir = 'results_{}cars{}routes/Noisy_QAOA_Reg_{}_Cust_p={}_error{}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"], args["multiplier"])
    else:
        if args["symmetrise"]:
            filedir = 'results_{}cars{}routes/Ideal_QAOA_Symm_{}_Cust_p={}_error{}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"], args["multiplier"])
        else:
            filedir = 'results_{}cars{}routes/Ideal_QAOA_Reg_{}_Cust_p={}_error{}.csv'.format(args["no_cars"], args["no_routes"], args["no_samples"], args["p_max"], args["multiplier"])
    
    if not args["customise"]:
        filedir = filedir.replace("Cust", "Base")
    
    #Save results to file
    save_results = np.append( qaoa.prob_s, qaoa.eval_s )
    save_results = np.append( save_results, qaoa.approx_s )
    with open(filedir, 'w') as f:
        np.savetxt(f, save_results, delimiter=',')

    result_saved_string = "Results saved in {}".format(filedir)
    print(result_saved_string)
    finish = time()
    print("\nTime taken: {} s".format(finish - start))
    print("_"*len(result_saved_string))  

def reduce_qubo(qubo):
    #Perform reduction as many times possible up to some threshold
    max_coeff = np.max( np.append( qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array() ) )
    total_normalize_factor = 1.0 #To recover original qubo by multiplying this number to its objective
    qubo.objective.linear._coefficients = qubo.objective.linear._coefficients / max_coeff
    qubo.objective.quadratic._coefficients = qubo.objective.quadratic._coefficients / max_coeff
    qubo.objective.constant = qubo.objective.constant / max_coeff
    total_normalize_factor *= max_coeff
    return qubo, total_normalize_factor

def build_noise_model(noise_rates, multiplier):
    noise_model = NoiseModel(basis_gates = ['cx', 'id', 'reset', 'rz', 'sx', 'x'])
    gate_errors = {x[0]: x[1]*multiplier for x in noise_rates.items()}
    for gate in noise_rates:
        #Depolarizing error on CX
        if gate == 'cx':
            depol_err_cx = depolarizing_error(noise_rates[gate] * multiplier, 2)
            noise_model.add_all_qubit_quantum_error(depol_err_cx, ['cx'])
        #Depolarizing error on single qubit gates
        else:
            depol_err = depolarizing_error(noise_rates[gate] * multiplier, 1, standard_gates = False) 
            noise_model.add_all_qubit_quantum_error(depol_err, [gate])
    
    #Save error rates
    with open("gate_errors_{}.json".format(multiplier), "w") as f:
        json.dump(gate_errors, f)
    
    return noise_model

class QAOA_Base:
    def __init__(self, **kwargs):
        self.qubo = kwargs.get("qubo", None)
        if self.qubo is not None:
            self.operator, self.offset = self.qubo.to_ising()
        self.no_cars = kwargs.get("no_cars", 0)
        self.no_routes = kwargs.get("no_routes", 0)
        self.symmetrise = kwargs.get("symmetrise", False)
        self.customise = kwargs.get("customise", False)
        opt_str = kwargs.get('opt_str', "LN_COBYLA")
        print("Optimizer: {}".format(opt_str))
        self.optimizer = NLOPT_Optimizer(opt_str)
        self.optimizer.set_options(maxeval = 100)
        self.original_qubo = deepcopy(self.qubo)

        #Benchmarking, using classical result for ground state energy, and then random energy measurement.
        self.classical_result = kwargs.get("classical_result", None)
        self.solve_classically()
        self.random_instance = QuantumInstance(backend = Aer.get_backend("aer_simulator_matrix_product_state"), shots = 1000) #1000 randomly measured 
        self.get_random_energy()

        #Simulation methods
        simulator = kwargs.get("simulator", None)
        noise_model = kwargs.get("noise_model", None)
        if simulator == None:
            simulator = "aer_simulator_density_matrix"
        print("Using "+simulator)
        
        self.quantum_instance = QuantumInstance( backend = Aer.get_backend(simulator), 
                                                 shots = 8192, 
                                                 noise_model = noise_model,
                                                 basis_gates = ["cx", "x", "sx", "rz", "id"]
                                               )
        
        #Symmetrise
        if self.symmetrise:
            self.symmetrise_qubo()

        #Customise QAOA
        if self.customise:
            self.construct_initial_state(symmetrise = self.symmetrise)
            self.construct_mixer()
        else:
            self.initial_state = None
            self.mixer = None
        
        #Results placeholder
        self.prob_s = []
        self.eval_s = []
        self.approx_s = []

    def symmetrise_qubo(self):
            new_operator = []
            operator, _ = self.qubo.to_ising()
            for op_1 in operator:
                coeff, op_1 = op_1.to_pauli_op().coeff, op_1.to_pauli_op().primitive
                op_1_str = op_1.to_label()
                Z_counts = op_1_str.count('Z')
                if Z_counts == 1:
                    op_1_str = "Z" + op_1_str  #Add a Z in the last qubit to single Z terms
                else:
                    op_1_str = "I" + op_1_str #Add an I in the last qubit to ZZ terms (no change in operator)
                pauli = PauliOp( primitive = Pauli(op_1_str), coeff = coeff )
                new_operator.append(pauli)
            symmetrised_qubo = QuadraticProgram()
            symmetrised_operator = sum(new_operator)
            symmetrised_qubo.from_ising(symmetrised_operator, self.offset, linear=True)
            self.qubo = symmetrised_qubo
            self.rename_qubo_variables()
            self.operator, self.offset = self.qubo.to_ising()
    
    def rename_qubo_variables(self):
        original_qubo = self.original_qubo
        qubo = self.qubo
        variable_names = [ variable.name for variable in qubo.variables ]
        original_variable_names = [ (variable.name, 1) for variable in original_qubo.variables ]
        new_variable_names = original_variable_names + [("X_anc", 1)] if self.symmetrise else original_variable_names
        variables_dict = dict(zip(variable_names, new_variable_names))
        for new_variable_name in variables_dict.values():
            qubo.binary_var(name = new_variable_name[0])
        qubo = qubo.substitute_variables(variables = variables_dict)
        if qubo.status == QuadraticProgram.Status.INFEASIBLE:
            raise QiskitOptimizationError('Infeasible due to variable substitution')
        self.qubo = qubo

    def get_random_energy(self):
        #Get random benchmark energy for 0 layer Custom-QAOA (achieved by using layer 1 Cust-QAOA with [0,0] angles i.e. sampling from feasible states with equal prob)
        self.construct_initial_state(symmetrise=False)
        self.construct_mixer()
        random_energy, _ = CustomQAOA(operator = self.operator,
                    quantum_instance = self.random_instance,
                    optimizer = self.optimizer,
                    reps = 1,
                    initial_state = self.initial_state,
                    mixer = self.mixer,
                    solve = False,
                    )
        #Remove custom initial state if using BASE QAOA
        self.initial_state = None
        self.mixer = None
        temp = random_energy
        self.random_energy = temp + self.offset
        if np.round( self.random_energy - self.opt_value, 6 ) < 1e-7:
            print("0 layer QAOA converged to exact solution. Shifting value up by |exact_ground_energy| instead to avoid dividing by 0 in approx quality.")
            self.random_energy += np.abs(self.random_energy)
        self.benchmark_energy = self.random_energy
        print("random energy: {}\n".format(self.random_energy))

    def construct_initial_state(self, **kwargs):
        symmetrise = kwargs.get("symmetrise", False)
        self.initial_state = construct_initial_state(self.no_routes, self.no_cars)
        if symmetrise:
            ancilla_reg = QuantumRegister(1, 'ancilla')
            self.initial_state.add_register(ancilla_reg)
            self.initial_state.h(ancilla_reg[0]) 

    def construct_mixer(self):
        self.mixer = n_qbit_mixer(self.initial_state)
    
    def solve_classically(self):
        if self.classical_result:
            print("There is an existing classical result. Using this as code proceeds.")
            print(self.classical_result)
            self.opt_value = self.classical_result.fval
        else:
            print("No classical result already available.")
            print("Now solving classically")
            _, opt_value, classical_result, _ = find_all_ground_states(self.original_qubo)
            self.classical_result = classical_result
            self.opt_value = opt_value

    def solve_qaoa(self, p, **kwargs):
        point = kwargs.get("point", None)
        fourier_parametrise = True
        tqa = kwargs.get('tqa', False)
        points = kwargs.get("points", None)
        construct_circ = False
                
        if tqa:
            deltas = np.arange(0.25, 0.91, 0.05)
            point = np.append( [ (i+1)/p for i in range(p) ] , [ 1-(i+1)/p for i in range(p) ] )
            points = [delta*point for delta in deltas]
            fourier_parametrise = True
            if fourier_parametrise:
                points = [ convert_to_fourier_point(point, len(point)) for point in points ]
            qaoa_results, _ = CustomQAOA( self.operator,
                                            self.quantum_instance,
                                            self.optimizer,
                                            reps = p,
                                            initial_state = self.initial_state,
                                            mixer = self.mixer,
                                            fourier_parametrise = fourier_parametrise,
                                            list_points = points,
                                            qubo = self.qubo,
                                            construct_circ = construct_circ
                                                        )
        elif points is not None:
            fourier_parametrise = True
            if fourier_parametrise:
                points = [ convert_to_fourier_point(point, len(point)) for point in points ]
            if point is not None:
                points.append( convert_to_fourier_point(point, len(point)) )
            qaoa_results, _ = CustomQAOA( self.operator,
                                                        self.quantum_instance,
                                                        self.optimizer,
                                                        reps = p,
                                                        initial_state = self.initial_state,
                                                        mixer = self.mixer,
                                                        fourier_parametrise = fourier_parametrise,
                                                        list_points = points,
                                                        qubo = self.qubo,
                                                        construct_circ = construct_circ
                                                        )
        elif point is not None:
            fourier_parametrise = True
            if fourier_parametrise:
                point =  convert_to_fourier_point(point, len(point))
            qaoa_results, _ = CustomQAOA( self.operator,
                                        self.quantum_instance,
                                        self.optimizer,
                                        reps = p,
                                        initial_state = self.initial_state,
                                        initial_point = point,
                                        mixer = self.mixer,
                                        fourier_parametrise = fourier_parametrise,
                                        qubo = self.qubo,
                                        construct_circ = construct_circ
                                        )
        else:
            points = [ [0]*(2*p) ] + [ [ 1.98 * np.pi* ( np.random.rand() - 0.5 ) for _ in range(2*p)] for _ in range(10) ]
            fourier_parametrise = True
            qaoa_results, _ = CustomQAOA( self.operator,
                                        self.quantum_instance,
                                        self.optimizer,
                                        reps = p,
                                        initial_state = self.initial_state,
                                        list_points = points,
                                        mixer = self.mixer,
                                        fourier_parametrise = fourier_parametrise,
                                        qubo = self.qubo,
                                        construct_circ = construct_circ
                                        )
        point = qaoa_results.optimal_point
        eigenvalue = sum( [ x[1] * x[2] for x in qaoa_results.eigenstate ] )
        qaoa_results.eigenvalue = eigenvalue
        self.optimal_point = convert_to_fourier_point(point, len(point)) if fourier_parametrise else point
        self.qaoa_result = qaoa_results

        #Sort states by decreasing probability
        sorted_eigenstate_by_prob = sorted(qaoa_results.eigenstate, key = lambda x: x[2], reverse = True)
        
        #print sorted state in a table
        self.print_state(sorted_eigenstate_by_prob)

        #Other print stuff
        print("Eigenvalue: {}".format(eigenvalue))
        print("Optimal point: {}".format(qaoa_results.optimal_point))
        print("Optimizer Evals: {}".format(qaoa_results.optimizer_evals))
        scale = self.random_energy - self.opt_value

        approx_quality_2 = np.round( ( self.random_energy - sorted_eigenstate_by_prob[0][1] ) / scale, 3 )
        energy_prob = {}
        for x in qaoa_results.eigenstate:
            energy_prob[ np.round(x[1], 6) ] = energy_prob.get(np.round(x[1], 6), 0) + x[2]
        prob_s = np.round( energy_prob.get(np.round(self.opt_value, 6), 0), 6 )
        self.prob_s.append( prob_s )
        self.eval_s.append( eigenvalue )
        self.approx_s.append( approx_quality_2 )
        print( "\nQAOA most probable solution: {}".format(sorted_eigenstate_by_prob[0]) )
        print( "Approx_quality: {}".format(approx_quality_2) ) 

    def print_state(self, eigenstate):
            header = '|'
            for var in self.qubo.variables:
                var_name = var.name
                header += '{:<5}|'.format(var_name.replace("_", ""))
            header += '{:<6}|'.format('Cost')
            header += '{:<5}|'.format("Prob")
            print("-"*len(header))
            print(header)
            print("-"*len(header))
            
            count = 0
            for item in eigenstate:
                string = '|'
                for binary_var in item[0]:
                    string += '{:<5} '.format(binary_var) #Binary string
                string += '{:<6} '.format(np.round(item[1], 2)) #Cost
                string += '{:<5}|'.format(np.round(item[2], 3)) #Prob
                print(string)
                
                #Print only first 20 states of lowest energy
                count += 1
                if count == 20: 
                    break
            print("-"*len(header))

if __name__ == "__main__":
    main()