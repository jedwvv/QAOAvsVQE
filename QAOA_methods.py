import multiprocessing as mp
import numpy as np
from qiskit import QuantumCircuit
from generate_qubos import solve_classically, arr_to_str
import QAOAEx
from qiskit.algorithms import QAOA as QAOA_Base
from qiskit.algorithms.variational_algorithm import VariationalResult
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from qiskit.algorithms.optimizers import Optimizer
from time import time
import logging

logger = logging.getLogger(__name__)
class QAOA(QAOA_Base):
    _parameterise_point_for_energy_evaluation: Callable[[Union[List[float], np.ndarray], int], List[float]] = None
    latest_parameterised_point = None

    def set_parameterise_point_for_energy_evaluation(self,
                                                parameterise_point_for_optimisation: Callable[[Union[List[float], np.ndarray], int], List[float]]
                                                ) -> None:
        self._parameterise_point_for_energy_evaluation = parameterise_point_for_optimisation

    def find_minimum(self,
                     initial_point: Optional[np.ndarray] = None,
                     ansatz: Optional[QuantumCircuit] = None,
                     cost_fn: Optional[Callable] = None,
                     optimizer: Optional[Optimizer] = None,
                     gradient_fn: Optional[Callable] = None) -> 'VariationalResult':
        """Optimize to find the minimum cost value.

        Args:
            initial_point: If not `None` will be used instead of any initial point supplied via
                constructor. If `None` and `None` was supplied to constructor then a random
                point will be used if the optimizer requires an initial point.
            ansatz: If not `None` will be used instead of any ansatz supplied via constructor.
            cost_fn: If not `None` will be used instead of any cost_fn supplied via
                constructor.
            optimizer: If not `None` will be used instead of any optimizer supplied via
                constructor.
            gradient_fn: Optional gradient function for optimizer

        Returns:
            dict: Optimized variational parameters, and corresponding minimum cost value.

        Raises:
            ValueError: invalid input
        """
        initial_point = initial_point if initial_point is not None else self.initial_point
        ansatz = ansatz if ansatz is not None else self.ansatz
        cost_fn = cost_fn if cost_fn is not None else self._cost_fn
        optimizer = optimizer if optimizer is not None else self.optimizer

        if ansatz is None:
            raise ValueError('Ansatz neither supplied to constructor nor find minimum.')
        if cost_fn is None:
            raise ValueError('Cost function neither supplied to constructor nor find minimum.')
        if optimizer is None:
            raise ValueError('Optimizer neither supplied to constructor nor find minimum.')

        nparms = ansatz.num_parameters

        if hasattr(ansatz, 'parameter_bounds') and ansatz.parameter_bounds is not None:
            bounds = ansatz.parameter_bounds
        else:
            bounds = [(None, None)] * nparms

        if initial_point is not None and len(initial_point) != nparms:
            raise ValueError(
                'Initial point size {} and parameter size {} mismatch'.format(
                    len(initial_point), nparms))
        if len(bounds) != nparms:
            raise ValueError('Ansatz bounds size does not match parameter size')
        # If *any* value is *equal* in bounds array to None then the problem does *not* have bounds
        problem_has_bounds = not np.any(np.equal(bounds, None))
        # Check capabilities of the optimizer
        if problem_has_bounds:
            if not optimizer.is_bounds_supported:
                raise ValueError('Problem has bounds but optimizer does not support bounds')
        else:
            if optimizer.is_bounds_required:
                raise ValueError('Problem does not have bounds but optimizer requires bounds')
        if initial_point is not None:
            if not optimizer.is_initial_point_supported:
                raise ValueError('Optimizer does not support initial point')
        else:
            if optimizer.is_initial_point_required:
                if hasattr(ansatz, 'preferred_init_points'):
                    # Note: default implementation returns None, hence check again after below
                    initial_point = ansatz.preferred_init_points

                if initial_point is None:  # If still None use a random generated point
                    low = [(l if l is not None else -2 * np.pi) for (l, u) in bounds]
                    high = [(u if u is not None else 2 * np.pi) for (l, u) in bounds]
                    initial_point = algorithm_globals.random.uniform(low, high)

        start = time()
        if not optimizer.is_gradient_supported:  # ignore the passed gradient function
            gradient_fn = None
        else:
            if not gradient_fn:
                gradient_fn = self._gradient

        logger.info('Starting optimizer.\nbounds=%s\ninitial point=%s', bounds, initial_point)
        opt_params, opt_val, num_optimizer_evals = optimizer.optimize(nparms,
                                                                      cost_fn,
                                                                      variable_bounds=bounds,
                                                                      initial_point=initial_point,
                                                                      gradient_function=gradient_fn)
        eval_time = time() - start

        if self._parameterise_point_for_energy_evaluation != None:
            self.latest_parameterised_point = self._parameterise_point_for_energy_evaluation(opt_params, nparms)
        else:
            self.latest_parameterised_point = opt_params

        result = VariationalResult()
        result.optimizer_evals = num_optimizer_evals
        result.optimizer_time = eval_time
        result.optimal_value = opt_val
        result.optimal_point = self.latest_parameterised_point
        result.optimal_parameters = dict(zip(self._ansatz_params, self.latest_parameterised_point))

        return result

    def _energy_evaluation(self,
                           parameters: Union[List[float], np.ndarray]
                           ) -> Union[float, List[float]]:
        """Evaluate energy at given parameters for the ansatz.

        This is the objective function to be passed to the optimizer that is used for evaluation.

        Args:
            parameters: The parameters for the ansatz.

        Returns:
            Energy of the hamiltonian of each parameter.


        Raises:
            RuntimeError: If the ansatz has no parameters.
        """
        num_parameters = self.ansatz.num_parameters

        if self._parameterise_point_for_energy_evaluation != None:
            parameters = self._parameterise_point_for_energy_evaluation(parameters, num_parameters)
            
        if self._ansatz.num_parameters == 0:
            raise RuntimeError('The ansatz cannot have 0 parameters.')

        parameter_sets = np.reshape(parameters, (-1, num_parameters))
        # Create dict associating each parameter with the lists of parameterization values for it
        param_bindings = dict(zip(self._ansatz_params,
                                  parameter_sets.transpose().tolist()))  # type: Dict

        start_time = time()
        sampled_expect_op = self._circuit_sampler.convert(self._expect_op, params=param_bindings)
        means = np.real(sampled_expect_op.eval())

        if self._callback is not None:
            variance = np.real(self._expectation.compute_variance(sampled_expect_op))
            estimator_error = np.sqrt(variance / self.quantum_instance.run_config.shots)
            for i, param_set in enumerate(parameter_sets):
                self._eval_count += 1
                self._callback(self._eval_count, param_set, means[i], estimator_error[i])
        else:
            self._eval_count += len(means)

        end_time = time()
        logger.info('Energy evaluation returned %s - %.5f (ms), eval count: %s',
                    means, (end_time - start_time) * 1000, self._eval_count)

        return means if len(means) > 1 else means[0]


def CustomQAOA(operator, quantum_instance, optimizer, reps, **kwargs):
    initial_state = kwargs.get("initial_state", None)
    mixer = kwargs.get("mixer", None)
    construct_circ = kwargs.get("construct_circ", False)
    fourier_parametrise = kwargs.get("fourier_parametrise", False)
    qubo = kwargs.get("qubo", None) 
    solve = kwargs.get("solve", True)
    if quantum_instance.is_statevector:
        include_custom = True
    else:
        include_custom = False
    qaoa_instance = QAOAEx.QAOACustom(quantum_instance = quantum_instance,
                                        reps = reps,
                                        force_shots = False,
                                        optimizer = optimizer,
                                        qaoa_name = "example_qaoa",
                                        initial_state = initial_state,
                                        mixer = mixer,
                                        include_custom = include_custom,
                                        max_evals_grouped = 1
                                        )
    
    if solve:    
        if fourier_parametrise:
            qaoa_instance.set_parameterise_point_for_energy_evaluation(QAOAEx.convert_from_fourier_point)
        bounds = [(-np.pi, np.pi), (-np.pi, np.pi)]*reps
        
        if "list_points" in kwargs:
            list_points = kwargs["list_points"]
            list_results = []
            for point in list_points:
                result = qaoa_instance.solve(operator, point, bounds=bounds)
                qc = qaoa_instance.get_optimal_circuit() if construct_circ else None
                list_results.append( (result, qc) )
            qaoa_results, qc = min(list_results, key=lambda x: x[0].eigenvalue)
        else:
            initial_point = kwargs["initial_point"] if "initial_points" in kwargs\
                                                    else [ np.pi * (np.random.rand() - 0.5) for _ in range(2*reps) ]

            qaoa_results = qaoa_instance.solve(operator, initial_point, bounds=bounds)
            qc = qaoa_instance.get_optimal_circuit() if construct_circ else None
        if qubo:
            if fourier_parametrise:
                optimal_point = qaoa_results.optimal_point
                state = qaoa_instance.calculate_statevector_at_point(operator = operator, point = QAOAEx.convert_from_fourier_point(optimal_point, len(optimal_point)))
                qaoa_results.eigenstate = qaoa_instance.eigenvector_to_solutions(state, quadratic_program=qubo)
            else:
                optimal_point = qaoa_results.optimal_point
                state = qaoa_instance.calculate_statevector_at_point(operator = operator, point = optimal_point)
                qaoa_results.eigenstate = qaoa_instance.eigenvector_to_solutions(state, quadratic_program=qubo) 
        return qaoa_results, qc

    else:
        random_energy = qaoa_instance.evaluate_energy_at_point(operator, [0,0]*reps)
        return random_energy, None

#Using QISKIT QAOA
def QiskitQAOA(operator, quantum_instance, optimizer, reps, **kwargs):
    initial_state = kwargs.get("initial_state", None)
    mixer = kwargs.get("mixer", None)
    construct_circ = kwargs.get("construct_circ", False)
    fourier_parametrise = kwargs.get("fourier_parametrise", False)
    if quantum_instance.is_statevector:
        include_custom = True
    else:
        include_custom = False
    qaoa_instance = QAOA(quantum_instance = quantum_instance,
                        reps = reps,
                        optimizer = optimizer,
                        initial_state = initial_state,
                        mixer = mixer,
                        include_custom = include_custom,
                        max_evals_grouped = 1
                        )

    if fourier_parametrise:
        qaoa_instance.set_parameterise_point_for_energy_evaluation(QAOAEx.convert_from_fourier_point)

    #Set up QAOA Ansatz for computation of expectation values
    qaoa_instance.construct_expectation(parameter=[0]*(2*reps), operator=operator)
    
    #Bounds
    bounds = [(-2*np.pi, 2*np.pi)]*(2*reps) 
    qaoa_instance.ansatz._bounds = bounds

    if "list_points" in kwargs:
        list_points = kwargs["list_points"]
        list_results = []
        for point in list_points:
            qaoa_instance.initial_point = point
            result = qaoa_instance.compute_minimum_eigenvalue(operator)
            qc = qaoa_instance.get_optimal_circuit() if construct_circ else None
            list_results.append( (result, qc) )
        qaoa_results, qc = min(list_results, key=lambda x: x[0].eigenvalue)
        
    else:
        initial_point = kwargs.get("initial_point", None)
        qaoa_instance.initial_point = initial_point
        qaoa_results = qaoa_instance.compute_minimum_eigenvalue(operator)
        qc = qaoa_instance.get_optimal_circuit() if construct_circ else None
    return qaoa_results, qc


def generate_points(point, no_perturb, penalty):
    points = [point.copy() for _ in range(no_perturb+1)]
    for r in range(1,no_perturb+1):
        for i in range(len(point)):
            random_sign = np.random.choice([-1,1])
            if point[i] == 0:
                points[r][i] += penalty*0.05*random_sign
            else:
                noise = np.random.normal(0, (point[i]/np.pi)**2)
                points[r][i] += penalty*noise*random_sign
            if points[r][i] < -np.pi/2:
                points[r][i] = -np.pi/2+0.01
            if points[r][i] > np.pi/2:
                points[r][i] = np.pi/2-0.01

    return points

def eval_cost(x_str_arr, qubo_problem, no_qubits):
    if isinstance(x_str_arr, np.ndarray):
        cost = qubo_problem.objective.evaluate([int(x_str_arr[i]) for i in range(no_qubits)])
    else:
        cost = qubo_problem.objective.evaluate([int(x_str_arr[no_qubits-1-i]) \
                                                for i in range(no_qubits)]
                                            )
    return x_str_arr, cost

def get_costs(qubo):
    """[summary]

    Args:
        qubo ([type]): [description]
        no_qubits ([type]): [description]

    Returns:
        [type]: [description]
    """
    no_qubits = qubo.get_num_binary_vars()
    cpus = mp.cpu_count()
    pool = mp.Pool(cpus-1)
    params = ( ('0'*(no_qubits-len('{0:b}'.format(k)))+'{0:b}'.format(k),
            qubo,
            no_qubits
            ) for k in range(2**no_qubits))
    values = pool.starmap(eval_cost, params)
    values = dict(values)
    pool.close()
    pool.join()
    sort_values = sorted(values.items(), key=lambda x: x[1])
    return sort_values

def find_all_ground_states(qubo):
    classical_result, worst_result = solve_classically(qubo)
    x_arr = classical_result.x
    opt_value = classical_result.fval
    x_str = arr_to_str(x_arr)
    sort_values = get_costs(qubo)
    ground_energy = sort_values[0][1]
    x_s = [x_str]
    for i in range(1,10):
        if np.round(sort_values[i][1], 4) == np.round(ground_energy, 4):
            print("Other ground state(s) found: '{}'".format(sort_values[i][0]))
            x_s.append(sort_values[i][0])
    
    return x_s, opt_value, classical_result, worst_result


def count_coupling_terms(operator):
    """[summary]

    Args:
        operator ([type]): [description]
    """
    count = 0
    for op_i in operator.to_pauli_op().oplist:
        op_str = str(op_i.primitive)
        if op_str.count('Z') == 2:
            count += 1
    return count

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
        if interp_gammas[i] < -np.pi/2:
            interp_gammas[i] = -np.pi/2+0.01
        if interp_betas[i] < -np.pi/2:
            interp_betas[i] = -np.pi/2+0.01
        if interp_gammas[i] > np.pi/2:
            interp_gammas[i] = np.pi/2-0.01
        if interp_betas[i] > np.pi/2:
            interp_betas[i] = np.pi/2-0.01
    
    point = interp_gammas[1:p+2] + interp_betas[1:p+2]

    return point

def construct_initial_state(no_routes: int, no_cars: int):
    """[summary]

    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    R = no_routes
    N = no_cars
    w_one = QuantumCircuit(R)
    for r in range(0,R-1):
        if r == 0:
            w_one.ry(2*np.arccos(1/np.sqrt(R-r)),r)
        elif r != R-1:
            w_one.cry(2*np.arccos(1/np.sqrt(R-r)), r-1, r)
    for r in range(1,R):
        w_one.cx(R-r-1,R-r)
    w_one.x(0)

    #Tensor product for N cars
    initial_state = w_one.copy()
    for _ in range(0,N-1):
        initial_state = initial_state.tensor(w_one)

    initial_state.name = "U_init"

    return initial_state

def n_qbit_mixer(initial_state: QuantumCircuit):
    from qiskit.circuit.parameter import Parameter
    no_qubits = initial_state.num_qubits
    t = Parameter('t')
    mixer = QuantumCircuit(no_qubits)
    mixer.append(initial_state.inverse(), range(no_qubits))
    mixer.rz(2*t, range(no_qubits))
    mixer.append(initial_state, range(no_qubits))
    return mixer