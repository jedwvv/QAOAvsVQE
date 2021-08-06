from qiskit_optimization import QuadraticProgram
from generate_problem import main as generate_problem
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.algorithms.minimum_eigen_solvers import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from itertools import combinations
from time import time
import osmnx as ox
import numpy as np
import pickle as pkl
from parser_all import parse

def main(raw_args = None):
    #Generate and save valid qubos
    args = parse(raw_args)
    for qubo_no in range(args["no_samples"]):
        qubo, max_coeff, operator, offset, routes = generate_valid_qubo(args)
        with open('qubos_{}_car_{}_routes/qubo_{}.pkl'.format(args["no_cars"], args["no_routes"], qubo_no), 'wb') as f:
            pkl.dump([qubo, max_coeff, operator, offset, routes],f)
    

def generate_valid_qubo(args):
    while True: #Generate valid QUBO
        try:
            #Generate a QUBO
            if args["visual"] is True:
                qubo, max_coeff, operator, offset, _, _, results = \
                    generate_qubo( [
                                    "-N "+str(args["no_cars"]),
                                    "-R "+str(args["no_routes"]),
                                    "-P "+str(args["penalty_multiplier"]),
                                    "-V"
                                    ]
                                )
            else:
                qubo, max_coeff, operator, offset, _, _, results = \
                    generate_qubo( [
                                    "-N "+str(args["no_cars"]),
                                    "-R "+str(args["no_routes"]),
                                    "-P "+str(args["penalty_multiplier"])
                                    ]
                                )
            #Classically solve generated QUBO
            classical_result = solve_classically(qubo)

            #Check solution
            variables_dict = classical_result.variables_dict
            routes = filter_solutions(results, variables_dict, args["no_cars"])
            check_solutions(routes)

            #If valid solution, save the solution.
            x_arr = classical_result.x
            x_str = arr_to_str(x_arr)

            break #break loop if valid solution, else increase penalty.

        except TypeError:
            #QUBO has invalid solution, start again with increased penalty.
            print("Starting with new qubo")
            # args["penalty_multiplier"] += 0.05 #increase penalty for more likelihood to generate valid QUBO
    return qubo, max_coeff, operator, offset, routes

def generate_qubo(raw_args=None):
    args = parse(raw_args)
    G, results = generate_problem(["-N "+str(args["no_cars"]), "-R "+str(args["no_routes"]), "-V"]) if args["visual"] == True else generate_problem(["-N "+str(args["no_cars"]), "-R "+str(args["no_routes"])])
    results = dict_routes(results, args)
    qubo, linear, quadratic = build_qubo_unconstrained(G, results)
    qubo = add_qubo_constraints(qubo, args["no_cars"], args["no_routes"])
    penalty_multiplier = args["penalty_multiplier"]
    qubo, max_coeff = convert_qubo(qubo, linear, quadratic, penalty_multiplier = penalty_multiplier)
    op, offset = qubo.to_ising()
    op *= (1/max_coeff)
    return qubo, max_coeff, op, offset, linear, quadratic, results

def check_solutions(routes):
    """[summary]

    Args:
        routes ([type]): [description]

    Raises:
        TypeError: [description]
    """
    try:
        len(routes)
        print("This is a valid solution")
    except TypeError as typ:
        raise TypeError("This is not a valid solution") from typ

def solve_classically(qubo):
    """[summary]

    Args:
        qubo ([type]): [description]

    Returns:
        [type]: [description]
    """
    exact_mes = NumPyMinimumEigensolver()
    exact = MinimumEigenOptimizer(exact_mes)
    exact_result = exact.solve(qubo)
    print(exact_result)
    return exact_result

def arr_to_str(x_arr):
    """[summary]

    Args:
        x_arr ([type]): [description]

    Returns:
        [type]: [description]
    """
    x_arr = x_arr[::-1]
    string = ''.join(str(int(x_i)) for x_i in x_arr)
    return string

def filter_solutions(results, qaoa_dict, no_cars):
    """[summary]

    Args:
        results ([type]): [description]
        qaoa_dict ([type]): [description]
        no_cars ([type]): [description]

    Returns:
        [type]: [description]
    """
    routes = []
    variables = []
    for var_route, var_value in zip( results.items(), qaoa_dict.items() ):
        if var_route[0] == var_value[0] and int(var_value[1]) == 1 \
            and var_route[0] not in variables:
            routes.append(var_route[1])
            variables.append(var_route[0])
        elif var_route[0] != var_value[0]:
            print("Error, function returned non-matching variables")
            return None
        elif var_route[0] in variables:
            print("Error: Solution found two routes for one car")
            return None
    if len(variables) != no_cars:
        print("Error: At least one car did not have a valid route")
        return None
    return routes

def dict_routes(results, args):
    """Re-arranges array of results into dictionary with labels for each car's routes

    Args:
        results ([np.ndarray]): a numpy array of shape no_cars*no_routes passed as the main function's arguments --no_cars/-N, --no_routes/-R

    Returns:
        [dict]: Returns the routes but as a dictionary with keys as the car route label, and the value as the node sequence routing.
    """
    results = np.reshape(results, args["no_cars"]*args["no_routes"])
    results_dict = { "X_{}_{}".format(int((i-i%args["no_routes"])/args["no_routes"]),i%args["no_routes"]): result for i, result in enumerate(results) }
    return results_dict

def build_qubo_unconstrained(G, routes_dict):
    all_edges_dict = get_edges_dict(routes_dict)
    qubo = QuadraticProgram()
    linear, quadratic = get_linear_quadratic_coeffs(G, all_edges_dict)
    for var in routes_dict.keys():
        qubo.binary_var(var)
    qubo.minimize(linear = linear, quadratic=quadratic)
    return qubo, linear, quadratic

def get_linear_quadratic_coeffs(G, all_edges_dict):
    linear = {}
    quadratic = {}
    for edge in all_edges_dict:
        #Linear terms
        for var in all_edges_dict[edge]:
            if var in linear:
                linear[var] += G[edge[0]][edge[1]][0]['length']**2/1000
            else:
                linear[var] = G[edge[0]][edge[1]][0]['length']**2/1000

        #Quadratic terms
        for vars_couple in combinations(all_edges_dict[edge],2):
            if vars_couple in quadratic:
                quadratic[vars_couple] += 2*(G[edge[0]][edge[1]][0]['length']**2/1000)
            else:
                quadratic[vars_couple] = 2*(G[edge[0]][edge[1]][0]['length']**2/1000)
    return linear, quadratic

def get_edges_dict(routes_dict):
    all_edges_dict = {}
    for var, route in routes_dict.items():
        edge_list = get_edge_list(route)
        for edge in edge_list:
            if edge not in all_edges_dict:
                all_edges_dict[edge] = [var]
            else:
                all_edges_dict[edge].append(var)
    return all_edges_dict

def get_edge_list(nodes_list):
    """Returns edge_list from sequence of nodes

    Args:
        nodes_list ([list]): List of sequence of nodes that appear in a graph

    Returns:
        [list]: List of edges (edge = 2-tuple of nodes)
    """
    edge_list = [ (nodes_list[i], nodes_list[i+1]) for i in range(len(nodes_list)-1)]

    return edge_list

def add_qubo_constraints(qubo, no_cars, no_routes):
    for i in range(no_cars):
        qubo.linear_constraint(linear = {"X_{}_{}".format(i, j):1 for j in range(no_routes)}, sense= "==", rhs = 1, name = "Car_{}".format(i) )
    return qubo

def convert_qubo(qubo, linear, quadratic, penalty_multiplier = 1):
    coefficients = np.append(list(linear.values()), list(quadratic.values()))
    max_coeff = np.max(abs(coefficients))
    penalty = max_coeff * penalty_multiplier
    converter = QuadraticProgramToQubo(penalty = penalty)
    qubo = converter.convert(qubo)
    return qubo, max_coeff

if __name__ == "__main__":
    main(["-N 3", "-R 3", "-P 1.5", "-OSLSQP", "-M 5", "-T 5", "-S 100"])