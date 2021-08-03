from qiskit_optimization import QuadraticProgram
from generate_problem import main as generate_problem
from qiskit_optimization.converters import QuadraticProgramToQubo
from itertools import combinations
from time import time
import osmnx as ox
import numpy as np
import argparse

def main(raw_args=None):
    args = parse(raw_args)
    G, results = generate_problem(["-N "+str(args["no_cars"]), "-R "+str(args["no_routes"]), "-V"]) if args["visual"] == True else generate_problem(["-N "+str(args["no_cars"]), "-R "+str(args["no_routes"])])
    results = dict_routes(results, args)
    qubo, linear, quadratic = build_qubo_unconstrained(G, results)
    qubo = add_qubo_constraints(qubo, args["no_cars"], args["no_routes"])
    penalty_multiplier = args["penalty_multiplier"]
    qubo, max_coeff = convert_qubo(qubo, linear, quadratic, penalty_multiplier = penalty_multiplier)
    op, offset = qubo.to_ising()
    # op *= (1/max_coeff)
    max_coeff = 1.0
    # print(op)
    # print(max_coeff)
    # print(offset)
    return qubo, max_coeff, op, offset, linear, quadratic, results

def parse(raw_args):
    """Parse inputs for no_cars (number of cars),
                        no_routes (number of routes)               

    Args:
        raw_args (list): list of strings describing the inputs e.g. ["-N 3", "-R 3"]

    Returns:
        dict: A dictionary of the inputs as values, and the name of variables as the keys
    """

    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('Required arguments')
    requiredNamed.add_argument("--no_cars", "-N", required = True, help="Set the number of cars", type = int)
    requiredNamed.add_argument("--no_routes", "-R", required = True, help="Set the number of routes for each car", type = int)
    requiredNamed.add_argument("--penalty_multiplier", "-P", required = True, help="Set the penalty multiplier for QUBO constraint", type = float)
    parser.add_argument("--visual", "-V", default = False, help="Activate routes visualisation with '-V' ", action="store_true")
    args = parser.parse_args(raw_args)
    args = vars(args)
    return args


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
                linear[var] += G[edge[0]][edge[1]][0]['length']/1000
            else:
                linear[var] = G[edge[0]][edge[1]][0]['length']/1000

        #Quadratic terms
        for vars_couple in combinations(all_edges_dict[edge],2):
            if vars_couple in quadratic:
                quadratic[vars_couple] += 2*(G[edge[0]][edge[1]][0]['length']/1000)
            else:
                quadratic[vars_couple] = 2*(G[edge[0]][edge[1]][0]['length']/1000)
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
    start = time()
    qubo, max_coeff, op, offset, linear, quadratic, results = main()
    finish = time()
    print("Time taken: {}s".format(finish-start))