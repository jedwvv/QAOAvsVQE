from qiskit_optimization import QuadraticProgram
from generate_problem import main as generate_problem
from generate_problem import import_map
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.algorithms.minimum_eigen_solvers import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from itertools import combinations
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
import pickle as pkl
from tqdm import tqdm
from parser_all import parse

def main(args = None):
    #Generate and save valid qubos
    if args == None:
        args = parse()

    print("Arguments: {}".format(args))
    print("Now generating QUBOs with {} cars {} routes".format(args["no_cars"], args["no_routes"]))
    for qubo_no in tqdm(range(args["no_samples"])):
        generate_valid_qubo(args, qubo_no)
    

def generate_valid_qubo(args, qubo_no):
    infeasible = True
    while infeasible: #Generate valid QUBO
        try:
            qubo, max_coeff, operator, offset, _, _, results = generate_qubo(args)

            #Classically solve generated QUBO
            classical_result = solve_classically(qubo)
            #Check solution
            variables_dict = classical_result.variables_dict
            routes = filter_solutions(results, variables_dict, args["no_cars"])
            check_solutions(routes)
            
            break #break loop if valid solution, else increase penalty in exception below.
        
        except TypeError:
            #QUBO has invalid solution, start again with increased penalty.
            args["penalty_multiplier"] += 0.05 #increase penalty for more likelihood to generate valid QUBO
    
    if args["visual"]:
        graph = import_map('melbourne_2.pkl')
        visualise_solution(graph, routes)

    #Save generated valid qubo
    with open('qubos_{}_car_{}_routes/qubo_{}.pkl'.format(args["no_cars"], args["no_routes"], qubo_no), 'wb') as f:
        pkl.dump([qubo, max_coeff, operator, offset, routes],f)

def generate_qubo(args):
    G, all_routes = generate_problem(args)
    routes_dictionary = dict_routes(all_routes, args)
    qubo, linear, quadratic = build_qubo_unconstrained(G, routes_dictionary)
    qubo = add_qubo_constraints(qubo, args["no_cars"], args["no_routes"])
    penalty_multiplier = args["penalty_multiplier"]
    # print("Before: ", qubo)
    qubo, max_coeff = convert_qubo(qubo, linear, quadratic, penalty_multiplier = penalty_multiplier)
    # qubo.objective *= 2
    # print("After: ", qubo)
    op, offset = qubo.to_ising()
    # print("Operator:", op)
    pauli_coeffs = [operator.to_pauli_op().coeff for operator in op]
    max_coeff = np.max(np.abs(pauli_coeffs))
    max_coeff = 1.0
    # op *= 1/max_coeff
    return qubo, max_coeff, op, offset, linear, quadratic, routes_dictionary

def check_solutions(routes):
    """[summary]

    Args:
        routes ([type]): [description]

    Raises:
        TypeError: [description]
    """
    try:
        len(routes)
        # print("This is a valid solution")
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
            # print("Error, function returned non-matching variables")
            return None
        elif var_route[0] in variables:
            # print("Error: Solution found two routes for one car")
            return None
    if len(variables) != no_cars:
        # print("Error: At least one car did not have a valid route")
        return None
    return routes

def dict_routes(results, args):
    """Re-arranges array of all_routes into dictionary with labels for each car's routes

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
                linear[var] += np.ceil(G[edge[0]][edge[1]][0]['length']/1000)
                # print(linear[var])
            else:
                linear[var] = np.ceil(G[edge[0]][edge[1]][0]['length']/1000)
                # print(linear[var])
        #Quadratic terms
        for vars_couple in combinations(all_edges_dict[edge],2):
            if vars_couple in quadratic:
                quadratic[vars_couple] += 2*np.ceil(G[edge[0]][edge[1]][0]['length']/1000)
                # print(quadratic[vars_couple])
            else:
                quadratic[vars_couple] = 2*np.ceil(G[edge[0]][edge[1]][0]['length']/1000)
                # print(quadratic[vars_couple])
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
    max_coeff = np.max(np.abs(coefficients))
    penalty = np.floor( max_coeff * penalty_multiplier )
    converter = QuadraticProgramToQubo(penalty = penalty)
    qubo = converter.convert(qubo)
    return qubo, max(penalty, max_coeff)

def visualise_solution(graph, routes):
    """[summary]

    Args:ca
        graph ([type]): [description]
        routes ([type]): [description]

    Returns:
        [type]: [description]
    """
    colors = np.array(["r", "y", "b", "g", "w"]*10)[0:len(routes)]
    fig, axes = ox.plot_graph_routes(graph,
                                    routes,
                                    route_colors=colors,
                                    route_linewidth=4,
                                    node_size=10,
                                    show = False,
                                    close = False
                                    )
    fig.suptitle('Optimal routes'.format())
    plt.show()
    return fig, axes

if __name__ == "__main__":
    main()