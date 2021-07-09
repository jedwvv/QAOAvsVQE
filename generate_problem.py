import numpy as np 
import osmnx as ox
import argparse
import pickle as pkl
import multiprocessing as mp
import matplotlib.pyplot as plt

def main(raw_args=None):
    """Brings all methods together to return a traffic routing instance.

    Args:
        raw_args (list, optional): list of strings to describe arguments. Defaults to None, but function fails if required arguments are not included

    Returns:
        list: In position 0, returns the graph of problem, In position 1, a Numpy Array with shape "no_cars" by "no_routes" given by argument for the generated routes for each car with the route descibed as a sequence of nodes of the graph.
    """
    args = parse(raw_args)
    G = import_map("melbourne.pkl")
    cpus = mp.cpu_count()
    pool = mp.Pool(cpus)
    while True:
        cars_origin_destination = generate_start_finish_points(G, args["no_cars"])
        params = ( (G, start_destination, args["no_routes"]) for start_destination in list( cars_origin_destination.values() ) )
        results = pool.starmap(generate_routes, params)
        results = np.array(results, dtype = object)
        if results.shape == (args["no_cars"], args["no_routes"]): #If results returns "no_cars" x "no_routes" shape, then it is valid
            break
    pool.close()
    pool.join()

    if args["visual"]==True:
        visualise(G, results)

    return G, results


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
    parser.add_argument("--visual", "-V", default = False, help="Activate routes visualisation with '-V' ", action="store_true")
    args = parser.parse_args(raw_args)
    args = vars(args)
    return args

def import_map(filepath):
    """Import pre-made graph from pickle file

    Args:
        filepath (string): string of filepath for pickle file containing pre-made graph

    Returns:
        [NetworkX.MultiGraph]: the Networkx graph for the map# print(cars_origin_destination, "\n \n", shortest_routes)
    """

    with open(filepath, 'rb') as f:
        G = pkl.load(f)

    return G


def generate_start_finish_points(G, number_of_cars):
    """Use normal distribution to generate random 

    Args:
        G (NetworkX.Graph): NetworkX graph to generate 
        number_of_cars (int): [description]

    Returns:
        [type]: [description]
    """

    indices = np.random.normal(size=int(2*number_of_cars))
    stdev_multiply = 1.6 #Standard Deviation
    indices =  np.ceil(stdev_multiply * indices)
    maximum_index = np.max(indices)
    minimum_index = np.min(indices)

    no_nodes = len(G.nodes)
    start = np.random.randint(low=minimum_index, high=no_nodes-maximum_index)
    finish = np.random.randint(low=minimum_index, high=no_nodes-maximum_index)

    cars_start_dest = {}
    for i in range(number_of_cars):
        car_label = "car_{}".format(i)
        origin_node = list(G.nodes())[int(start + indices[i])]
        destination_node = list(G.nodes())[int(finish + indices[number_of_cars + i])]
        cars_start_dest[car_label] = (origin_node, destination_node)

    return cars_start_dest


def generate_routes(G, origin_destination, no_routes):
    """Generate "no_routes" number of routes from origin, destiniation node of G 

    Args:
        G (NetworkX Graph): the NetworkX graph of the map of the traffic routings
        origin_destination (tuple): (origin node, destination node) [Both nodes must be in G.nodes()]
        no_routes (int): number of routes to generate from origin to destination node

    Returns:
        [type]: [description]
    """
    orig, dest = origin_destination
    try:
        routes = list(ox.k_shortest_paths(G, orig, dest, k = no_routes, weight="length"))
        return routes
    except Exception:
        # for unsolvable routes (due to directed graph pecar_labelrimeter effects)
        return None

def visualise(G, results):
    # Visualise result #
    for i in range(len(results)):     
        routes = results[i]
        colors = np.array(["r", "y", "b", "g", "w", "gr", "o"]*2)[0:len(routes)]
        fig, ax = ox.plot_graph_routes(G, routes, route_colors=colors, route_linewidth=4, node_size=10)
    return fig, ax

if __name__ == "__main__":
    results = main()
    # print(results[1].shape)
    
