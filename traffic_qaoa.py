import pickle as pkl
import json
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations
from copy import deepcopy
from generate_qubos import get_edges_dict, build_qubo_unconstrained_from_edges_dict, convert_qubo
from time import time
from QAOA_methods import QiskitQAOA
from qiskit.algorithms.optimizers import NELDER_MEAD
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit import Aer, QuantumCircuit, QuantumRegister

with open('unimelb_2.pkl', 'rb') as f:
    G = pkl.load(f)
with open("coupling_map.pkl", "rb") as f:
    coupling_map = pkl.load(f)
with open("qubit_mapping.json", "r") as f:
    qubit_mapping = json.load(f)

global mapping

class Ox_Route:
    def __init__(self, routing: list):
        self._routing = routing
    
    def compose(self, ox_route_2: list):
        """Compose self with another route(list) provided self has terminal node is same with the 2nd routing's initial node. 

        Args:
            ox_route_2 ([list]): 2nd route to compose self with.
        """
        if self._routing[::-1][0] == ox_route_2[0]:
            self._routing.remove(self._routing[::-1][0]) #Remove last node otherwise it will be repeated.
            self._routing += ox_route_2
        else:
            raise Exception('The 2nd route does not start at the same node ({}) at which the 1st terminates ({})'.format(ox_route_2[0], self._routing[::-1][0]))
        return self
    
    def remove(self, ox_route_2: list):
        """Removes a segment of self.routing() and returns the removed segment as another Ox_Route. 
            The removed segment must be the last segment of the original route(so that it can be split off).
            The new routing will end at the starting point of the removed route.
            Inverse of self.compose() method.

        Args:
            ox_route_2 ([list]): Inner route to remove from self.
    
        Returns:
            self: Returns the new route with removed inner-route
        """
        #Index of sub-route within route:
        idx = sublist(ox_route_2, self.routing()) #Idx = 0 for empty route to be removed, and -1 if subroute does not exist.
        if idx > 0 and idx + len(ox_route_2) == len(self.routing()):
            new_routing = self._routing
            self._routing = new_routing[0:idx+1]
            return self
        else:
            raise Exception("Route cannot be removed because it either isn't a subroute or it doesn't end in the same node")

    def orig_dest(self):
        routing = self._routing
        return routing[0], routing[::-1][0]

    def obtain_last_k_nodes(self, k: int):
        """obtain last k nodes of a route as a route.

        Args:
            k (int): number of nodes to obtain.

        Returns:
            Ox_Route: Ox_Route of last k-nodes
        """
        route_length = len(self._routing)
        routing = self._routing[route_length - k: route_length]
        return Ox_Route(routing)
    
    def obtain_list_of_edges(self):
        """Obtain routing as a list of edges(2-tuple of nodes) instead of list of subsequent nodes. e.g [A, B, C] -> [(A, B), (B, C)]

        Returns:
            [list]: list of 2-tuples.
        """
        list_of_edges = [ (self._routing[i], self._routing[i+1]) for i in range(len(self._routing)-1) ]

        return list_of_edges
    
    def get_common_edges(self, other):
        """Checks whether self routing has common edges with other routing.

        Args:
            other ([Ox_Route]): other routing to compare edges with

        Returns:
            [bool]: Whether or not self has common edges with other
        """ 
        overlapping_edges = []
        for edge in self.obtain_list_of_edges():
            if edge in other.obtain_list_of_edges():
                overlapping_edges += [edge]
        if len(overlapping_edges) == 0:
            return None
        else:
            return overlapping_edges

        return False

    def __add__(self, ox_route_2):
        #Same as compose method but with ox_route_2 as another Ox_Route class
        return self.compose(ox_route_2.routing())

    def __sub__(self, ox_route_2):
        #Same as remove method but with ox_route_2 as another Ox_Route class
        return self.remove(ox_route_2.routing())
    #Represent object by its route.
    def __repr__(self) -> list:
        return self._routing

    #String representation of self.
    def __str__(self) -> str:
        return str(self.__repr__())

    #Getter for self._routing
    def routing(self):
        return self._routing
class QAOA_Traffic:
    def __init__(self, no_cars, no_routes, G_map = None, coupling_map = None, qubit_mapping = None):
        self._map = G_map
        self._no_cars = no_cars
        self._no_routes = no_routes
        self._coupling_map = coupling_map
        self._qubit_mapping = qubit_mapping
        if self._qubit_mapping:
            self.compute_qubit_neighbours()
        self.set_variables()

    def set_variables(self):
        variables = ["Car_{}_Route_{}".format(str(u).zfill(2),v) for u,v in product(range(self._no_cars), range(self._no_routes))]
        self._variables = variables
        routing_dict = dict(zip(variables, [None for _ in range(len(variables))]))
        self._routing_dict = routing_dict
    
    def set_map(self, G_map):
        self._map = G_map
    
    def set_coupling_map(self, coupling_map):
        self._coupling_map = coupling_map
    
    def set_qubit_mapping(self, qubit_mapping):
        self._qubit_mapping = qubit_mapping
        self.compute_qubit_neighbours()
    
    def compute_qubit_neighbours(self):
        #Compute up to 2nd qubit neighbours.
        coupling_map = nx.Graph()
        coupling_map.add_edges_from(self._coupling_map)
        temp_qubit_neighbours = {}
        qubit_neighbours = {}
        for qubit in range(max(coupling_map)+1):
            neighbouring_qubits = set(coupling_map.neighbors(qubit))
            temp_qubit_neighbours[qubit] = neighbouring_qubits
            qubit_neighbours[qubit] = neighbouring_qubits
        for qubit in range(max(coupling_map)+1):
            for first_neighbour in temp_qubit_neighbours[qubit]:
                second_neighbours = temp_qubit_neighbours[first_neighbour]
                qubit_neighbours[qubit] = qubit_neighbours[qubit] | second_neighbours
            qubit_neighbours[qubit].remove(qubit)
        self._qubit_neighbours = qubit_neighbours
        return qubit_neighbours

    def generate_k_routes_from_clicks(self, k):
        G = self._map
        fig, ax = ox.plot_graph(G, figsize = (8,8), bgcolor='white', edge_color='black', edge_linewidth=0.5, show=False)
        fig.suptitle("Press X while hovering desired nodes in order of (orig, dest).")
        self._temp_nodes = []
        def append_two_nodes_onclick(event):
            nodes = self._temp_nodes
            if event.key == "x" or event.key == "X":
                x, y = (event.ydata, event.xdata)
                node = ox.nearest_nodes(G, y, x)
                nodes.append(node)
            elif event.key != "q" and event.key != "Q":
                raise Exception("Press X when selecting nodes. Try again")
            if len(nodes) == 2:
                plt.close()
        cid = fig.canvas.mpl_connect('key_press_event', append_two_nodes_onclick)
        plt.show()
        try:
            routes = list(ox.k_shortest_paths(G, *self._temp_nodes, k))
        except:
            routes = None
            raise Exception("No valid routes between two nodes")
        if routes and len(routes) > 1:
            return routes
        elif routes and len(routes) == 1:
            return routes[0]

    def generate_k_routes_from_random_nodes(self, k, seed=123):
        rng = np.random.default_rng(seed=seed)
        G = self._map
        high = len(G.nodes()) #max index for node
        start_idx , dest_idx = rng.integers(low=0, high=high, size=2)
        start, dest = (list(G.nodes())[start_idx], list(G.nodes())[dest_idx])
        try:
            routes = list(ox.k_shortest_paths(G, start, dest, k))
        except:
            routes = None
            raise Exception("No valid routes between two nodes")
        if routes and len(routes) > 1:
            return routes
        elif routes and len(routes) == 1:
            return routes[0]

    def generate_k_routes_from_origin_to_dest(self, k, origin, dest):
        G = self._map
        try:
            routes = list(ox.k_shortest_paths(G, origin, dest, k))
        except:
            routes = None
            raise Exception("No valid routes between two nodes")
        if routes and len(routes) > 1:
            return routes
        elif routes and len(routes) == 1:
            return routes[0]
    
    def generate_route_from_orig_to_dest_include_nodes(self, origin, dest, nodes):
        G = self._map
        if len(nodes) == 0:
            route = Ox_Route(self.generate_k_routes_from_origin_to_dest(k=1, origin=origin, dest=dest))
        else:
            paths = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)] + [(nodes[len(nodes)-1], dest)]
            route = Ox_Route(self.generate_k_routes_from_origin_to_dest(k=1, origin=origin, dest=nodes[0]))
            for path in paths:
                route = route + Ox_Route(self.generate_k_routes_from_origin_to_dest(k=1, origin=path[0], dest=path[1]))
        return route

    def add_car(self, n):
        new_variables = ["Car_{}_Route_{}".format(u,v) for u,v in product(range(self._no_cars, self._no_cars+n), range(self._no_routes))]
        self._variables += new_variables
        self._no_cars += n

    def obtain_map(self, plot=False, **args):
        if plot:
            import matplotlib.pyplot as plt
            ox.plot_graph(self._map, bgcolor='white', edge_color='black', edge_linewidth=1.0)
            plt.show()
        return self._map

    def routing_dict(self):
        return self._routing_dict

    def no_cars(self):
        return self._no_cars

    def no_routes(self):
        return self._no_routes
    
    def coupling_map(self):
        return self._coupling_map

    def qubit_mapping(self):
        return self._qubit_mapping
    
    def qubit_neighbours(self):
        return self._qubit_neighbours

    def assign_route(self, var, route):
        self._routing_dict[var] = route
        try:
            self.check_valid_coupling()
        except Exception:
            self.remove_route(var)
            raise Exception("Not a valid coupling so the assigned route has been removed.")

    def remove_route(self, var):
        self._routing_dict.pop(var, None)

    def get_qubit_from_var(self, var):
        if self._qubit_mapping:
            return self._qubit_mapping[var]
        else:
            raise Exception("There are no qubit assignments provided.")
    
    def get_var_from_qubit(self, var):
        if self._qubit_mapping:
            for qubit, temp_var in self._qubit_mapping.items():
                if temp_var == var:
                    return qubit
            return None
        else:
            raise Exception("There are no qubit assignments provided.")
    
    def get_route_from_var(self, var):
        return self._routing_dict[var]

    def check_valid_coupling(self):
        qubit_neighbours = self._qubit_neighbours
        list_routing = list(self._routing_dict.items())
        for k in range(len(self._routing_dict)):
            var, route = list_routing[k]
            if route:
                for k_2 in range(k+1, len(self._routing_dict)):
                    var_2, route_2 = list_routing[k_2]
                    if route_2:
                        overlapping_edges = route.get_common_edges(route_2)
                        qubit = self.get_qubit_from_var(var)
                        qubit_2 = self.get_qubit_from_var(var_2)
                        if overlapping_edges:
                            weight = 0
                            print("{} and {} have overlapping routes, now checking if their corresponding qubits can interact (up to 2nd neighbours)...".format(var, var_2))
                            for edge in overlapping_edges:
                                edge_data = self._map.get_edge_data(*edge, default=[None])[0]
                                if edge_data:
                                    weight += edge_data["length"]
                            print("Length of overlapping segments(m): {}".format(weight))
                            if qubit_2 not in qubit_neighbours[qubit] and weight > 50:
                                raise Exception("Not a valid coupling and overlapping length is too significant.")
                            else:
                                print("...Ok")
        print("Yay, valid assignment of routes!")
        return True
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return str(dict( [(str(a), str(b)) for a,b in self._routing_dict.items()]))

def sublist(list_a, list_b):
    if 0 == len(list_a):
        return 0

    if len(list_b) < len(list_a):
        return -1

    idx = -1
    while list_a[0] in list_b[idx+1:]:
        idx = list_b.index(list_a[0], idx + 1)
        if list_a == list_b[idx:idx+len(list_a)]:
            return idx

    return -1


def add_qubo_constraints(qubo, no_cars, no_routes):
    for i in range(no_cars):
        qubo.linear_constraint(linear = {"X_{}_{}".format(str(i).zfill(2), j):1 for j in range(no_routes)}, sense= "==", rhs = 1, name = "Car_{}".format(i) )
    return qubo

def main():
    #Must load after importing/defining Ox_Route and QAOA_Traffic classes
    with open("traffic.pkl", 'rb') as f:
        traffic = pkl.load(f)


    routing_dict =  {var: traffic.routing_dict()[var].routing() for var in traffic.routing_dict()} 
    edges_dict = get_edges_dict(routing_dict)
    new_dict = deepcopy(edges_dict)
    #Remove edges with <50m overlaps.
    for edge, value in edges_dict.items():
        weight = 0
        edge_data = traffic.obtain_map().get_edge_data(*edge, default=[None])[0]
        if edge_data:
            weight += edge_data["length"]
        if weight < 50:
            # print(edge, value, weight)
            new_dict.pop(edge, None)
    variable_renaming = {"Car_{}_Route_{}".format(str(i).zfill(2), j): "X_{}_{}".format(str(i).zfill(2), j) for i, j in product(range(21), range(3))}
    new_edges_dict = {}
    for edge, variables in new_dict.items():
        new_edges_dict[edge] = [variable_renaming[var] for var in variables]
    del edges_dict
    del new_dict
    global qubo
    unconstrained_qubo, linear, quadratic = build_qubo_unconstrained_from_edges_dict(traffic.obtain_map(), new_edges_dict, variables=variable_renaming.values())
    qubo = add_qubo_constraints(unconstrained_qubo, no_cars=21, no_routes=3)
    qubo, max_coeff = convert_qubo(qubo, linear, quadratic, penalty_multiplier=1.5)
    min_results = {}
    min_total_obj = 0
    max_results = {}
    max_total_obj = 0
    for car in range(21):
        zeros = [0] * 3
        assignments = [ [0]*i + [1] + [0]*(2-i) for i in range(3) ]
        assignments_obj = {}
        for assignment in assignments:
            x = zeros*(car) + assignment + zeros*(20-car)
            var_dict = {"X_{}_{}".format(str(int((k-k%3)/3)).zfill(2), k%3): x[k] for k in range(len(x)) }
            obj = unconstrained_qubo.objective.evaluate(var_dict)
            assignments_obj[tuple(assignment)] = obj
        min_assignment = min(assignments_obj.items(), key = lambda x: x[1])
        max_assignment = max(assignments_obj.items(), key = lambda x: x[1])
        min_total_obj += min_assignment[1]
        max_total_obj += max_assignment[1]
        min_car_dict_assignment = {"X_{}_{}".format(str(car).zfill(2), k): min_assignment[0][k] for k in range(3) }
        max_car_dict_assignment = {"X_{}_{}".format(str(car).zfill(2), k): max_assignment[0][k] for k in range(3) }
        for var, value in min_car_dict_assignment.items():
            min_results[var] = value
        for var, value in max_car_dict_assignment.items():
            max_results[var] = value
    greedy = qubo.objective.evaluate(min_results)
    max_greedy = qubo.objective.evaluate(max_results)
    print("Zeros:", qubo.objective.evaluate([0 for _ in range(63)]))
    print("MIN GREEDY:", greedy)
    print("Total distance traversed(min):", min_total_obj)
    print(min_results)
    print("MAX GREEDY:", max_greedy)
    print("Total distance traversed(max):", max_total_obj)
    print(max_results)
    optimizer = NELDER_MEAD()
    optimizer.set_options(maxiter=100)
    backend = Aer.get_backend("aer_simulator_matrix_product_state")
    quantum_instance = QuantumInstance(backend, shots=1024)
    op, offset = qubo.to_ising()
    start = time()
    result, qc = QiskitQAOA(operator=op, quantum_instance=quantum_instance, optimizer=optimizer, reps=1, construct_circ=False)
    end = time()
    print(result.eigenstate)
    print(result.eigenvalue+offset)
    # print(qc.draw(fold=200))
    print("Time taken: {}s".format(end-start))
    print(result.cost_function_evals)

if __name__ == "__main__":
    main()
