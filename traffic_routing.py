import pickle as pkl
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations

filepath='unimelb_2.pkl'
with open(filepath, 'rb') as f:
    G = pkl.load(f)
class Ox_Route:
    def __init__(self, routing: list):
        self._routing = routing
    
    def compose(self, ox_route_2: list):
        """Compose self with another route(list) provided self has terminal node is same with the 2nd routing's initial node. 

        Args:
            ox_route_2 ([list]): 2nd route to compose self with.
        """
        if self._routing[::-1][0] == ox_route_2[0]:
            self._routing += ox_route_2
        else:
            raise Exception('The 2nd route does not start at the same node at which the 1st terminates')
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
    
    def check_common_edges(self, other):
        """Checks whether self routing has common edges with other routing.

        Args:
            other ([Ox_Route]): other routing to compare edges with

        Returns:
            [bool]: Whether or not self has common edges with other
        """

        for edge in self.obtain_list_of_edges():
            if edge in other.obtain_list_of_edges():
                return True

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
    
    def compute_qubit_neigbours(self, max_no_):
        self._qubit_neibours = {}
        qubits = list(range(np.amax(self._coupling_map)))

    def generate_k_routes_from_origin_to_dest(self, k):
        G = self._map
        fig, ax = ox.plot_graph(G, figsize = (10,4), bgcolor='white', edge_color='black', edge_linewidth=0.5, show=False)
        self._temp_nodes = []
        def append_two_nodes_onclick(event):
            nodes = self._temp_nodes
            x, y = (event.ydata, event.xdata)
            node = ox.nearest_nodes(G, y, x)
            nodes.append(node)
            if len(nodes) == 2:
                plt.close()
        cid = fig.canvas.mpl_connect('button_press_event', append_two_nodes_onclick)
        plt.show()
        try:
            routes = list(ox.k_shortest_paths(G, *self._temp_nodes, k))
        except:
            routes = None
            raise Exception("No valid routes between two nodes")
        if routes:
            return routes

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
        if routes:
            return routes
        
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

    def assign_route(self, var, route):
        if var not in self._routing_dict:
            self._routing_dict[var] = route
        else:
            raise Exception("This variable already has an existing Route")

    def remove_route(self, var):
        self._routing_dict.pop(var, None)

    def get_qubit_from_var(self, var):
        if self._qubit_mapping:
            return self._qubit_mapping[var]
        else:
            raise Exception("There are no qubit assignments provided.")

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

with open("coupling_map.pkl", "rb") as f:
    coupling_map = pkl.load(f)
import json
with open("qubit_mapping.json", "r") as f:
    qubit_mapping = json.load(f)
traffic = QAOA_Traffic(21,3,G, coupling_map, qubit_mapping)
