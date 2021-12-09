import osmnx as ox

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

from itertools import product, combinations

class QAOA_Traffic:
    def __init__(self, no_cars, no_routes, G_map = None, coupling_map = None):
        self._map = G_map
        self._no_cars = no_cars
        self._no_routes = no_routes
        self._coupling_map = coupling_map
        self.set_variables()

    def no_cars(self):
        return self._no_cars

    def no_routes(self):
        return self._no_routes
    
    def coupling_map(self):
        return self._coupling_map

    def set_variables(self):
        variables = ["Car_{}_Route_{}".format(u,v) for u,v in product(range(self._no_cars), range(self._no_routes))]
        self._variables = variables
        return variables
    
    def set_map(self, G_map):
        self._map = G_map

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

    def set_variables_routing_dict(self):
        if not hasattr(self, '_variables'):
            self.set_variables()
        routing_dict = {var:None for var in self._variables}
        return routing_dict
import pickle as pkl
# import matplotlib.pyplot as plt
import numpy as np

filepath='unimelb_2.pkl'
with open(filepath, 'rb') as f:
    G = pkl.load(f)
# fig = ox.plot_graph(G, figsize = (10,4), bgcolor='white', edge_color='black', edge_linewidth=1.0)


traffic = QAOA_Traffic(3, 2)
# traffic.set_map(G)
# traffic.obtain_map(plot=True)
print(traffic.set_variables_routing_dict())
# rng = np.random.default_rng(123)
# start, dest = rng.integers(low=0, high=len(G.nodes()), size=2)

# routing = ox.shortest_path(G, list(G.nodes())[start], list(G.nodes())[dest])
# route = Ox_Route(routing)
# print("ROUTE:")
# print(route)

# k=5
# removed_routing = route.obtain_last_k_nodes(k)
# print("Last {} nodes".format(k))
# print(removed_routing)
# print("")
# print("TEST before removing")
# print(route.check_common_edges(removed_routing))
# route.remove(removed_routing.routing())
# print("Removed last {} nodes".format(k))
# print(route)
# print("TEST AFTER REMOVING")
# print(route.check_common_edges(removed_routing))
# print( removed_route )
# print("\nRecombined:")
# print( route+removed_route )
# print(route.routing()[len(route.routing())-5:len(route.routing())])