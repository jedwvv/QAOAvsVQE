import osmnx as ox
import pickle as pkl

def main(filepath, radius):
    melbourne = (-37.798346, 144.961559) #coordinates
    G = ox.graph_from_point(melbourne, dist=radius, network_type="drive") #Generate graph
    fig, ax = ox.plot_graph(G, node_size=1) #Visualize
    
    #Save graph file
    with open(filepath, 'wb') as f:
        pkl.dump(G, f)
    return fig, ax
    
if __name__ == "__main__":
    main() 