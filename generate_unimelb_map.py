import osmnx as ox
import pickle as pkl

def main():
    unimelb_physics = (-37.797112, 144.963746) #coordinates
    distance = 1000 #meters
    G = ox.graph_from_point(unimelb_physics, dist=distance, network_type="drive") #Generate graph
    fig, ax = ox.plot_graph(G, node_size=1) #Visualize
    
    #Save graph file
    with open('unimelb_graph.pkl', 'wb') as f:
        pkl.dump(G, f)
    return fig, ax
    
if __name__ == "__main__":
    main() 