import osmnx as ox
import pickle as pkl

def main():
    melbourne = (-37.798346, 144.961559) #coordinates
    distance = 1000 #meters
    G = ox.graph_from_point(melbourne, dist=distance, network_type="drive") #Generate graph
    fig, ax = ox.plot_graph(G, node_size=1) #Visualize
    
    #Save graph file
    with open('uni_melbourne.pkl', 'wb') as f:
        pkl.dump(G, f)
    return fig, ax
    
if __name__ == "__main__":
    main() 