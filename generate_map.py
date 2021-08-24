import osmnx as ox
import pickle as pkl

def main():
    melbourne = (-37.813318, 144.965546) #coordinates
    distance = 800 #meters
    G = ox.graph_from_point(melbourne, dist=distance, network_type="drive") #Generate graph
    fig, ax = ox.plot_graph(G, node_size=1) #Visualize
    
    #Save graph file
    with open('melbourne_2.pkl', 'wb') as f:
        pkl.dump(G, f)
    return fig, ax
    
if __name__ == "__main__":
    main() 