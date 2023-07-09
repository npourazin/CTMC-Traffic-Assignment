import networkx as nx
from matplotlib import pyplot as plt
import numpy as np


def setup_graph(nodes, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges, weight='weight')
    return graph


def draw_graph(graph):
    circ_layout = nx.circular_layout(graph)
    nx.draw(graph, circ_layout, with_labels=True, connectionstyle='arc3, rad = 0.25')
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, circ_layout, edge_labels=labels)

    plt.show()
    plt.savefig("graph.png")


def cost_calc(graph, flow_list=None, avg_speed=1, num_lines=1, num_cars=0):
    """main run function. returns the calculated costs after creating the TPM"""

    global road_lens

    # Validate input lists
    if flow_list:
        num_cars = len(flow_list)
    else:
        return None

    if graph is None:
        return None

    new_edge_costs = []

    # TODO first, the shortest path for each flow is found

    # TODO then, assuming all flows going on SPs, we can make the transition probability matrix.

    # TODO make TPM dual, to see from which road we go to which one (traffic flows), like paper 1 proposed.

    # TODO normalize travel times to fill out the new tpm like paper 2 proposed.

    # TODO find steady states (there might not be any, so approx it).

    # TODO calculate utility functions for the state.

    # TODO return new costs for the next iteration.

    return new_edge_costs


if __name__ == '__main__':
    # For ease of use, nodes and edges are hard coded.
    # Nodes of the city graph, the intersections between the roads.
    n = [0, 1, 2, 3, 4, 5]

    # Edges of the city graph, the roads.
    e = [(0, 1, 8), (2, 3, 5), (4, 1, 3), (3, 1, 5), (4, 5, 1)]

    # create city graph
    city_graph = setup_graph(n, e)
    print("Nodes: ", city_graph.nodes.data())
    print("Edges: ", city_graph.edges.data())

    # show the city graph
    draw_graph(city_graph)

    # List of flows in the city, defined by a pair of nodes, indicating src & dst.
    # This list is also hard-coded, could have been taken as input.
    city_flow_list = [(0, 1), (2, 1), (4, 5)]

    # Road lengths are to be copied. they are the initial values to road weights
    # (weights change later, thus the need of initial copy)
    road_lens = [i[2] for i in e]

    prev_avg_cost = None
    while True:
        new_costs = cost_calc(graph=city_graph, flow_list=city_flow_list, avg_speed=50, num_lines=1)
        if new_costs is None:
            print("Something went WRONG")
            break

        # The average cost used for checking convergence
        new_avg_cost = sum(new_costs) / len(new_costs)
        if prev_avg_cost:
            # The default no.isclose tolerance is used, could be set o.w. if desired by setting the input parameters.
            np.isclose(prev_avg_cost, new_avg_cost)
            break
        prev_avg_cost = new_avg_cost

    print(new_costs)
