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


def find_shortest_path(graph, src, dst):
    """Dijkstra Alg. is used, to find the SP from src to dst."""

    return nx.dijkstra_path(graph, source=src, target=dst, weight='weight')


def make_normalized_tpm_from_flow_sps_on_intersections(graph, sp_list):
    """Using the flow count on each path, we calculate the transition possibility from a node to other."""

    # To make the tpm, we will check each sp to count the number of jumps from each node to other
    size = len(graph.nodes)
    probs = [[0 for x in range(size)] for y in range(size)]
    print(probs)
    for sp in sp_list:
        for k in range(0, (len(sp) - 1)):
            probs[sp[k]][sp[k + 1]] += 1
    print(probs)

    # now sum of each row must be 1. so for non-zeroes, we normalize and if not, the probs[i][i] = 1
    for i in range(size):
        s = sum(probs[i])
        if s != 0:
            for j in range(size):
                probs[i][j] /= s
        else:
            # we have a 0 row
            probs[i][i] = 1.0
    print(probs)
    return probs


def make_dual_graph_and_tpm(graph, tpm):
    """A graph and it's tpm are given. the function return the dual grapg and it's tpm"""

    edges = list(graph.edges)
    print(edges)
    dual_edge_list = []
    new_probs = [[0 for x in range(len(edges))] for y in range(len(edges))]

    for i in range(len(edges)):
        s1, d1 = edges[i]
        for j in range(len(edges)):
            s2, d2 = edges[j]
            if d1 == s2:
                ne = (edges[i], edges[j], tpm[s1][d1] + tpm[s2][d2])
                dual_edge_list.append(ne)
                new_probs[i][j] = tpm[s1][d1] + tpm[s2][d2]

    dual_graph = setup_graph(edges, dual_edge_list)
    # draw_graph(dual_graph)

    for i in range(len(edges)):
        s = sum(new_probs[i])
        if s != 0:
            for j in range(len(edges)):
                new_probs[i][j] /= s
        # here we didn't touch the 0 rows since they'd be filled up later
    # print(new_probs)
    return dual_graph, new_probs


def make_new_tpm_with_tt_values(tpm):
    global tt

    if len(tpm) == 0:
        print("TPM thrown with len 0")
        return tpm
    if len(tpm) != len(tpm[0]):
        print("TPM not symmetric.")
        return tpm

    new_probs = [[0 for x in range(len(tpm))] for y in range(len(tpm))]
    for i in range(len(tpm)):
        new_probs[i][i] = (tt[i] - 1) / (tt[i])

    for i in range(len(tpm)):
        for j in range(len(tpm)):
            if i != j:
                new_probs[i][j] = (1 - new_probs[i][i]) * tpm[i][j]

    return new_probs


def cost_calc(graph, flow_list=None, avg_speed=1, num_lanes=1, num_cars=0):
    """main run function for iterations. returns the calculated costs on each iter using the CTMC it creates."""

    global road_lens

    # Validate input lists
    if flow_list:
        num_cars = len(flow_list)
    else:
        return None

    if graph is None:
        return None

    new_edge_costs = []
    sp_list = []

    # First, the shortest path for each flow is found
    for flow in flow_list:
        src, dst = flow
        sp_list.append(find_shortest_path(graph, src, dst))
    # print("The sp for each flow:", sp_list)

    # Then, assuming all flows going on SPs, the first transition probability matrix is made.
    tpm = make_normalized_tpm_from_flow_sps_on_intersections(graph, sp_list)

    # Made the graph dual, to see from which road we go to which one (traffic flows), like paper 1 proposed.
    dual_graph, new_tpm = make_dual_graph_and_tpm(graph, tpm)
    # draw_graph(dual_graph)
    # print(new_tpm)

    # normalized travel times once in the main function,
    # Now we use it to fill out the new tpm like paper 1 and 2 proposed.
    new_tpm_with_tt = make_new_tpm_with_tt_values(new_tpm)
    print(new_tpm_with_tt)
    # todo: check the previously empty rows later

    # TODO find steady states (there might not be any, so approx it).

    # TODO calculate utility functions for the state.

    # TODO return new costs for the next iteration.

    return new_edge_costs


if __name__ == '__main__':
    # For ease of use, nodes and edges are hard coded.
    # Nodes of the city graph, the intersections between the roads.
    n = [0, 1, 2, 3, 4, 5]

    # Edges of the city graph, the roads.
    e = [(0, 1, 8), (2, 3, 5), (4, 1, 3), (3, 1, 5), (4, 5, 1), (5, 1, 1), (0, 5, 4), (5, 2, 3)]

    # create city graph
    city_graph = setup_graph(n, e)
    print("Nodes: ", city_graph.nodes.data())
    print("Edges: ", city_graph.edges.data())

    # show the city graph
    draw_graph(city_graph)

    # List of flows in the city, defined by a pair of nodes, indicating src & dst.
    # This list is also hard-coded, could have been taken as input.
    city_flow_list = [(0, 1), (2, 1), (4, 1), (5, 2)]

    # Some values are set to be constant, like the average car speed or the number of lanes in roads.
    # These values are hard-coded for the sake of simplicity and again could be input from the prompt if desired.
    avg_car_speed = 50
    num_of_road_lanes = 1

    # Road lengths are to be copied. They are the initial values to road weights.
    # (weights change later, thus the need of initial copy)
    road_lens = [i[2] for i in e]

    # The road lengths are used, to create a relatively normalized list of travel times .
    # (i.e. the smallest is set to 1 and all are scaled relative to that)
    tt = [x / avg_car_speed for x in road_lens]
    tt = [x / min(tt) for x in tt]

    prev_avg_cost = None
    while True:
        new_costs = cost_calc(graph=city_graph, flow_list=city_flow_list, avg_speed=avg_car_speed,
                              num_lanes=num_of_road_lanes)
        if new_costs is None or len(new_costs) == 0:
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
