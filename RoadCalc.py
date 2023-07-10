import math
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from scipy import linalg as LA


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
    """
    Dijkstra Alg. is used, to find the SP from src to dst.
    """

    return nx.dijkstra_path(graph, source=src, target=dst, weight='weight')


def make_normalized_tpm_from_flow_sps_on_intersections(graph, sp_list):
    """
    Using the flow count on each path, we calculate the transition possibility from a node to other.
    """

    # To make the tpm, we will check each sp to count the number of jumps from each node to other
    size = len(graph.nodes)
    probs = [[0 for x in range(size)] for y in range(size)]
    # print(probs)
    for sp in sp_list:
        for k in range(0, (len(sp) - 1)):
            probs[sp[k]][sp[k + 1]] += 1
    # print(probs)

    # now sum of each row must be 1. so for non-zeroes, we normalize and if not, the probs[i][i] = 1
    for i in range(size):
        s = sum(probs[i])
        if s != 0:
            for j in range(size):
                probs[i][j] /= s
        else:
            # we have a 0 row
            probs[i][i] = 1.0
    # print(probs)
    return probs


def make_dual_graph_and_tpm(graph, tpm):
    """
    A graph and it's tpm are given. the function return the dual grapg and it's tpm
    """

    edges = list(graph.edges)
    # print(edges)
    dual_edge_list = []
    new_probs = [[0 for x in range(len(edges))] for y in range(len(edges))]

    for i in range(len(edges)):
        s1, d1 = edges[i]
        for j in range(len(edges)):
            s2, d2 = edges[j]
            if d1 == s2:
                ne = (edges[i], edges[j], tpm[s1][d1] * tpm[s2][d2])
                dual_edge_list.append(ne)
                new_probs[i][j] = tpm[s1][d1] * tpm[s2][d2]

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


def find_the_index_of_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_steady_state(tpm):
    """
    he steady state for the given TPM is found.
    """
    # one of the methods is to find eigen values and vectors
    # (A - I)v = 0
    eig_val, eig_vec = LA.eig(tpm, left=True, right=False)
    eig_vec = eig_vec.T
    idx = find_the_index_of_nearest(eig_val, 1)
    # print(idx)
    # print("-----------------------")
    # print(eig_val)
    # print("-----------------------")
    # print(eig_vec)

    if math.isclose(eig_val[idx], 1, abs_tol=0.001, rel_tol=0.001):
        vec = eig_vec[idx]
        # print(eig_val[idx])
        # print(eig_vec[idx])
    else:
        print("No steady state! No eigen value is 1")
        return None

    steady_state = []
    if len(vec) == 0:
        print("No steady state!")
        return None
    else:
        state = (vec / np.sum(vec))
        for i in range(len(state)):
            steady_state.append(np.round(state[i].real, 6))

    return steady_state


def calc_road_density(steady_state, num_cars, num_lanes):
    """
    First for each road we calculate that how many cars and for how long are in a road.
    Then, we calculate road capacity by it's length and how many lanes it has.
    division of this two results in road density over time.
    """

    global road_lens

    den_list = []
    for i in range(0, len(steady_state)):
        mass = num_cars * steady_state[i]
        volume = num_lanes * road_lens[i]
        den_list.append(mass / volume)

    return den_list


def normalize_arr(my_list):
    if my_list is None or len(my_list) == 0:
        return my_list

    for i in range(len(my_list)):
        if max(my_list) != 0:
            my_list[i] /= max(my_list)
        else:
            print("Arr max is 0")
            return my_list

    return my_list


def utilize(graph, den_list, power):
    """
    utilization function which multiplies current cost by the mentioned power of corresponding density.
    That means cost[i] = cost[i] * (d[i]^p)
    """
    graph_data = list(graph.edges.data())
    weights = []
    for i in range(len(graph_data)):
        s, d, w = graph_data[i]
        weights.append(w['weight'])
    return [weights[i] * (math.pow(den_list[i], power)) for i in range(len(den_list))]


def calc_avg_path_cost(w, sp_list):
    s = 0
    for sp in sp_list:
        for k in range(0, (len(sp) - 1)):
            s += w[(sp[k], sp[k + 1])]
    return s / len(sp_list)


def cost_calc(graph, flow_list=None, avg_speed=1, num_lanes=1, num_cars=0):
    """
    main run function for iterations. returns the calculated costs on each iter using the CTMC it creates.
    each time the new costs are returned along with the new graph that was made with weights.
    """

    global road_lens
    global avg_path_cost_list

    # Validate input lists
    if flow_list:
        num_cars = len(flow_list)
    else:
        return None

    if graph is None:
        return None

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
    # print(new_tpm_with_tt)

    # Find steady states (there might not be any, so maybe(?) approx it).
    steady_state = find_steady_state(new_tpm_with_tt)
    if steady_state is None:
        return None
    # print(steady_state)

    # Calc road density
    den_list = calc_road_density(steady_state, num_cars=num_cars, num_lanes=num_lanes)
    # print(den_list)
    # colors = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # plt.scatter(range(len(den_list)), den_list, c=colors, cmap='viridis')
    # plt.show()
    den_list = normalize_arr(den_list)
    # print(den_list)

    # Define a utility function for the state.
    # the denser a road is, the higher it's cost should be so it's preferably avoided
    new_edge_costs = utilize(graph, den_list, 1)
    # print(new_edge_costs)

    new_edges = []
    weight_map = {}
    for i in range(len(graph.edges)):
        s, d = list(graph.edges)[i]
        weight_map[(s, d)] = new_edge_costs[i]
        new_edges.append((s, d, new_edge_costs[i]))
    new_graph = setup_graph(graph.nodes, new_edges)

    avg_path_cost = calc_avg_path_cost(weight_map, sp_list)
    print(avg_path_cost)
    avg_path_cost_list.append(avg_path_cost)
    return new_edge_costs, new_graph


if __name__ == '__main__':
    # For ease of use, nodes and edges are hard coded.
    # Nodes of the city graph, the intersections between the roads.
    n = [0, 1, 2, 3, 4, 5]

    # Edges of the city graph, the roads.
    # e = [(0, 1, 8), (2, 3, 5), (4, 1, 3), (3, 1, 5), (4, 5, 1), (5, 1, 1), (0, 5, 4), (5, 2, 3)]
    e = [(0, 1, 8), (0, 3, 4),
         (1, 2, 10), (1, 4, 2),
         (2, 0, 1), (2, 3, 3),
         (3, 5, 5),
         (4, 1, 2), (4, 5, 8),
         (5, 0, 4), (5, 2, 4)
         ]

    # create city graph
    city_graph = setup_graph(n, e)
    print("Nodes: ", city_graph.nodes.data())
    print("Edges: ", city_graph.edges.data())

    # show the city graph
    draw_graph(city_graph)

    # List of flows in the city, defined by a pair of nodes, indicating src & dst.
    # This list is also hard-coded, could have been taken as input.
    # city_flow_list = [(0, 1), (2, 1), (4, 1), (5, 2)]

    src_dst = [{'src': 0, 'dst': 1},
               {'src': 0, 'dst': 2},
               {'src': 0, 'dst': 2},
               {'src': 0, 'dst': 3},
               {'src': 0, 'dst': 3},
               {'src': 0, 'dst': 3},
               {'src': 0, 'dst': 3},
               {'src': 0, 'dst': 4},
               {'src': 0, 'dst': 4},
               {'src': 0, 'dst': 5},
               {'src': 0, 'dst': 5},
               {'src': 1, 'dst': 0},
               {'src': 1, 'dst': 0},
               {'src': 1, 'dst': 2},
               {'src': 1, 'dst': 3},
               {'src': 1, 'dst': 4},
               {'src': 1, 'dst': 5},
               {'src': 1, 'dst': 5},
               {'src': 2, 'dst': 0},
               {'src': 2, 'dst': 0},
               {'src': 2, 'dst': 1},
               {'src': 2, 'dst': 3},
               {'src': 2, 'dst': 4},
               {'src': 2, 'dst': 5},
               {'src': 3, 'dst': 0},
               {'src': 3, 'dst': 1},
               {'src': 3, 'dst': 4},
               {'src': 3, 'dst': 5},
               {'src': 3, 'dst': 5},
               {'src': 4, 'dst': 0},
               {'src': 4, 'dst': 0},
               {'src': 4, 'dst': 0},
               {'src': 4, 'dst': 0},
               {'src': 4, 'dst': 1},
               {'src': 4, 'dst': 2},
               {'src': 4, 'dst': 3},
               {'src': 4, 'dst': 5},
               {'src': 5, 'dst': 0},
               {'src': 5, 'dst': 1},
               {'src': 5, 'dst': 1},
               {'src': 5, 'dst': 2},
               {'src': 5, 'dst': 2},
               {'src': 5, 'dst': 3},
               {'src': 5, 'dst': 3},
               {'src': 5, 'dst': 3},
               {'src': 5, 'dst': 4}
               ]
    city_flow_list = []
    for a in src_dst:
        city_flow_list.append((a['src'], a['dst']))

    # Some values are set to be constant, like the average car speed or the number of lanes in roads.
    # These values are hard-coded for the sake of simplicity and again could be input from the prompt if desired.
    # avg_car_speed = 50
    avg_car_speed = 60
    num_of_road_lanes = 1

    # Road lengths are to be copied. They are the initial values to road weights.
    # (weights change later, thus the need of initial copy)
    road_lens = [i[2] for i in e]

    # The road lengths are used, to create a relatively normalized list of travel times .
    # (i.e. the smallest is set to 1 and all are scaled relative to that)
    tt = [x / avg_car_speed for x in road_lens]
    tt = [x / min(tt) for x in tt]

    prev_avg_cost = None
    avg_path_cost_list = []
    while True:
        new_costs, city_graph = cost_calc(graph=city_graph, flow_list=city_flow_list, avg_speed=avg_car_speed,
                                          num_lanes=num_of_road_lanes)
        if new_costs is None or len(new_costs) == 0:
            print("Something went WRONG")
            break

        # The average cost used for checking convergence
        new_avg_cost = sum(new_costs) / len(new_costs)
        # print(new_avg_cost)
        if prev_avg_cost:
            # The np.isclose tolerance could be set o.w. if desired by setting the input parameters.
            if np.isclose(prev_avg_cost, new_avg_cost, atol=0.001, rtol=0.001):
                break
        prev_avg_cost = new_avg_cost

    print(new_costs)
    plt.plot(range(len(avg_path_cost_list)), avg_path_cost_list)
    plt.xlabel('number of iteration')
    plt.ylabel('avg path cost')
    plt.show()
