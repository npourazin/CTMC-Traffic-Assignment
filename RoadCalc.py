import networkx as nx
from matplotlib import pyplot as plt


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
