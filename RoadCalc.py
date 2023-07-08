import networkx as nx
from matplotlib import pyplot as plt


if __name__ == '__main__':
    nodes = [0, 1, 2, 3, 4, 5]
    edges = [(0, 1, 8), (2, 3, 5), (4, 1, 3), (3, 1, 5)]
    city_graph = nx.DiGraph()
    city_graph.add_nodes_from(nodes)
    city_graph.add_weighted_edges_from(edges, weight='weight')

    print(city_graph.edges.data())

    circ_layout = nx.circular_layout(city_graph)
    nx.draw(city_graph, circ_layout, with_labels=True, connectionstyle='arc3, rad = 0.25')
    labels = nx.get_edge_attributes(city_graph, 'weight')
    nx.draw_networkx_edge_labels(city_graph, circ_layout, edge_labels=labels)

    plt.show()
    plt.savefig("graph.png")
