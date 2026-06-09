from soccer_agent.workflows.evaluation import build_evaluation_graph


def test_evaluation_graph_build():
    graph = build_evaluation_graph()

    assert graph is not None
    assert len(graph.nodes) == 5


def test_evaluation_graph_type():
    graph = build_evaluation_graph()

    assert isinstance(graph, type(graph))


def test_evaluation_graph_edges():
    graph = build_evaluation_graph()

    edges = graph.edges
    assert isinstance(edges, dict)
    assert len(edges) == 5

    assert ("find_pending", "fetch_results") in edges
    assert ("fetch_results", "compare") in edges
    assert ("update_db", "__end__") in edges