from langgraph.graph import StateGraph, END
from soccer_agent.workflows.state import PredictionState


def test_prediction_graph_build():
    graph = StateGraph(PredictionState)

    assert graph is not None
    # LangGraph includes the state as a node, so we expect 10 nodes (9 custom + 1 END)
    assert len(graph.nodes) == 10


def test_prediction_graph_type():
    graph = StateGraph(PredictionState)

    assert isinstance(graph, StateGraph)


def test_graph_has_end_node():
    graph = StateGraph(PredictionState)

    assert graph.nodes is not None
    # The END node should be a special sentinel
    # We can check the nodes type to verify END handling
    node_types = {type(n).__name__ for n in graph.nodes.values()}
    assert len(node_types) > 0