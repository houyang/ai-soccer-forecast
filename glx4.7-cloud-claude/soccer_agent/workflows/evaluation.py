from langgraph.graph import StateGraph, END
from soccer_agent.workflows.state import EvaluationState
from soccer_agent.db.models import Prediction, Match, Evaluation


def test_evaluation_graph_build():
    graph = build_evaluation_graph()

    assert graph is not None
    assert len(graph.nodes) == 5  # find_pending, fetch_results, compare, reflect, update_db, END


def test_evaluation_graph_type():
    graph = build_evaluation_graph()

    assert isinstance(graph, StateGraph)


def test_evaluation_graph_edges():
    graph = build_evaluation_graph()

    edges = graph.edges
    assert isinstance(edges, dict)
    assert len(edges) == 5

    assert ("find_pending", "fetch_results") in edges
    assert ("fetch_results", "compare") in edges
    assert ("update_db", "__end__") in edges