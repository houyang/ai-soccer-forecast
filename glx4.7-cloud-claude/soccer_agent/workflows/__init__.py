# soccer_agent/workflows/__init__.py
from soccer_agent.workflows.state import PredictionState, EvaluationState
from soccer_agent.workflows.prediction import build_prediction_graph
from soccer_agent.workflows.evaluation import build_evaluation_graph

__all__ = [
    "PredictionState", "EvaluationState",
    "build_prediction_graph", "build_evaluation_graph"
]