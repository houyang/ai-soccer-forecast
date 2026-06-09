import pytest

from soccer.agent import PredictionAgent
from soccer.harness import run_scenario
from soccer.reasoning.fake import DeterministicReasoner
from soccer.scenarios import SCENARIO_NAMES, load_scenario


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_each_scenario_runs_end_to_end(name: str) -> None:
    scenario = load_scenario(name)
    agent = PredictionAgent(registry=scenario.registry, reasoner=DeterministicReasoner())
    report = run_scenario(scenario, agent)
    assert report.n == len(scenario.matches) >= 1
    assert 0.0 <= report.accuracy <= 1.0


def test_unknown_scenario_raises() -> None:
    with pytest.raises(KeyError):
        load_scenario("does-not-exist")
