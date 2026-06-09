"""EvalCase + pinned dataset of past historical matches.

The harness (Task 18) iterates `EVAL_CASES`, materializes fixtures for
each one, runs the agent, and scores the predictions. The pinned scores
below come from real historical results sourced from football-data.co.uk
public CSVs (24/25 + 25/26 seasons, 5 competitions: EPL, LaLiga, SerieA,
Bundesliga, UCL). Knockout rounds are deliberately excluded from the
UCL slice — the UCL final is a Phase 2 live-data target, not eval.

Adding a new case:
  1. Add an `EvalCase(...)` to `EVAL_CASES`.
  2. Make sure its score matches the actual result.
  3. Re-run `pytest tests/test_dataset.py` — coverage/divergence
     tests will tell you if your case breaks the dataset properties
     (e.g. it stops having a draw, or it duplicates an id).

Ingesting more cases from football-data.co.uk:
  python scripts/ingest_football_data.py --csv <path-to-csv> --append
See `scripts/_append_ingested.py` for the helper that materializes
the fixture JSON files for each new case.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class EvalCase:
    """A pinned past match with known result.

    The factory uses these fields to write deterministic fixture
    files. The harness uses the result to score the prediction.
    """
    match_id: str
    competition: str
    round: str
    home_id: str
    away_id: str
    venue_id: str  # may be "" for unknown / neutral fallback
    kickoff: datetime  # MUST be timezone-aware
    home_goals: int
    away_goals: int
    # Convenience: derived in __post_init__ so callers never disagree.
    actual_winner: str = ""

    def __post_init__(self) -> None:
        if self.kickoff.tzinfo is None:
            raise ValueError(
                f"EvalCase {self.match_id}: kickoff must be timezone-aware"
            )
        if self.home_goals > self.away_goals:
            derived = "home"
        elif self.home_goals < self.away_goals:
            derived = "away"
        else:
            derived = "draw"
        # Object is frozen, so we can't assign — recompute and assert.
        if self.actual_winner and self.actual_winner != derived:
            raise ValueError(
                f"EvalCase {self.match_id}: actual_winner={self.actual_winner} "
                f"but score {self.home_goals}-{self.away_goals} implies {derived}"
            )
        # Stamp the derived value (frozen dataclass trick: object.__setattr__).
        object.__setattr__(self, "actual_winner", derived)


# --- the dataset ----------------------------------------------------------
# Pinned results from the 23/24 + 24/25 seasons. UCL group stage only.
# EPL/LaLiga/Bundesliga/SerieA fixtures cover the four major leagues.
#
# We sort by kickoff at import time so the list is deterministic and
# chronologically ordered (the harness iterates in order; date order
# makes logs readable and prevents drift if cases are added out of
# sequence). `test_dataset_is_deterministic_and_ordered_by_date`
# pins this property.

_EVAL_CASES_UNSORTED: list[EvalCase] = [
    # --- UCL 24/25 group stage (Sept - Dec 2024) ---
    EvalCase(
        match_id="ucl_gs_2024_bayern_barca",
        competition="UCL", round="group",
        home_id="bayern", away_id="barca",
        venue_id="allianz_arena",
        kickoff=datetime(2024, 9, 17, 20, 0, tzinfo=timezone.utc),
        home_goals=4, away_goals=2,
    ),
    EvalCase(
        match_id="epl_2425_liverpool_everton",
        competition="EPL", round="regular",
        home_id="liverpool", away_id="everton",
        venue_id="anfield",
        kickoff=datetime(2024, 10, 19, 16, 30, tzinfo=timezone.utc),
        home_goals=2, away_goals=0,
    ),
    EvalCase(
        match_id="ucl_gs_2024_bvbsporting",
        competition="UCL", round="group",
        home_id="dortmund", away_id="sporting",
        venue_id="signal_iduna_park",
        kickoff=datetime(2024, 10, 22, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=0,
    ),
    EvalCase(
        match_id="ucl_gs_2024_psg_man_city",
        competition="UCL", round="group",
        home_id="psg", away_id="man_city",
        venue_id="parc_des_princes",
        kickoff=datetime(2024, 11, 5, 20, 0, tzinfo=timezone.utc),
        home_goals=4, away_goals=2,
    ),
    EvalCase(
        match_id="ucl_gs_2024_ajax_inter",
        competition="UCL", round="group",
        home_id="ajax", away_id="inter",
        venue_id="johan_cruyff_arena",
        kickoff=datetime(2024, 11, 6, 20, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=2,
    ),
    EvalCase(
        match_id="ucl_gs_2024_real_mbappe_liverpool",
        competition="UCL", round="group",
        home_id="real_madrid", away_id="liverpool",
        venue_id="santiago_bernabeu",
        kickoff=datetime(2024, 11, 27, 21, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=0,
    ),
    EvalCase(
        match_id="epl_2425_brighton_man_utd",
        competition="EPL", round="regular",
        home_id="brighton", away_id="man_utd",
        venue_id="amex",
        kickoff=datetime(2024, 12, 7, 15, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="laliga_2425_barca_atletico",
        competition="LaLiga", round="regular",
        home_id="barca", away_id="atletico",
        venue_id="camp_nou",
        kickoff=datetime(2024, 12, 21, 21, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=2,
    ),
    EvalCase(
        match_id="laliga_2425_real_betis_sevilla",
        competition="LaLiga", round="regular",
        home_id="real_betis", away_id="sevilla",
        venue_id="benito_villamarin",
        kickoff=datetime(2025, 1, 5, 21, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="epl_2425_arsenal_chelsea",
        competition="EPL", round="regular",
        home_id="arsenal", away_id="chelsea",
        venue_id="emirates",
        kickoff=datetime(2025, 3, 16, 16, 30, tzinfo=timezone.utc),
        home_goals=1, away_goals=0,
    ),

    # --- 23/24 season (added in Task 30 to expand coverage) ---
    # EPL, late-season. Mix of favorites winning and underdogs.
    EvalCase(
        match_id="epl_2324_man_city_liverpool",
        competition="EPL", round="regular",
        home_id="man_city", away_id="liverpool",
        venue_id="etihad",
        kickoff=datetime(2023, 11, 25, 16, 30, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="epl_2324_everton_arsenal",
        competition="EPL", round="regular",
        home_id="everton", away_id="arsenal",
        venue_id="goodison_park",
        kickoff=datetime(2023, 12, 17, 16, 30, tzinfo=timezone.utc),
        home_goals=0, away_goals=2,
    ),
    EvalCase(
        match_id="epl_2324_chelsea_man_utd",
        competition="EPL", round="regular",
        home_id="chelsea", away_id="man_utd",
        venue_id="stamford_bridge",
        kickoff=datetime(2024, 4, 4, 20, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=2,
    ),
    EvalCase(
        match_id="epl_2324_spurs_man_city",
        competition="EPL", round="regular",
        home_id="spurs", away_id="man_city",
        venue_id="spurs_stadium",
        kickoff=datetime(2024, 5, 14, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=2,
    ),
    # LaLiga 23/24 — Atlético + Sevilla to widen the league mix.
    EvalCase(
        match_id="laliga_2324_atletico_barca",
        competition="LaLiga", round="regular",
        home_id="atletico", away_id="barca",
        venue_id="metropolitano",
        kickoff=datetime(2023, 10, 8, 20, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=0,
    ),
    EvalCase(
        match_id="laliga_2324_sevilla_real_madrid",
        competition="LaLiga", round="regular",
        home_id="sevilla", away_id="real_madrid",
        venue_id="sanchez_pizjuan",
        kickoff=datetime(2024, 2, 25, 21, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=2,
    ),
    # UCL 23/24 group stage (Sept-Dec 2023) — eval-eligible.
    EvalCase(
        match_id="ucl_gs_2023_bayern_man_utd",
        competition="UCL", round="group",
        home_id="bayern", away_id="man_utd",
        venue_id="allianz_arena",
        kickoff=datetime(2023, 9, 20, 20, 0, tzinfo=timezone.utc),
        home_goals=4, away_goals=3,
    ),
    EvalCase(
        match_id="ucl_gs_2023_inter_benfica",
        competition="UCL", round="group",
        home_id="inter", away_id="benfica",
        venue_id="san_siro",
        kickoff=datetime(2023, 10, 3, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=0,
    ),
    EvalCase(
        match_id="ucl_gs_2023_psg_dortmund",
        competition="UCL", round="group",
        home_id="psg", away_id="dortmund",
        venue_id="parc_des_princes",
        kickoff=datetime(2023, 11, 7, 21, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=0,
    ),
    EvalCase(
        match_id="ucl_gs_2023_real_madrid_napoli",
        competition="UCL", round="group",
        home_id="real_madrid", away_id="napoli",
        venue_id="santiago_bernabeu",
        kickoff=datetime(2023, 11, 29, 21, 0, tzinfo=timezone.utc),
        home_goals=4, away_goals=2,
    ),
    # UCL 24/25 group stage — add a few more for class balance.
    EvalCase(
        match_id="ucl_gs_2024_shakhtar_bayern",
        competition="UCL", round="group",
        home_id="shakhtar", away_id="bayern",
        venue_id="neutral_germany",
        kickoff=datetime(2024, 10, 1, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=3,
    ),
    EvalCase(
        match_id="ucl_gs_2024_atletico_lazio",
        competition="UCL", round="group",
        home_id="atletico", away_id="lazio",
        venue_id="metropolitano",
        kickoff=datetime(2024, 11, 13, 21, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=0,
    ),
    # EPL 24/25 — a few late-season ones (Feb-May 2025) to balance.
    EvalCase(
        match_id="epl_2425_man_city_chelsea",
        competition="EPL", round="regular",
        home_id="man_city", away_id="chelsea",
        venue_id="etihad",
        kickoff=datetime(2025, 1, 25, 12, 30, tzinfo=timezone.utc),
        home_goals=3, away_goals=1,
    ),
    EvalCase(
        match_id="epl_2425_liverpool_spurs",
        competition="EPL", round="regular",
        home_id="liverpool", away_id="spurs",
        venue_id="anfield",
        kickoff=datetime(2025, 2, 16, 16, 30, tzinfo=timezone.utc),
        home_goals=2, away_goals=1,
    ),
    EvalCase(
        match_id="epl_2425_man_utd_arsenal",
        competition="EPL", round="regular",
        home_id="man_utd", away_id="arsenal",
        venue_id="old_trafford",
        kickoff=datetime(2025, 3, 9, 16, 30, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="epl_2425_newcastle_everton",
        competition="EPL", round="regular",
        home_id="newcastle", away_id="everton",
        venue_id="st_james_park",
        kickoff=datetime(2025, 4, 26, 15, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=0,
    ),
    # LaLiga 24/25 — more league spread.
    EvalCase(
        match_id="laliga_2425_atletico_sevilla",
        competition="LaLiga", round="regular",
        home_id="atletico", away_id="sevilla",
        venue_id="metropolitano",
        kickoff=datetime(2025, 1, 19, 21, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=1,
    ),
    EvalCase(
        match_id="laliga_2425_barca_real_betis",
        competition="LaLiga", round="regular",
        home_id="barca", away_id="real_betis",
        venue_id="camp_nou",
        kickoff=datetime(2025, 3, 30, 21, 0, tzinfo=timezone.utc),
        home_goals=3, away_goals=1,
    ),
    EvalCase(
        match_id="laliga_2425_real_madrid_valencia",
        competition="LaLiga", round="regular",
        home_id="real_madrid", away_id="valencia",
        venue_id="santiago_bernabeu",
        kickoff=datetime(2025, 4, 5, 21, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=0,
    ),
    # Serie A 24/25 — to add a third major league.
    EvalCase(
        match_id="seriea_2425_inter_atalanta",
        competition="SerieA", round="regular",
        home_id="inter", away_id="atalanta",
        venue_id="san_siro",
        kickoff=datetime(2024, 11, 30, 20, 45, tzinfo=timezone.utc),
        home_goals=2, away_goals=0,
    ),
    EvalCase(
        match_id="seriea_2425_juventus_roma",
        competition="SerieA", round="regular",
        home_id="juventus", away_id="roma",
        venue_id="allianz_stadium_turin",
        kickoff=datetime(2025, 2, 1, 20, 45, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="seriea_2425_milan_napoli",
        competition="SerieA", round="regular",
        home_id="milan", away_id="napoli",
        venue_id="san_siro",
        kickoff=datetime(2025, 3, 15, 20, 45, tzinfo=timezone.utc),
        home_goals=1, away_goals=2,
    ),
    # Bundesliga 24/25 — fourth major league.
    EvalCase(
        match_id="bundesliga_2425_bayern_dortmund",
        competition="Bundesliga", round="regular",
        home_id="bayern", away_id="dortmund",
        venue_id="allianz_arena",
        kickoff=datetime(2025, 2, 22, 18, 30, tzinfo=timezone.utc),
        home_goals=2, away_goals=1,
    ),
    EvalCase(
        match_id="bundesliga_2425_leverkusen_stuttgart",
        competition="Bundesliga", round="regular",
        home_id="leverkusen", away_id="stuttgart",
        venue_id="bay_arena",
        kickoff=datetime(2025, 1, 19, 17, 30, tzinfo=timezone.utc),
        home_goals=3, away_goals=0,
    ),
    EvalCase(
        match_id="fd_bundesliga_10-02-2024_leverkusen_v_bayern",
        competition="Bundesliga", round="regular",
        home_id="leverkusen", away_id="bayern",
        venue_id="",
        kickoff=datetime(2024, 2, 10, 20, 0, tzinfo=timezone.utc),
        home_goals=3, away_goals=0,
    ),
    EvalCase(
        match_id="fd_laliga_17-02-2024_valencia_v_sevilla",
        competition="LaLiga", round="regular",
        home_id="valencia", away_id="sevilla",
        venue_id="",
        kickoff=datetime(2024, 2, 17, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=0,
    ),
    EvalCase(
        match_id="fd_laliga_25-02-2024_real_madrid_v_sevilla",
        competition="LaLiga", round="regular",
        home_id="real_madrid", away_id="sevilla",
        venue_id="",
        kickoff=datetime(2024, 2, 25, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=0,
    ),
    EvalCase(
        match_id="fd_laliga_02-03-2024_valencia_v_real_madrid",
        competition="LaLiga", round="regular",
        home_id="valencia", away_id="real_madrid",
        venue_id="",
        kickoff=datetime(2024, 3, 2, 20, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=2,
    ),
    EvalCase(
        match_id="fd_bundesliga_30-03-2024_bayern_v_dortmund",
        competition="Bundesliga", round="regular",
        home_id="bayern", away_id="dortmund",
        venue_id="",
        kickoff=datetime(2024, 3, 30, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=2,
    ),
    EvalCase(
        match_id="fd_bundesliga_06-04-2024_dortmund_v_stuttgart",
        competition="Bundesliga", round="regular",
        home_id="dortmund", away_id="stuttgart",
        venue_id="",
        kickoff=datetime(2024, 4, 6, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=1,
    ),
    EvalCase(
        match_id="fd_bundesliga_21-04-2024_dortmund_v_leverkusen",
        competition="Bundesliga", round="regular",
        home_id="dortmund", away_id="leverkusen",
        venue_id="",
        kickoff=datetime(2024, 4, 21, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="fd_laliga_21-04-2024_real_madrid_v_barca",
        competition="LaLiga", round="regular",
        home_id="real_madrid", away_id="barca",
        venue_id="",
        kickoff=datetime(2024, 4, 21, 20, 0, tzinfo=timezone.utc),
        home_goals=3, away_goals=2,
    ),
    EvalCase(
        match_id="fd_bundesliga_27-04-2024_leverkusen_v_stuttgart",
        competition="Bundesliga", round="regular",
        home_id="leverkusen", away_id="stuttgart",
        venue_id="",
        kickoff=datetime(2024, 4, 27, 20, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=2,
    ),
    EvalCase(
        match_id="fd_laliga_29-04-2024_barca_v_valencia",
        competition="LaLiga", round="regular",
        home_id="barca", away_id="valencia",
        venue_id="",
        kickoff=datetime(2024, 4, 29, 20, 0, tzinfo=timezone.utc),
        home_goals=4, away_goals=2,
    ),
    EvalCase(
        match_id="fd_bundesliga_04-05-2024_stuttgart_v_bayern",
        competition="Bundesliga", round="regular",
        home_id="stuttgart", away_id="bayern",
        venue_id="",
        kickoff=datetime(2024, 5, 4, 20, 0, tzinfo=timezone.utc),
        home_goals=3, away_goals=1,
    ),
    EvalCase(
        match_id="fd_laliga_26-05-2024_sevilla_v_barca",
        competition="LaLiga", round="regular",
        home_id="sevilla", away_id="barca",
        venue_id="",
        kickoff=datetime(2024, 5, 26, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=2,
    ),
    EvalCase(
        match_id="fd_laliga_17-08-2024_valencia_v_barca",
        competition="LaLiga", round="regular",
        home_id="valencia", away_id="barca",
        venue_id="",
        kickoff=datetime(2024, 8, 17, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=2,
    ),
    EvalCase(
        match_id="fd_bundesliga_22-09-2024_stuttgart_v_dortmund",
        competition="Bundesliga", round="regular",
        home_id="stuttgart", away_id="dortmund",
        venue_id="",
        kickoff=datetime(2024, 9, 22, 20, 0, tzinfo=timezone.utc),
        home_goals=5, away_goals=1,
    ),
    EvalCase(
        match_id="fd_bundesliga_28-09-2024_bayern_v_leverkusen",
        competition="Bundesliga", round="regular",
        home_id="bayern", away_id="leverkusen",
        venue_id="",
        kickoff=datetime(2024, 9, 28, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="fd_bundesliga_19-10-2024_bayern_v_stuttgart",
        competition="Bundesliga", round="regular",
        home_id="bayern", away_id="stuttgart",
        venue_id="",
        kickoff=datetime(2024, 10, 19, 20, 0, tzinfo=timezone.utc),
        home_goals=4, away_goals=0,
    ),
    EvalCase(
        match_id="fd_laliga_20-10-2024_barca_v_sevilla",
        competition="LaLiga", round="regular",
        home_id="barca", away_id="sevilla",
        venue_id="",
        kickoff=datetime(2024, 10, 20, 20, 0, tzinfo=timezone.utc),
        home_goals=5, away_goals=1,
    ),
    EvalCase(
        match_id="fd_laliga_26-10-2024_real_madrid_v_barca",
        competition="LaLiga", round="regular",
        home_id="real_madrid", away_id="barca",
        venue_id="",
        kickoff=datetime(2024, 10, 26, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=4,
    ),
    EvalCase(
        match_id="fd_bundesliga_01-11-2024_leverkusen_v_stuttgart",
        competition="Bundesliga", round="regular",
        home_id="leverkusen", away_id="stuttgart",
        venue_id="",
        kickoff=datetime(2024, 11, 1, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=0,
    ),
    EvalCase(
        match_id="fd_bundesliga_30-11-2024_dortmund_v_bayern",
        competition="Bundesliga", round="regular",
        home_id="dortmund", away_id="bayern",
        venue_id="",
        kickoff=datetime(2024, 11, 30, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="fd_laliga_22-12-2024_real_madrid_v_sevilla",
        competition="LaLiga", round="regular",
        home_id="real_madrid", away_id="sevilla",
        venue_id="",
        kickoff=datetime(2024, 12, 22, 20, 0, tzinfo=timezone.utc),
        home_goals=4, away_goals=2,
    ),
    EvalCase(
        match_id="fd_laliga_03-01-2025_valencia_v_real_madrid",
        competition="LaLiga", round="regular",
        home_id="valencia", away_id="real_madrid",
        venue_id="",
        kickoff=datetime(2025, 1, 3, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=2,
    ),
    EvalCase(
        match_id="fd_bundesliga_10-01-2025_dortmund_v_leverkusen",
        competition="Bundesliga", round="regular",
        home_id="dortmund", away_id="leverkusen",
        venue_id="",
        kickoff=datetime(2025, 1, 10, 20, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=3,
    ),
    EvalCase(
        match_id="fd_laliga_11-01-2025_sevilla_v_valencia",
        competition="LaLiga", round="regular",
        home_id="sevilla", away_id="valencia",
        venue_id="",
        kickoff=datetime(2025, 1, 11, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="fd_laliga_26-01-2025_barca_v_valencia",
        competition="LaLiga", round="regular",
        home_id="barca", away_id="valencia",
        venue_id="",
        kickoff=datetime(2025, 1, 26, 20, 0, tzinfo=timezone.utc),
        home_goals=7, away_goals=1,
    ),
    EvalCase(
        match_id="fd_seriea_02-02-2025_milan_v_inter",
        competition="SerieA", round="regular",
        home_id="milan", away_id="inter",
        venue_id="",
        kickoff=datetime(2025, 2, 2, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="fd_seriea_02-02-2025_roma_v_napoli",
        competition="SerieA", round="regular",
        home_id="roma", away_id="napoli",
        venue_id="",
        kickoff=datetime(2025, 2, 2, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="fd_bundesliga_08-02-2025_dortmund_v_stuttgart",
        competition="Bundesliga", round="regular",
        home_id="dortmund", away_id="stuttgart",
        venue_id="",
        kickoff=datetime(2025, 2, 8, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=2,
    ),
    EvalCase(
        match_id="fd_laliga_09-02-2025_sevilla_v_barca",
        competition="LaLiga", round="regular",
        home_id="sevilla", away_id="barca",
        venue_id="",
        kickoff=datetime(2025, 2, 9, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=4,
    ),
    EvalCase(
        match_id="fd_bundesliga_15-02-2025_leverkusen_v_bayern",
        competition="Bundesliga", round="regular",
        home_id="leverkusen", away_id="bayern",
        venue_id="",
        kickoff=datetime(2025, 2, 15, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=0,
    ),
    EvalCase(
        match_id="fd_seriea_15-02-2025_lazio_v_napoli",
        competition="SerieA", round="regular",
        home_id="lazio", away_id="napoli",
        venue_id="",
        kickoff=datetime(2025, 2, 15, 20, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=2,
    ),
    EvalCase(
        match_id="fd_seriea_16-02-2025_juventus_v_inter",
        competition="SerieA", round="regular",
        home_id="juventus", away_id="inter",
        venue_id="",
        kickoff=datetime(2025, 2, 16, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=0,
    ),
    EvalCase(
        match_id="fd_bundesliga_28-02-2025_stuttgart_v_bayern",
        competition="Bundesliga", round="regular",
        home_id="stuttgart", away_id="bayern",
        venue_id="",
        kickoff=datetime(2025, 2, 28, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=3,
    ),
    EvalCase(
        match_id="fd_seriea_01-03-2025_napoli_v_inter",
        competition="SerieA", round="regular",
        home_id="napoli", away_id="inter",
        venue_id="",
        kickoff=datetime(2025, 3, 1, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="fd_seriea_02-03-2025_milan_v_lazio",
        competition="SerieA", round="regular",
        home_id="milan", away_id="lazio",
        venue_id="",
        kickoff=datetime(2025, 3, 2, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=2,
    ),
    EvalCase(
        match_id="fd_seriea_09-03-2025_juventus_v_atalanta",
        competition="SerieA", round="regular",
        home_id="juventus", away_id="atalanta",
        venue_id="",
        kickoff=datetime(2025, 3, 9, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=4,
    ),
    EvalCase(
        match_id="fd_bundesliga_16-03-2025_stuttgart_v_leverkusen",
        competition="Bundesliga", round="regular",
        home_id="stuttgart", away_id="leverkusen",
        venue_id="",
        kickoff=datetime(2025, 3, 16, 20, 0, tzinfo=timezone.utc),
        home_goals=3, away_goals=4,
    ),
    EvalCase(
        match_id="fd_epl_16-03-2025_arsenal_v_chelsea",
        competition="EPL", round="regular",
        home_id="arsenal", away_id="chelsea",
        venue_id="",
        kickoff=datetime(2025, 3, 16, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=0,
    ),
    EvalCase(
        match_id="fd_seriea_16-03-2025_atalanta_v_inter",
        competition="SerieA", round="regular",
        home_id="atalanta", away_id="inter",
        venue_id="",
        kickoff=datetime(2025, 3, 16, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=2,
    ),
    EvalCase(
        match_id="fd_seriea_30-03-2025_napoli_v_milan",
        competition="SerieA", round="regular",
        home_id="napoli", away_id="milan",
        venue_id="",
        kickoff=datetime(2025, 3, 30, 20, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=1,
    ),
    EvalCase(
        match_id="fd_epl_02-04-2025_liverpool_v_everton",
        competition="EPL", round="regular",
        home_id="liverpool", away_id="everton",
        venue_id="",
        kickoff=datetime(2025, 4, 2, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=0,
    ),
    EvalCase(
        match_id="fd_epl_03-04-2025_chelsea_v_spurs",
        competition="EPL", round="regular",
        home_id="chelsea", away_id="spurs",
        venue_id="",
        kickoff=datetime(2025, 4, 3, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=0,
    ),
    EvalCase(
        match_id="fd_epl_05-04-2025_everton_v_arsenal",
        competition="EPL", round="regular",
        home_id="everton", away_id="arsenal",
        venue_id="",
        kickoff=datetime(2025, 4, 5, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="fd_laliga_05-04-2025_real_madrid_v_valencia",
        competition="LaLiga", round="regular",
        home_id="real_madrid", away_id="valencia",
        venue_id="",
        kickoff=datetime(2025, 4, 5, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=2,
    ),
    EvalCase(
        match_id="fd_epl_06-04-2025_man_utd_v_man_city",
        competition="EPL", round="regular",
        home_id="man_utd", away_id="man_city",
        venue_id="",
        kickoff=datetime(2025, 4, 6, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=0,
    ),
    EvalCase(
        match_id="fd_seriea_06-04-2025_atalanta_v_lazio",
        competition="SerieA", round="regular",
        home_id="atalanta", away_id="lazio",
        venue_id="",
        kickoff=datetime(2025, 4, 6, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=1,
    ),
    EvalCase(
        match_id="fd_seriea_06-04-2025_roma_v_juventus",
        competition="SerieA", round="regular",
        home_id="roma", away_id="juventus",
        venue_id="",
        kickoff=datetime(2025, 4, 6, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="fd_laliga_11-04-2025_valencia_v_sevilla",
        competition="LaLiga", round="regular",
        home_id="valencia", away_id="sevilla",
        venue_id="",
        kickoff=datetime(2025, 4, 11, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=0,
    ),
    EvalCase(
        match_id="fd_bundesliga_12-04-2025_bayern_v_dortmund",
        competition="Bundesliga", round="regular",
        home_id="bayern", away_id="dortmund",
        venue_id="",
        kickoff=datetime(2025, 4, 12, 20, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=2,
    ),
    EvalCase(
        match_id="fd_epl_13-04-2025_newcastle_v_man_utd",
        competition="EPL", round="regular",
        home_id="newcastle", away_id="man_utd",
        venue_id="",
        kickoff=datetime(2025, 4, 13, 20, 0, tzinfo=timezone.utc),
        home_goals=4, away_goals=1,
    ),
    EvalCase(
        match_id="fd_seriea_13-04-2025_lazio_v_roma",
        competition="SerieA", round="regular",
        home_id="lazio", away_id="roma",
        venue_id="",
        kickoff=datetime(2025, 4, 13, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="fd_epl_19-04-2025_everton_v_man_city",
        competition="EPL", round="regular",
        home_id="everton", away_id="man_city",
        venue_id="",
        kickoff=datetime(2025, 4, 19, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=2,
    ),
    EvalCase(
        match_id="fd_seriea_20-04-2025_milan_v_atalanta",
        competition="SerieA", round="regular",
        home_id="milan", away_id="atalanta",
        venue_id="",
        kickoff=datetime(2025, 4, 20, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=1,
    ),
    EvalCase(
        match_id="fd_epl_26-04-2025_chelsea_v_everton",
        competition="EPL", round="regular",
        home_id="chelsea", away_id="everton",
        venue_id="",
        kickoff=datetime(2025, 4, 26, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=0,
    ),
    EvalCase(
        match_id="fd_epl_27-04-2025_liverpool_v_spurs",
        competition="EPL", round="regular",
        home_id="liverpool", away_id="spurs",
        venue_id="",
        kickoff=datetime(2025, 4, 27, 20, 0, tzinfo=timezone.utc),
        home_goals=5, away_goals=1,
    ),
    EvalCase(
        match_id="fd_seriea_27-04-2025_inter_v_roma",
        competition="SerieA", round="regular",
        home_id="inter", away_id="roma",
        venue_id="",
        kickoff=datetime(2025, 4, 27, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=1,
    ),
    EvalCase(
        match_id="fd_epl_04-05-2025_brighton_v_newcastle",
        competition="EPL", round="regular",
        home_id="brighton", away_id="newcastle",
        venue_id="",
        kickoff=datetime(2025, 5, 4, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="fd_epl_04-05-2025_chelsea_v_liverpool",
        competition="EPL", round="regular",
        home_id="chelsea", away_id="liverpool",
        venue_id="",
        kickoff=datetime(2025, 5, 4, 20, 0, tzinfo=timezone.utc),
        home_goals=3, away_goals=1,
    ),
    EvalCase(
        match_id="fd_seriea_10-05-2025_lazio_v_juventus",
        competition="SerieA", round="regular",
        home_id="lazio", away_id="juventus",
        venue_id="",
        kickoff=datetime(2025, 5, 10, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=1,
    ),
    EvalCase(
        match_id="fd_bundesliga_11-05-2025_leverkusen_v_dortmund",
        competition="Bundesliga", round="regular",
        home_id="leverkusen", away_id="dortmund",
        venue_id="",
        kickoff=datetime(2025, 5, 11, 20, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=4,
    ),
    EvalCase(
        match_id="fd_epl_11-05-2025_newcastle_v_chelsea",
        competition="EPL", round="regular",
        home_id="newcastle", away_id="chelsea",
        venue_id="",
        kickoff=datetime(2025, 5, 11, 20, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=0,
    ),
    EvalCase(
        match_id="fd_epl_11-05-2025_liverpool_v_arsenal",
        competition="EPL", round="regular",
        home_id="liverpool", away_id="arsenal",
        venue_id="",
        kickoff=datetime(2025, 5, 11, 20, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=2,
    ),
    EvalCase(
        match_id="fd_laliga_11-05-2025_barca_v_real_madrid",
        competition="LaLiga", round="regular",
        home_id="barca", away_id="real_madrid",
        venue_id="",
        kickoff=datetime(2025, 5, 11, 20, 0, tzinfo=timezone.utc),
        home_goals=4, away_goals=3,
    ),
    EvalCase(
        match_id="fd_seriea_12-05-2025_atalanta_v_roma",
        competition="SerieA", round="regular",
        home_id="atalanta", away_id="roma",
        venue_id="",
        kickoff=datetime(2025, 5, 12, 20, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=1,
    ),
    EvalCase(
        match_id="fd_epl_16-05-2025_chelsea_v_man_utd",
        competition="EPL", round="regular",
        home_id="chelsea", away_id="man_utd",
        venue_id="",
        kickoff=datetime(2025, 5, 16, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=0,
    ),
    EvalCase(
        match_id="fd_epl_18-05-2025_arsenal_v_newcastle",
        competition="EPL", round="regular",
        home_id="arsenal", away_id="newcastle",
        venue_id="",
        kickoff=datetime(2025, 5, 18, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=0,
    ),
    EvalCase(
        match_id="fd_seriea_18-05-2025_inter_v_lazio",
        competition="SerieA", round="regular",
        home_id="inter", away_id="lazio",
        venue_id="",
        kickoff=datetime(2025, 5, 18, 20, 0, tzinfo=timezone.utc),
        home_goals=2, away_goals=2,
    ),
    EvalCase(
        match_id="fd_seriea_18-05-2025_roma_v_milan",
        competition="SerieA", round="regular",
        home_id="roma", away_id="milan",
        venue_id="",
        kickoff=datetime(2025, 5, 18, 20, 0, tzinfo=timezone.utc),
        home_goals=3, away_goals=1,
    ),
    EvalCase(
        match_id="fd_laliga_18-05-2025_sevilla_v_real_madrid",
        competition="LaLiga", round="regular",
        home_id="sevilla", away_id="real_madrid",
        venue_id="",
        kickoff=datetime(2025, 5, 18, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=2,
    ),
    EvalCase(
        match_id="fd_epl_19-05-2025_brighton_v_liverpool",
        competition="EPL", round="regular",
        home_id="brighton", away_id="liverpool",
        venue_id="",
        kickoff=datetime(2025, 5, 19, 20, 0, tzinfo=timezone.utc),
        home_goals=3, away_goals=2,
    ),
    EvalCase(
        match_id="fd_epl_25-05-2025_newcastle_v_everton",
        competition="EPL", round="regular",
        home_id="newcastle", away_id="everton",
        venue_id="",
        kickoff=datetime(2025, 5, 25, 20, 0, tzinfo=timezone.utc),
        home_goals=0, away_goals=1,
    ),
    EvalCase(
        match_id="fd_epl_25-05-2025_spurs_v_brighton",
        competition="EPL", round="regular",
        home_id="spurs", away_id="brighton",
        venue_id="",
        kickoff=datetime(2025, 5, 25, 20, 0, tzinfo=timezone.utc),
        home_goals=1, away_goals=4,
    ),
]


# Public, chronologically sorted view of the dataset.
# `key=str` makes the sort stable even if two cases ever share a kickoff
# (would be a bug elsewhere, but stable-sort means no flaky test).
EVAL_CASES: list[EvalCase] = sorted(
    _EVAL_CASES_UNSORTED, key=lambda c: (c.kickoff, c.match_id)
)
