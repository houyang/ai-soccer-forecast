import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from soccer_agent.db.base import Base
from soccer_agent.db.models import Competition, Team, Match, Prediction


@pytest_asyncio.fixture
async def engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session(engine):
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


@pytest.mark.asyncio
async def test_competition_crud(session: AsyncSession):
    comp = Competition(
        id="premier_league",
        name="Premier League",
        type="league",
        api_source="api-football",
        current_season="2024-25"
    )
    session.add(comp)
    await session.commit()

    result = await session.get(Competition, "premier_league")
    assert result.name == "Premier League"
    assert result.type == "league"


@pytest.mark.asyncio
async def test_match_with_prediction(session: AsyncSession):
    from datetime import datetime

    # Create teams
    home = Team(id="team_a", name="Team A", api_source="api-football")
    away = Team(id="team_b", name="Team B", api_source="api-football")
    comp = Competition(id="pl", name="PL", type="league", api_source="api-football")
    session.add_all([home, away, comp])
    await session.flush()

    # Create match
    match = Match(
        id="match_1",
        competition_id="pl",
        home_team_id="team_a",
        away_team_id="team_b",
        kickoff_utc=datetime(2025, 6, 1, 19, 45, 0),
        status="upcoming"
    )
    session.add(match)
    await session.flush()

    # Create prediction
    pred = Prediction(
        match_id="match_1",
        predicted_outcome="home",
        confidence_score=78.0,
        rationale="Strong home form",
        reasoning_json={"form": 0.8, "h2h": 0.2},
        tools_used=["form", "h2h"]
    )
    session.add(pred)
    await session.commit()

    result = await session.get(Prediction, 1)
    assert result.predicted_outcome == "home"
    assert result.confidence_score == 78.0