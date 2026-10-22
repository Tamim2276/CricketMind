"""Every engine, reading one DuckDB connection (Day 46)."""

from cricketmind_v3.application.engine_source import EngineFactSource, Repositories
from cricketmind_v3.infrastructure.duckdb_repo import DuckDBDeliveryRepository
from cricketmind_v3.infrastructure.duckdb_results import DuckDBResultRepository
from cricketmind_v3.infrastructure.duckdb_reviews import DuckDBReviewRepository
from cricketmind_v3.infrastructure.duckdb_venues import DuckDBVenueRepository
from cricketmind_v3.infrastructure.duckdb_workload import DuckDBWorkloadRepository
from cricketmind_v3.infrastructure.snapshot import data_fingerprint


def engines_on(db) -> EngineFactSource:
    """The real FactSource on an open DuckDB connection (read-only is enough)."""
    return EngineFactSource(
        Repositories(
            deliveries=DuckDBDeliveryRepository(db),
            venues=DuckDBVenueRepository(db),
            workload=DuckDBWorkloadRepository(db),
            reviews=DuckDBReviewRepository(db),
            results=DuckDBResultRepository(db),
        ),
        data=data_fingerprint(db),
    )
