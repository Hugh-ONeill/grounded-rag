"""Postgres + pgvector storage. This boilerplate is done; the schema is the interesting bit."""
import psycopg
from psycopg.types.json import Json
from pgvector.psycopg import register_vector
from config import settings


def connect():
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn


def init_schema():
    """Create the extension + chunks table. Idempotent."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          BIGSERIAL PRIMARY KEY,
                source      TEXT NOT NULL,      -- citation label, e.g. "gen9ou_chaos.json#Heatran"
                title       TEXT,
                content     TEXT NOT NULL,
                metadata    JSONB DEFAULT '{{}}',
                embedding   vector({settings.embed_dim})
            )
        """)
        # ANN index for fast cosine search. ivfflat needs data first; create after first ingest
        # or use hnsw (better recall, no training):
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops)
        """)
        # For V2 hybrid retrieval, add a full-text index:
        # cur.execute("CREATE INDEX IF NOT EXISTS chunks_fts ON chunks USING gin (to_tsvector('english', content))")
        conn.commit()


def insert_chunks(rows):
    """rows: iterable of (source, title, content, metadata_dict, embedding_list)."""
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (source, title, content, metadata, embedding) VALUES (%s,%s,%s,%s,%s)",
            [(s, t, c, Json(m), e) for s, t, c, m, e in rows],
        )
        conn.commit()


def clear():
    with connect() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE chunks RESTART IDENTITY")
        conn.commit()
