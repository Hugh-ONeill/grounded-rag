"""Central config, loaded from environment / .env (see .env.example)."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    corpus: str = "markdown_dir"
    corpus_path: str = "./docs"
    crystal_battle_path: str = ""
    pokeapi_path: str = ""

    ollama_host: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    llm_model: str = "gemma4:26b-a4b-it-q4_K_M"

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    chunk_size: int = 900
    chunk_overlap: int = 150
    # 8, not 5: with multiple corpora, near-duplicate docs (a species' learnset,
    # pokedex entry, usage stats) crowd the top ranks; k=5 starved the generator
    top_k: int = 8
    # Tuned via eval/, re-tuned when the corpus grew 10x: on-topic top-1 sims run
    # 0.67-0.84, off-topic up to 0.65 (topically-adjacent chunks creep up as the
    # corpus grows; the margin narrowed from ~0.10 to ~0.02). A score-based gate
    # (reranker) is the durable fix -- see roadmap.
    min_similarity: float = 0.66
    # Second gate signal: weighted keyword rank of the best passage. Entity-named
    # questions ("Is a Sitrus Berry better...") measure >=0.32 via title matches;
    # vocabulary coincidences ("boiling point of water" -> Water Gem) cap at ~0.18.
    min_keyword_rank: float = 0.25

    # Cross-encoder reranker: rescoring (question, passage) pairs after hybrid
    # retrieval. Also the gate's primary signal when enabled: its scores measure
    # "does this passage answer this question", so they stay comparable as the
    # corpus grows, unlike raw cosine bands (see README).
    use_reranker: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Measured over the gold set: answerable questions bottom out at 0.099
    # (telegraphic stat text scores low even when it directly answers), off-topic
    # tops at 0.059. MiniLM chosen over bge-reranker-v2-m3, whose off-topic floor
    # (0.500) sat inside its answerable band.
    min_rerank_score: float = 0.08

    # nomic-embed-text is 768-dim. Change if you swap embedding models.
    embed_dim: int = 768

    class Config:
        env_file = ".env"


settings = Settings()
