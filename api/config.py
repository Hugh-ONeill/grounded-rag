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
    top_k: int = 5
    # Tuned via eval/: nomic-embed cosine sims run hot (on-topic ~0.64-0.78,
    # off-topic ~0.40-0.54). 0.30 never fires; 0.60 splits the two cleanly.
    min_similarity: float = 0.60

    # nomic-embed-text is 768-dim. Change if you swap embedding models.
    embed_dim: int = 768

    class Config:
        env_file = ".env"


settings = Settings()
