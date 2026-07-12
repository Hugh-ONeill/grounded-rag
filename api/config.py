"""Central config, loaded from environment / .env (see .env.example)."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    corpus: str = "markdown_dir"
    corpus_path: str = "./docs"
    crystal_battle_path: str = ""

    ollama_host: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    llm_model: str = "gemma4:26b-a4b-it-q4_K_M"

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    min_similarity: float = 0.30

    # nomic-embed-text is 768-dim. Change if you swap embedding models.
    embed_dim: int = 768

    class Config:
        env_file = ".env"


settings = Settings()
