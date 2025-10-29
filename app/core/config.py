# Directory: HelixRAG/app/core/config.py
from typing import Literal, Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # -------- Supabase --------
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_anon_key: str = Field(..., env="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(..., env="SUPABASE_SERVICE_ROLE_KEY")

    # -------- Provider --------
    ai_provider: Literal["openai", "anthropic", "aliyun"] = Field(default="aliyun", env="AI_PROVIDER")

    # -------- 阿里云通义千问--------
    aliyun_api_key: str = Field(..., env="ALIYUN_API_KEY")
    aliyun_chat_model: str = Field(default="qwen-plus", env="ALIYUN_CHAT_MODEL")
    aliyun_embed_model: str = Field(default="text-embedding-v4", env="ALIYUN_EMBED_MODEL")
    dashscope_api_key: str = Field(default="", env="DASHSCOPE_API_KEY")

    #保留 OpenAI/Anthropic 字段以兼容老逻辑
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="", env="OPENAI_CHAT_MODEL")
    openai_embed_model: str = Field(default="", env="OPENAI_EMBED_MODEL")

    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    anthropic_chat_model: str = Field(default="claude-3-5-sonnet-20241022", env="ANTHROPIC_CHAT_MODEL")

    # -------- App / RAG --------
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    default_top_k: int = Field(default=6)
    chunk_size: int = Field(default=400)
    chunk_overlap: int = Field(default=60)
    temperature: float = Field(default=0.1)
    embedding_dimensions: int = Field(default=1024)
    
    # -------- Enhanced RAG Features --------
    enable_reranking: bool = Field(default=True, env="ENABLE_RERANKING")
    rerank_top_n: int = Field(default=6, env="RERANK_TOP_N")
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    supported_file_types: str = Field(default="pdf,docx,txt,md,html,csv", env="SUPPORTED_FILE_TYPES")

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

def get_settings() -> Settings:
    return settings

# Convenience accessor so the rest of the codebase can rely on a single
# Qwen/DashScope credential without requiring duplicate env vars.
def get_effective_dashscope_key() -> str:
    """Return DashScope API key, falling back to ALIYUN_API_KEY when unset."""
    return settings.dashscope_api_key or settings.aliyun_api_key
