from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str = "OPENAI_API_KEY"
    LLM_MODEL: str = "LLM_MODEL"
    EMBEDDING_MODEL: str = "EMBEDDING_MODEL"

    class Config:
        env_file = ".env"


settings = Settings()
