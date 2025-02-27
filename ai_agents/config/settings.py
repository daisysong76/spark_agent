import os
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # Twilio
    TWILIO_ACCOUNT_SID: str = Field(..., env="TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN: str = Field(..., env="TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER: str = Field(..., env="TWILIO_PHONE_NUMBER")

    # SendGrid
    SENDGRID_API_KEY: str = Field(..., env="SENDGRID_API_KEY")
    EMAIL_FROM: str = Field(default="noreply@yourdomain.com", env="EMAIL_FROM")

    # LinkedIn
    LINKEDIN_ACCESS_TOKEN: str = Field(..., env="LINKEDIN_ACCESS_TOKEN")

    # Google Custom Search
    GOOGLE_API_KEY: str = Field(..., env="GOOGLE_API_KEY")
    GOOGLE_CX: str = Field(..., env="GOOGLE_CX")  # Search Engine ID

    # Milvus
    MILVUS_HOST: str = Field(default="localhost", env="MILVUS_HOST")
    MILVUS_PORT: str = Field(default="19530", env="MILVUS_PORT")
    MILVUS_COLLECTION_NAME: str = Field(default="documents", env="MILVUS_COLLECTION_NAME")

    # Model
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    GENERATIVE_MODEL: str = Field(default="gpt2", env="GENERATIVE_MODEL")

    # Others (Logging, environment, etc.)
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"  # optional local dev file
        case_sensitive = True


settings = Settings()
