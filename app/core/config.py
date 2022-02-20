# import os
#
# from pydantic import BaseSettings, Field
#
#
# class Settings(BaseSettings):
#     db_url: str = Field(..., env='DATABASE_URL')
#
# settings = Settings()

from databases import DatabaseURL
import os
from pydantic import BaseSettings, Field, SecretStr
# from starlette.config import Config
# from starlette.datastructures import Secret
#
# config = Config(".env")
#
# PROJECT_NAME = "Tunr"
# VERSION = "1.0.0"
#
# SECRET_KEY = config("SECRET_KEY", cast=Secret, default="CHANGEME")
#
# POSTGRES_USER = config("POSTGRES_USER", cast=str)
# POSTGRES_PASSWORD = config("POSTGRES_PASSWORD", cast=Secret)
# POSTGRES_SERVER = config("POSTGRES_SERVER", cast=str, default="db")
# POSTGRES_PORT = config("POSTGRES_PORT", cast=str, default="5432")
# POSTGRES_DB = config("POSTGRES_DB", cast=str)

#DATABASE_URL = config(
#  "DATABASE_URL",
#  cast=DatabaseURL,
#  default=f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"
#)

basepath = os.path.dirname(__file__)

class Settings(BaseSettings):
  db_url: str = Field(..., env='DATABASE_URL')
  db_username: str = Field(..., env="POSTGRES_USER")
  db_password: str = Field(..., env="POSTGRES_PASSWORD")
  db_server: str = Field(..., env="POSTGRES_SERVER")
  db_port: str = Field(..., env="POSTGRES_PORT")
  db_db: str = Field(..., env="POSTGRES_DB")
  project_name: str = Field(..., env="PROJECT_NAME")
  project_version: str = Field(..., env="PROJECT_VERSION")

  # JWT settings
  JWT_SECRET: str = "TEST_SECRET_DO_NOT_USE_IN_PROD"
  ALGORITHM: str = "HS256"
  ACCESS_TOKEN_EXPIRE_MINUTES: int = 60*24*8  # 60 minutes * 24 hours * 8 days

  class Config:
    env_file = os.path.join(os.path.join(basepath, os.pardir), '.env')

settings = Settings()