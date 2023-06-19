import os

from ramjet.settings import prd
from .router import bind_handle


if prd.OPENAI_TYPE == "azure":
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = prd.OPENAI_AZURE_API + "/"
    os.environ["OPENAI_API_KEY"] = prd.OPENAI_AZURE_TOKEN

    azure_embeddings_deploymentid = "embedding"
    azure_gpt_deploymentid = "gpt35"
elif prd.OPENAI_TYPE == "openai":
    os.environ["OPENAI_API_KEY"] = prd.OPENAI_TOKEN
