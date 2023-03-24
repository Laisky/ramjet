import os
from collections import namedtuple
from ramjet.settings import OPENAI_TOKEN
from langchain.chains import VectorDBQAWithSourcesChain
from langchain import OpenAI
from ramjet.engines import thread_executor
import asyncio
from ..base import logger
from .data import load_store, prepare_data

os.environ["OPENAI_API_KEY"] = OPENAI_TOKEN

store = None

Response = namedtuple("Response", ["question", "text", "url"])

def setup():
    prepare_data()
    global store
    store = load_store()
    logger.info("succeed setup basebit data")

async def query(question: str):
    return await asyncio.get_running_loop().run_in_executor(thread_executor, _query, question)

def _query(question: str) -> Response:
    chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
    resp = chain({"question": question})
    return Response(question=question, text=resp["answer"], url=resp.get("sources", ""))
