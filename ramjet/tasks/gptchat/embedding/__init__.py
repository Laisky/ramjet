import os
from collections import namedtuple
from ramjet.settings import OPENAI_TOKEN
from langchain.chains import VectorDBQAWithSourcesChain
from langchain import OpenAI
from ramjet.engines import thread_executor
import asyncio
from ..base import logger
from .data import load_all_stores, prepare_data

os.environ["OPENAI_API_KEY"] = OPENAI_TOKEN

all_chains = {}

Response = namedtuple("Response", ["question", "text", "url"])


def setup():
    prepare_data()
    global all_chains
    for project_name, store in load_all_stores().items():
        all_chains[project_name] = VectorDBQAWithSourcesChain.from_llm(
            llm=OpenAI(
                temperature=0,
                max_tokens=1000,
                model_name="text-davinci-003",
                streaming=False,
            ),
            vectorstore=store,
            reduce_k_below_max_tokens=True,
        )

        logger.info(f"load chain for project: {project_name}")


async def query(project_name: str, question: str):
    return await asyncio.get_running_loop().run_in_executor(
        thread_executor, _query, project_name, question
    )


def _query(project_name: str, question: str) -> Response:
    resp = all_chains[project_name](
        {"question": question},
        return_only_outputs=True,
    )
    return Response(question=question, text=resp["answer"], url=resp.get("sources", ""))
