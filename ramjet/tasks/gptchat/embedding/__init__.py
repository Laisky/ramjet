import os
from collections import namedtuple
from textwrap import dedent

from ramjet import settings
from langchain.chains import VectorDBQAWithSourcesChain, RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from ramjet.engines import thread_executor
import asyncio

from ..base import logger
from .data import load_all_stores, prepare_data

if settings.OPENAI_TYPE == "openai":
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_TOKEN
    os.environ["OPENAI_API_BASE"] = settings.OPENAI_API
elif settings.OPENAI_TYPE == "azure":
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = settings.OPENAI_AZURE_VERSION
    os.environ["OPENAI_API_BASE"] = settings.OPENAI_AZURE_API
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_AZURE_TOKEN

    azure_embeddings_deploymentid = settings.OPENAI_AZURE_DEPLOYMENTS[
        "embeddings"
    ].deployment_id
    azure_gpt_deploymentid = settings.OPENAI_AZURE_DEPLOYMENTS["chat"].deployment_id


all_chains = {}
Response = namedtuple("Response", ["question", "text", "url"])


def setup():
    prepare_data()
    global all_chains

    system_template = dedent(
        """
        Use the following pieces of context to answer the users question.
        Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
        If you don't know the answer, just say that "I don't know", don't try to make up an answer.
        ----------------
        {summaries}
        """
    )
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    for project_name, store in load_all_stores().items():
        chain_type_kwargs = {"prompt": prompt}

        if settings.OPENAI_TYPE == "openai":
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                max_tokens=1000,
                streaming=False,
            )
        elif settings.OPENAI_TYPE == "azure":
            llm = AzureChatOpenAI(
                deployment_name="gpt35",
                model_name="gpt-3.5-turbo",
                temperature=0,
                max_tokens=1000,
                streaming=False,
            )

        # all_chains[project_name] = VectorDBQAWithSourcesChain.from_chain_type(
        #     llm=llm,
        #     vectorstore=store,
        #     return_source_documents=True,
        #     chain_type_kwargs=chain_type_kwargs,
        #     reduce_k_below_max_tokens=True,
        # )

        all_chains[project_name] = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs,
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
