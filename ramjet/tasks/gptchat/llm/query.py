import os
from collections import namedtuple
from textwrap import dedent

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from ..base import logger
from .data import load_all_stores, prepare_data
from .embeddings import build_chain, N_NEAREST_CHUNKS


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

        n_chunk = N_NEAREST_CHUNKS
        if os.environ.get("OPENAI_API_TYPE", "") == "azure":
            llm = AzureChatOpenAI(
                client=None,
                deployment_name="gpt35",
                # model_name="gpt-3.5-turbo",
                temperature=0,
                max_tokens=1000,
                streaming=False,
            )
        else:
            n_chunk = 10
            llm = ChatOpenAI(
                client=None,
                model="gpt-3.5-turbo-16k",
                temperature=0,
                max_tokens=4000,
                streaming=False,
            )

        # all_chains[project_name] = RetrievalQAWithSourcesChain.from_chain_type(
        #     llm=llm,
        #     chain_type="stuff",
        #     retriever=store.as_retriever(),
        #     return_source_documents=True,
        #     chain_type_kwargs=chain_type_kwargs,
        #     reduce_k_below_max_tokens=True,
        # )

        all_chains[project_name] = build_chain(llm=llm, store=store, nearest_k=n_chunk)
        logger.info(f"load chain for project: {project_name}")

    logger.info("all chains have been setup")


def query(project_name: str, question: str) -> Response:
    resp, refs = all_chains[project_name](question)
    # return Response(question=question, text=resp["answer"], url=resp.get("sources", ""))
    return Response(question=question, text=resp, url=list(set(refs)))


def classificate_query_type(query: str, apikey: str = None) -> str:
    """classify query type by user's query

    Args:
        query (str): user's query

    Returns:
        str: query type, 'search' or 'scan'
    """
    prompt = dedent(
        f"""
            there are some types of task, including search and scan.
            you should judge the task type by user's query and answer the exact type of task in your opinion,
            do not answer any other words.

            for example, if the query is "summary this", you should answer "scan".
            for example, if the query is "what is TEE's abilitity", you should answer "search".
            for example, if the query is "这是啥", you should answer "scan".

            the user's query is between "@>>>>>" and "@<<<<<":
            @>>>>>
            {query}
            @<<<<<
            your answer is:"""
    )
    apikey = apikey or os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(
        client=None,
        openai_api_key=apikey,
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=200,
        streaming=False,
    )
    task_type = llm.predict(prompt)
    if "scan" in task_type:
        return "scan"
    else:
        return "search"  # default
