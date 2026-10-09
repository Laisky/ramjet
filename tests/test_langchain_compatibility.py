"""Qualify the existing retrieval chain and OpenAI-compatible client behavior."""

import json

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from ramjet.settings import UserPermission
from ramjet.tasks.gptchat.llm import embeddings, query


@pytest.mark.parametrize(
    "responses, expected_refs",
    [
        (["offline answer"], ["offline://source"]),
        (
            ["I need more information about: topic", "offline answer"],
            ["offline://source", "offline://source"],
        ),
    ],
)
def test_real_qa_chain_preserves_answer_and_references(responses, expected_refs):
    """test_real_qa_chain_preserves_answer_and_references checks retrieval and follow-up queries."""
    store = embeddings.FAISS.from_documents(
        [
            embeddings.Document(
                page_content="offline source context",
                metadata={"source": "offline://source"},
            )
        ],
        embeddings.OpenAIEmbeddings(),
    )
    chain = embeddings.build_chain(store, nearest_k=1)
    answer, refs = chain(FakeListChatModel(responses=responses), "offline question")
    assert answer == "offline answer"
    assert refs == expected_refs


@pytest.mark.parametrize("model_name", ["gpt-4o-mini", "custom-model"])
def test_openai_compatible_client_preserves_custom_endpoint(monkeypatch, model_name):
    """test_openai_compatible_client_preserves_custom_endpoint checks the real SDK using a local transport."""
    requests = []

    def respond(request):
        """respond returns a synthetic completion and records the exact request."""
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "offline-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "offline answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    real_chat_model = query.ChatOpenAI
    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as client:

        def configured_model(**kwargs):
            """configured_model supplies an in-memory transport to the unchanged production constructor."""
            return real_chat_model(http_client=client, **kwargs)

        monkeypatch.setattr(query, "ChatOpenAI", configured_model)
        user = UserPermission(
            is_paid=False,
            uid="offline",
            n_concurrent=1,
            chat_model=model_name,
            apikey="public-offline-key",
            api_base="https://offline.invalid/custom/v1",
        )
        model = query.build_llm_for_user(user)
        assert model.invoke("offline question").content == "offline answer"
    assert len(requests) == 1
    assert str(requests[0].url) == "https://offline.invalid/custom/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["model"] == model_name
    assert body["messages"] == [{"role": "user", "content": "offline question"}]
    assert body.get("max_completion_tokens", body.get("max_tokens")) == 500
    assert body["stream"] is False
    assert requests[0].headers["Authorization"] == "Bearer public-offline-key"


def test_real_embeddings_client_preserves_public_sdk_behavior():
    """test_real_embeddings_client_preserves_public_sdk_behavior checks the unpatched embedding SDK."""
    from langchain_openai.embeddings.base import OpenAIEmbeddings

    requests = []

    def respond(request):
        """respond supplies an in-memory embedding response and records its request."""
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIEmbeddings(
            api_key="public-offline-key",
            model="text-embedding-3-small",
            base_url="https://offline.invalid/embedding/v1",
            http_client=client,
        )
        assert model.embed_documents(["offline text"]) == [[0.1, 0.2, 0.3]]
    assert len(requests) == 1
    assert str(requests[0].url) == "https://offline.invalid/embedding/v1/embeddings"
    body = json.loads(requests[0].content)
    assert body["model"] == "text-embedding-3-small"
    assert len(body["input"]) == 1
    assert body["encoding_format"] == "base64"


@pytest.mark.parametrize("encoding", ["utf-8", "gbk"])
def test_real_markdown_ingestion_preserves_encodings_and_sources(tmp_path, encoding):
    """test_real_markdown_ingestion_preserves_encodings_and_sources checks active splitters."""
    path = tmp_path / "offline.md"
    path.write_bytes("# 中文标题\nOffline markdown text".encode(encoding))
    chunks = embeddings.split_markdown(
        str(path),
        "offline://markdown",
        max_chunks=5,
        chunk_size=64,
        chunk_overlap=0,
    )
    assert chunks
    assert "中文标题" in "".join(chunk.text for chunk in chunks)
    assert all(
        chunk.metadata["source"].startswith("offline://markdown#") for chunk in chunks
    )


def test_real_docx_ingestion_preserves_text_and_sources(tmp_path):
    """test_real_docx_ingestion_preserves_text_and_sources checks the supported DOCX loader."""
    import zipfile

    path = tmp_path / "offline.docx"
    xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Offline DOCX 中文</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml.encode("utf-8"))
    chunks = embeddings.split_msword(
        str(path),
        "offline://docx",
        max_chunks=5,
        chunk_size=64,
        chunk_overlap=0,
    )
    assert chunks
    assert "Offline DOCX 中文" in "".join(chunk.text for chunk in chunks)
    assert all(
        chunk.metadata["source"].startswith("offline://docx#page=") for chunk in chunks
    )


def test_legacy_vector_store_remains_readable():
    """test_legacy_vector_store_remains_readable preserves existing serialized index compatibility."""
    import hashlib
    from pathlib import Path

    folder = Path(__file__).parent / "fixtures" / "legacy_vector_store"
    expected = {
        "index.faiss": "41e1888f62a020a33abcc286d20c56cb971155711259f09b7a9fa6c8b2cc55a0",
        "index.pkl": "b3b5316fc2c5508188d2e8869a3f2c2a335c0ea8785fe28ec6f1337decbe461d",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    # Only the hash-verified repository fixture is trusted for this offline load.
    store = embeddings.FAISS.load_local(
        str(folder),
        embeddings.OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    documents = store.similarity_search("Legacy offline context", k=1)
    assert len(documents) == 1
    assert documents[0].page_content == "Legacy offline context"
    assert documents[0].metadata["source"] == "offline://legacy-vector"
