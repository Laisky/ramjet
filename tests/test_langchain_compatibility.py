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


def test_openai_compatible_client_preserves_custom_endpoint(monkeypatch):
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
            chat_model="gpt-4o-mini",
            apikey="public-offline-key",
            api_base="https://offline.invalid/custom/v1",
        )
        model = query.build_llm_for_user(user)
        assert model.invoke("offline question").content == "offline answer"
    assert len(requests) == 1
    assert str(requests[0].url) == "https://offline.invalid/custom/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"] == [{"role": "user", "content": "offline question"}]
    assert body.get("max_completion_tokens", body.get("max_tokens")) == 500
    assert body["stream"] is False
    assert requests[0].headers["Authorization"] == "Bearer public-offline-key"
