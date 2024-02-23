from unittest import TestCase

from ramjet.tasks.gptchat.llm.base import Index
from ramjet.tasks.gptchat.llm.embeddings import new_store
from ramjet.settings.prd import OPENAI_TOKEN, OPENAI_API


class TestVectorStore(TestCase):
    def test_save_and_load(self):
        idx = new_store(
            apikey=OPENAI_TOKEN,
            api_base=OPENAI_API + "/v1",
        )
        data = idx.serialize()
        idx = Index.deserialize(data=data, api_key=OPENAI_TOKEN)
