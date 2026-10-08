from unittest import TestCase

from ramjet.tasks.gptchat.llm.base import Index
from ramjet.tasks.gptchat.llm.embeddings import new_store
from ramjet.settings.prd import OPENAI_TOKEN, OPENAI_API


class TestVectorStore(TestCase):
    def test_save_and_load(self):
        """test_save_and_load preserves vectors, source metadata, and scanned files."""
        idx = new_store(
            apikey=OPENAI_TOKEN,
            api_base=OPENAI_API + "/v1",
        )
        idx.scaned_files.add("offline-document")
        count = idx.store.index.ntotal
        data = idx.serialize()
        restored = Index.deserialize(data=data, api_key=OPENAI_TOKEN)
        self.assertEqual(restored.scaned_files, {"offline-document"})
        self.assertEqual(restored.store.index.ntotal, count)
        document = restored.store.similarity_search("", k=1)[0]
        self.assertEqual(document.page_content, "")
        self.assertEqual(document.metadata, {"source": ""})
