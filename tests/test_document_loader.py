import importlib.util
import unittest
from pathlib import Path


MISSING_DOCUMENT_DEPS = any(
    importlib.util.find_spec(package_name) is None
    for package_name in ("pandas", "pypdf")
)


@unittest.skipIf(
    MISSING_DOCUMENT_DEPS,
    "Document loader tests require pandas and pypdf from requirements.txt.",
)
class DocumentLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utils.document_loader import DocumentLoader

        cls.DocumentLoader = DocumentLoader

    def test_load_txt_rejects_empty_file(self):
        loader = self.DocumentLoader()

        with self.assertRaises(ValueError):
            loader.load_txt(b"   \n")

    def test_load_csv_converts_rows_to_readable_text(self):
        loader = self.DocumentLoader()
        csv_bytes = b"name,team\nAva,Data Science\nBen,Operations\n"

        text = loader.load_csv(csv_bytes)

        self.assertIn("Columns: name, team", text)
        self.assertIn("Row 1: name: Ava; team: Data Science", text)
        self.assertIn("Row 2: name: Ben; team: Operations", text)

    def test_chunk_text_uses_sentence_overlap(self):
        loader = self.DocumentLoader(chunk_size=65, chunk_overlap=25)
        text = (
            "Alpha team builds dashboards. "
            "Beta team monitors deliveries. "
            "Gamma team reviews documents."
        )

        chunks = loader.chunk_text(text)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("Beta team monitors deliveries.", chunks[0])
        self.assertIn("Beta team monitors deliveries.", chunks[1])

    def test_chunk_text_can_split_long_sentences(self):
        loader = self.DocumentLoader(chunk_size=20, chunk_overlap=5)
        chunks = loader.chunk_text("A" * 45)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 20 for chunk in chunks))

    def test_load_sample_pdf_extracts_text(self):
        loader = self.DocumentLoader()
        pdf_path = Path("sample_documents/sample_policy_document.pdf")

        text = loader.load_pdf(pdf_path.read_bytes())

        self.assertIn("Sample Policy Document", text)
        self.assertIn("source context", text)


if __name__ == "__main__":
    unittest.main()
