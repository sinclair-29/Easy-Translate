import importlib.util
import os
import tempfile
import unittest
from unittest import mock

import epub_converter


HAS_EPUB_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("ebooklib", "bs4", "lxml")
)


class EpubDependencyErrors(unittest.TestCase):
    def test_missing_dependencies_raise_helpful_error(self):
        with mock.patch.object(epub_converter, "ebooklib", None):
            with self.assertRaisesRegex(ImportError, "pip install ebooklib"):
                epub_converter.require_epub_dependencies()


@unittest.skipUnless(HAS_EPUB_DEPS, "EPUB dependencies are not installed")
class EpubConverter(unittest.TestCase):
    def create_book(self, path):
        from ebooklib import epub

        book = epub.EpubBook()
        book.set_identifier("tiny-book")
        book.set_title("Tiny Book")
        book.set_language("en")

        style = epub.EpubItem(
            uid="style",
            file_name="style/book.css",
            media_type="text/css",
            content="body { color: #222; }",
        )
        image = epub.EpubItem(
            uid="cover-image",
            file_name="images/pixel.png",
            media_type="image/png",
            content=(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
                b"\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
                b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
            ),
        )

        chapter_one = epub.EpubHtml(
            title="Chapter One",
            file_name="chapter1.xhtml",
            lang="en",
            uid="chapter-one",
        )
        chapter_one.content = """
        <html><head><link href="style/book.css" rel="stylesheet" /></head>
        <body>
          <h1>Chapter One</h1>
          <p>First paragraph. Second sentence.</p>
          <ul><li>Second item.</li></ul>
          <blockquote><p>Quoted line.</p></blockquote>
        </body></html>
        """

        chapter_two = epub.EpubHtml(
            title="Chapter Two",
            file_name="chapter2.xhtml",
            lang="en",
            uid="chapter-two",
        )
        chapter_two.content = """
        <html><body>
          <h2>Chapter Two</h2>
          <figure>
            <img src="images/pixel.png" />
            <figcaption>Caption text.</figcaption>
          </figure>
          <dl><dt>Definition</dt><dd>Description text.</dd></dl>
        </body></html>
        """

        book.add_item(style)
        book.add_item(image)
        book.add_item(chapter_one)
        book.add_item(chapter_two)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.toc = (chapter_one, chapter_two)
        book.spine = ["nav", chapter_one, chapter_two]
        epub.write_epub(path, book)

    def test_epub_to_text_extracts_spine_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            epub_path = os.path.join(tmpdirname, "source.epub")
            text_path = os.path.join(tmpdirname, "source.txt")
            manifest_path = os.path.join(tmpdirname, "manifest.json")
            self.create_book(epub_path)

            epub_converter.epub_to_text(epub_path, text_path, manifest_path)

            with open(text_path, "r", encoding="utf-8") as text_file:
                lines = [line.rstrip("\n") for line in text_file]

            self.assertEqual(
                lines,
                [
                    "Chapter One",
                    "First paragraph.",
                    "Second sentence.",
                    "Second item.",
                    "Quoted line.",
                    "Chapter Two",
                    "Caption text.",
                    "Definition",
                    "Description text.",
                ],
            )
            self.assertTrue(os.path.exists(manifest_path))

    def test_text_to_epub_rebuilds_book_and_preserves_assets(self):
        from ebooklib import epub

        with tempfile.TemporaryDirectory() as tmpdirname:
            source_epub_path = os.path.join(tmpdirname, "source.epub")
            source_text_path = os.path.join(tmpdirname, "source.txt")
            translated_text_path = os.path.join(tmpdirname, "translated.txt")
            manifest_path = os.path.join(tmpdirname, "manifest.json")
            output_epub_path = os.path.join(tmpdirname, "translated.epub")
            self.create_book(source_epub_path)

            epub_converter.epub_to_text(
                source_epub_path,
                source_text_path,
                manifest_path,
            )
            with open(translated_text_path, "w", encoding="utf-8") as text_file:
                for line in (
                    "Capitulo Uno",
                    "Primer parrafo.",
                    "Segunda frase.",
                    "Segundo elemento.",
                    "Linea citada.",
                    "Capitulo Dos",
                    "Texto de pie.",
                    "Definicion",
                    "Texto descriptivo.",
                ):
                    print(line, file=text_file)

            epub_converter.text_to_epub(
                source_epub_path,
                translated_text_path,
                manifest_path,
                output_epub_path,
            )

            rebuilt = epub.read_epub(output_epub_path)
            self.assertEqual(rebuilt.get_metadata("DC", "title")[0][0], "Tiny Book")
            self.assertIsNotNone(rebuilt.get_item_with_id("cover-image"))
            chapter = rebuilt.get_item_with_id("chapter-one").get_content().decode()
            self.assertIn("Capitulo Uno", chapter)
            self.assertIn("Primer parrafo. Segunda frase.", chapter)

    def test_text_to_epub_rejects_line_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            source_epub_path = os.path.join(tmpdirname, "source.epub")
            source_text_path = os.path.join(tmpdirname, "source.txt")
            translated_text_path = os.path.join(tmpdirname, "translated.txt")
            manifest_path = os.path.join(tmpdirname, "manifest.json")
            output_epub_path = os.path.join(tmpdirname, "translated.epub")
            self.create_book(source_epub_path)
            epub_converter.epub_to_text(
                source_epub_path,
                source_text_path,
                manifest_path,
            )
            with open(translated_text_path, "w", encoding="utf-8") as text_file:
                print("Only one line", file=text_file)

            with self.assertRaisesRegex(ValueError, "line count"):
                epub_converter.text_to_epub(
                    source_epub_path,
                    translated_text_path,
                    manifest_path,
                    output_epub_path,
                )
