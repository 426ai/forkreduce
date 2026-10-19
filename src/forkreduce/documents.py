"""Documents and blocks frozen into the shared prefix."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from forkreduce.errors import ForkReduceError
from forkreduce.textutil import strip_control_chars

END_MARKER = "===END_FIXED_DOCUMENTS==="


class Block(BaseModel):
    """One addressable span of a document.

    Citation checks look up ``block_id`` and search for a quote inside ``text``.
    """

    block_id: str
    text: str
    meta: dict[str, str] = Field(default_factory=dict)


class Document(BaseModel):
    """A document made of blocks. ``doc_id`` is stable and appears in the rendered prefix."""

    doc_id: str
    title: str = ""
    blocks: list[Block]


@dataclass
class DocumentIndex:
    """Cleaned documents plus the exact text that is frozen into the prefix."""

    documents: list[Document]
    blocks: dict[str, Block] = field(default_factory=dict)
    rendered: str = ""

    def block_text(self, block_id: str) -> str | None:
        block = self.blocks.get(block_id)
        if block is None:
            return None
        return block.text


DocumentsInput = str | Document | Block | Sequence[str | Document | Block]


def coerce_documents(documents: DocumentsInput) -> DocumentIndex:
    """Normalize caller input into cleaned blocks and a stable rendering.

    A bare string becomes one document with a single block ``b1``. A sequence of
    strings becomes blocks ``b1``, ``b2``, ... inside one document. Control
    characters are stripped before the text is hashed or shown to the model, so a
    null byte cannot break JSON tool arguments later.
    """
    docs = _coerce_many(documents)
    if not docs:
        raise ForkReduceError("documents must contain at least one block of text")
    blocks: dict[str, Block] = {}
    for doc in docs:
        if not doc.blocks:
            raise ForkReduceError(f"document {doc.doc_id!r} has no blocks")
        for block in doc.blocks:
            if not block.block_id:
                raise ForkReduceError("block_id must be a non-empty string")
            if block.block_id in blocks:
                raise ForkReduceError(f"duplicate block_id {block.block_id!r}")
            if not block.text.strip():
                raise ForkReduceError(f"block {block.block_id!r} is empty after cleaning")
            blocks[block.block_id] = block
    rendered = render_documents(docs)
    return DocumentIndex(documents=docs, blocks=blocks, rendered=rendered)


def render_documents(documents: Sequence[Document]) -> str:
    """Render documents into the frozen prefix body, ending with :data:`END_MARKER`."""
    lines: list[str] = []
    for doc in documents:
        title = f" title={doc.title}" if doc.title else ""
        lines.append(f"[document id={doc.doc_id}{title}]")
        for block in doc.blocks:
            lines.append(f"[block {block.block_id}]")
            lines.append(block.text)
            lines.append("")
    lines.append(END_MARKER)
    return "\n".join(lines)


def _coerce_many(documents: DocumentsInput) -> list[Document]:
    if isinstance(documents, str):
        return [_string_document("doc", [documents])]
    if isinstance(documents, Document):
        return [_clean_document(documents)]
    if isinstance(documents, Block):
        return [_blocks_document("doc", [documents])]
    if isinstance(documents, Sequence):
        strings: list[str] = []
        docs: list[Document] = []
        loose_blocks: list[Block] = []
        for item in documents:
            if isinstance(item, str):
                strings.append(item)
            elif isinstance(item, Document):
                docs.append(_clean_document(item))
            elif isinstance(item, Block):
                loose_blocks.append(item)
            else:
                raise ForkReduceError("documents must be a string, Block, Document, or a sequence of those")
        if strings and not docs and not loose_blocks:
            return [_string_document("doc", strings)]
        if loose_blocks:
            docs.append(_blocks_document("doc", loose_blocks))
        if strings:
            docs.append(_string_document("doc-text", strings))
        return docs
    raise ForkReduceError("documents must be a string, Block, Document, or a sequence of those")


def _string_document(doc_id: str, parts: list[str]) -> Document:
    blocks: list[Block] = []
    for index, part in enumerate(parts, start=1):
        blocks.append(Block(block_id=f"b{index}", text=strip_control_chars(part)))
    return Document(doc_id=doc_id, blocks=blocks)


def _blocks_document(doc_id: str, blocks: Sequence[Block]) -> Document:
    cleaned = [
        Block(block_id=block.block_id, text=strip_control_chars(block.text), meta=dict(block.meta)) for block in blocks
    ]
    return Document(doc_id=doc_id, blocks=cleaned)


def _clean_document(document: Document) -> Document:
    return Document(
        doc_id=document.doc_id,
        title=strip_control_chars(document.title),
        blocks=[
            Block(block_id=block.block_id, text=strip_control_chars(block.text), meta=dict(block.meta))
            for block in document.blocks
        ],
    )
