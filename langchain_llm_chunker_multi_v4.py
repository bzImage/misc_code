#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Demon - bzImage
# 
"""
Markdown with embedded images (docling output)-> Id and save Images -> Pre‑chunk -> Semantic merge -> LLM chunking -> FAISS

Stages / artifacts:
  0) read_md                           -> in‑memory
  1) replace_base64_images             -> <faiss_dir>/images/ + <input>_images_extracted.md
  2) pre_chunk (headers+tables)        -> in‑memory list[str]
  3) semantic_chunking (emb-merge)     -> in‑memory list[str], metrics
  4) process_chunks (LLM JSON chunks)  -> <faiss_dir>/artifacts/<doc_id>_chunks.jsonl
  5) FAISS index update                -> <faiss_dir>/faiss_index/

Basic metrics are logged and summarized at the end.
Verbose, colorized logging controllable via -v / --log-level.
"""

import os
import base64
import uuid
import sys
import json
import time
import glob
import regex as re
import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from pydantic import BaseModel, Field, ValidationError, validator

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

import numpy as np

# ---------------- Colors / Logging ----------------

class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

class ColorFormatter(logging.Formatter):
    def __init__(self, fmt, use_color: bool):
        super().__init__(fmt)
        self.use_color = use_color

    def _wrap(self, color, msg):
        return f"{color}{msg}{Colors.RESET}" if self.use_color else msg

    def format(self, record):
        msg = super().format(record)
        if record.levelno == logging.INFO:
            return self._wrap(Colors.CYAN, msg)
        elif record.levelno == logging.WARNING:
            return self._wrap(Colors.YELLOW, msg)
        elif record.levelno == logging.ERROR:
            return self._wrap(Colors.RED, msg)
        elif record.levelno == logging.DEBUG:
            return self._wrap(Colors.MAGENTA, msg)
        return msg

def build_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("md_pipeline")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        use_color = sys.stdout.isatty()
        handler.setFormatter(ColorFormatter("%(asctime)s [%(levelname)s] %(message)s", use_color))
        logger.addHandler(handler)
    return logger

logger = build_logger("INFO")

# ---------------- Config ----------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# Chunking
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# LLM
# KISS: fijo y seguro; el API recorta si es necesario.
MAX_COMPLETION_TOKENS = 3000
TEMPERATURE = 0.0

# Filtering
MIN_LENGTH = 50  # min number of characters to send (FIX: was MIN_LENGHT)
PNG_MIN_BYTES = 0  # no min here; only extraction; size filtering belongs upstream if needed

# ---------------- Models ----------------

class Chunk(BaseModel):
    title: Optional[str] = Field("", description="Section title or empty")
    text: str = Field(..., description="Self-contained text chunk")

    @validator("title", pre=True, always=True)
    def set_title_empty_string_if_none(cls, v):
        return v or ""

class ChunkList(BaseModel):
    chunks: List[Chunk]

json_parser = JsonOutputParser(pydantic_object=ChunkList)

# ---------------- Utils ----------------

def sha1_hex(data: bytes) -> str:
    import hashlib
    return hashlib.sha1(data).hexdigest()

def read_md(path: str) -> str:
    logger.info(f"Reading Markdown file: {Colors.BOLD}{path}{Colors.RESET}")
    with open(path, encoding="utf-8") as f:
        content = f.read()
    logger.debug(f"Read {len(content)} characters.")
    return content

def write_text(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(data)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ---------------- Stage 1: Embedded images → files + token markers ----------------

IMG_PATTERN = r"!\[.*?\]\(data:image/[^;]+;base64,([^)]+)\)"

def png_is_valid(sig: bytes) -> bool:
    return len(sig) >= 8 and sig[:8] == b"\x89PNG\r\n\x1a\n"

def png_dims_from_bytes(img: bytes) -> Optional[Tuple[int, int]]:
    # PNG: 8-byte signature + 4(len) + 4('IHDR') + 4(width) + 4(height)
    if len(img) >= 24 and img[12:16] == b'IHDR':
        w = int.from_bytes(img[16:20], 'big')
        h = int.from_bytes(img[20:24], 'big')
        return (w, h) if w > 0 and h > 0 else None
    return None

def replace_base64_images(md_text: str, document_id: str, image_dir: Path) -> Tuple[str, Dict[str, Any]]:
    logger.info(f"{Colors.BOLD}{Colors.CYAN}.:: Embedded images saver/replacer ::.{Colors.RESET}")
    ensure_dir(image_dir)

    replacements = 0
    total_bytes = 0
    details: List[Dict[str, Any]] = []

    def _repl(m: re.Match) -> str:
        nonlocal replacements, total_bytes
        b64_str = m.group(1)
        img_bytes = base64.b64decode(b64_str, validate=False)
        total_bytes += len(img_bytes)

        valid = png_is_valid(img_bytes)
        dims = png_dims_from_bytes(img_bytes)
        width, height = (dims or (None, None))

        image_uuid = str(uuid.uuid4())
        image_filename = f"{document_id}_{image_uuid}.png"
        image_path = image_dir / image_filename

        with image_path.open("wb") as f:
            f.write(img_bytes)

        replacements += 1
        details.append({
            "file": str(image_path),
            "bytes": len(img_bytes),
            "width": width,
            "height": height,
            "valid_png": valid,
        })

        logger.info(
            f"Saved image {Colors.BOLD}{image_filename}{Colors.RESET} "
            f"bytes={len(img_bytes)} dims={width}x{height} valid_png={valid}"
        )
        return f"<img:{document_id}_{image_uuid}>"

    new_text = re.sub(IMG_PATTERN, _repl, md_text)
    logger.info(f"Saved & replaced {Colors.BOLD}{replacements}{Colors.RESET} base64 image(s), total_bytes={total_bytes}.")
    metrics = {"count": replacements, "total_bytes": total_bytes, "images": details}
    return new_text, metrics

# ---------------- Stage 2: Header-aware pre-chunk + table preserving ----------------

def detect_tables(text: str) -> List[Tuple[int, int]]:
    table_spans = []
    lines = text.splitlines()
    in_table = False
    start_idx = 0
    current_pos = 0

    for i, line in enumerate(lines):
        if not in_table:
            if '|' in line:
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if re.match(r'^\s*\|?(\s*:?-+:?\s*\|)+\s*$', next_line):
                        in_table = True
                        start_idx = current_pos
        else:
            if line.strip() == '' or '|' not in line:
                end_idx = current_pos - 1
                table_spans.append((start_idx, end_idx))
                in_table = False
        current_pos += len(line) + 1

    if in_table:
        table_spans.append((start_idx, len(text)))

    logger.debug(f"Detected {len(table_spans)} table(s).")
    return table_spans

def pre_chunk(text: str) -> List[str]:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    sections = md_splitter.split_text(text)
    logger.info(f"Split into {Colors.BOLD}{len(sections)}{Colors.RESET} header sections.")

    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: List[str] = []

    for section in sections:
        section_text = section.page_content
        tables = detect_tables(section_text)

        if not tables:
            chunks.extend(recursive_splitter.split_text(section_text))
        else:
            last_idx = 0
            for start, end in tables:
                if start > last_idx:
                    pre_table_text = section_text[last_idx:start]
                    chunks.extend(recursive_splitter.split_text(pre_table_text))
                table_text = section_text[start:end]
                chunks.append(table_text)
                last_idx = end
            if last_idx < len(section_text):
                post_table_text = section_text[last_idx:]
                chunks.extend(recursive_splitter.split_text(post_table_text))

    logger.info(f"Pre-chunk produced {Colors.BOLD}{len(chunks)}{Colors.RESET} chunks.")
    return chunks

# ---------------- Stage 3: Semantic chunking (embedding merge) ----------------

def semantic_chunking(text: str, embeddings: OpenAIEmbeddings, similarity_threshold: float = 0.8) -> Tuple[List[str], Dict[str, Any]]:
    initial_chunks = pre_chunk(text)
    logger.info(f"Semantic merge starting with {Colors.BOLD}{len(initial_chunks)}{Colors.RESET} chunks.")

    t0 = time.time()
    logger.info("Embedding initial chunks...")
    chunk_embeddings = embeddings.embed_documents(initial_chunks)
    emb_time = time.time() - t0
    logger.info(f"Embeddings computed in {emb_time:.2f}s.")

    merged_chunks: List[str] = []
    merges = 0

    if not initial_chunks:
        return merged_chunks, {
            "initial_chunks": 0,
            "merged_chunks": 0,
            "merges": 0,
            "embed_time_sec": emb_time,
        }

    buffer = initial_chunks[0]
    buffer_emb = np.array(chunk_embeddings[0], dtype=np.float32)

    for i in range(1, len(initial_chunks)):
        emb = np.array(chunk_embeddings[i], dtype=np.float32)
        denom = (np.linalg.norm(buffer_emb) * np.linalg.norm(emb)) + 1e-10
        similarity = float(np.dot(buffer_emb, emb) / denom)
        logger.debug(f"similarity[{i}]={similarity:.3f}")

        if similarity > similarity_threshold:
            buffer += "\n\n" + initial_chunks[i]
            buffer_emb = (buffer_emb + emb) / 2.0
            merges += 1
        else:
            merged_chunks.append(buffer)
            buffer = initial_chunks[i]
            buffer_emb = emb
    merged_chunks.append(buffer)

    metrics = {
        "initial_chunks": len(initial_chunks),
        "merged_chunks": len(merged_chunks),
        "merges": merges,
        "embed_time_sec": emb_time,
    }
    logger.info(
        f"Semantic merge: merges={Colors.BOLD}{merges}{Colors.RESET} "
        f"final_chunks={Colors.BOLD}{len(merged_chunks)}{Colors.RESET}"
    )
    return merged_chunks, metrics

# ---------------- Stage 4: LLM-driven chunk shaping ----------------

def build_prompt_with_schema(part: str) -> List[dict]:
    format_instructions = json_parser.get_format_instructions()
    system_prompt = (
        "You are an expert text chunker specialized in Markdown documents. "
        "Split the following Markdown text into semantic blocks, each with a concise title and self-contained text.\n\n"
        "- Preserve Markdown tables; do not split tables across chunks.\n"
        "- Keep Markdown formatting (tables, lists, code).\n"
        "- Each block ≤ 300 words.\n"
        "- Respond with ONE JSON object ONLY, matching the schema below.\n"
        "- No extra text, no headings, no code fences.\n"
        "- Max 5 chunks per response.\n"
        "- Keep original language.\n"
        "- If input has no meaningful content, return an empty list.\n\n"
        f"{format_instructions}\n\n"
        "Now split the following text:\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": part}
    ]

def call_llm_with_retries(messages: List[dict], max_tokens: int = MAX_COMPLETION_TOKENS) -> str:
    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            finish = response.choices[0].finish_reason
            logger.debug(f"LLM finish_reason={finish}")
            if finish == "length":
                logger.warning("LLM indicated length truncation; retrying with smaller max_tokens.")
                max_tokens = max(int(max_tokens * 0.75), 512)
                continue
            return content
        except OpenAIError as e:
            logger.warning(f"LLM call failed attempt {attempt}/3: {e}")
            if attempt < 3:
                time.sleep(5 + attempt)
            else:
                raise

def process_chunks(parts: List[str], artifacts_dir: Path, doc_id: str) -> Tuple[List[Chunk], Dict[str, Any]]:
    result: List[Chunk] = []
    total_chars = sum(len(p) for p in parts)
    logger.info(
        f"Processing {Colors.BOLD}{len(parts)}{Colors.RESET} chunks "
        f"total_chars={Colors.BOLD}{total_chars}{Colors.RESET}"
    )

    ensure_dir(artifacts_dir)
    out_jsonl = artifacts_dir / f"{doc_id}_chunks.jsonl"
    created = 0
    calls = 0
    t0 = time.time()

    with out_jsonl.open("w", encoding="utf-8") as jf:
        for idx, part in enumerate(parts, 1):
            if len(part.strip()) < MIN_LENGTH:
                logger.warning(f"Skip chunk {idx}: too short ({len(part)} chars).")
                continue

            logger.info(f"LLM shaping {Colors.BOLD}{idx}/{len(parts)}{Colors.RESET} (len={len(part)})")
            messages = build_prompt_with_schema(part)
            calls += 1
            raw_output = call_llm_with_retries(messages)

            try:
                parsed = json_parser.parse(raw_output)
                # parsed is pydantic ChunkList
                count = len(parsed.chunks)
                created += count
                logger.info(f"LLM returned {Colors.BOLD}{count}{Colors.RESET} chunks.")
                for c in parsed.chunks:
                    result.append(Chunk(title=c.title, text=c.text))
                    jf.write(json.dumps({"title": c.title, "text": c.text}, ensure_ascii=False) + "\n")

            except OutputParserException as e:
                logger.error(f"Output parsing error at part {idx}: {e}")
                logger.debug(f"LLM raw output (first 200 chars): {raw_output[:200]!r}")
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                logger.error(f"JSON validation error at part {idx}: {e}")
                logger.debug(f"LLM raw output (first 200 chars): {raw_output[:200]!r}")

    elapsed = time.time() - t0
    metrics = {
        "llm_calls": calls,
        "llm_chunks_created": created,
        "elapsed_sec": elapsed,
        "jsonl_path": str(out_jsonl),
    }
    logger.info(
        f"LLM shaping done: calls={Colors.BOLD}{calls}{Colors.RESET} "
        f"chunks={Colors.BOLD}{created}{Colors.RESET} "
        f"time={Colors.BOLD}{elapsed:.2f}s{Colors.RESET} -> {out_jsonl}"
    )
    return result, metrics

# ---------------- Stage 5: FAISS ----------------

def faiss_index_exists(path: str) -> bool:
    # LC FAISS writes multiple files under a directory; existence check should be directory + files present.
    return os.path.isdir(path) and any(Path(path).iterdir())

def extract_id_from_filename(filename: str) -> str:
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    # Prefer content hash to avoid collisions by underscores.
    try:
        with open(filename, "rb") as f:
            content = f.read()
        h = sha1_hex(content)[:12]
        return f"{name}_{h}"
    except Exception:
        return name

def load_indexed_ids(index_dir: str) -> set:
    path = os.path.join(index_dir, "indexed_ids.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_indexed_ids(index_dir: str, ids: set):
    path = os.path.join(index_dir, "indexed_ids.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f, indent=2, ensure_ascii=False)

def add_documents_to_faiss(vectorstore: FAISS, chunks: List[Chunk], doc_id: str):
    texts = []
    metadatas = []
    for c in chunks:
        text = f"{c.title}\n{c.text}" if c.title else c.text
        texts.append(text)
        metadatas.append({"source_id": doc_id})
    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
    vectorstore.add_documents(docs)
    logger.info(f"Added {len(docs)} docs to FAISS index (source_id={doc_id}).")

def create_faiss_index(chunks: List[Chunk], output_dir: str, doc_id: str):
    logger.info("Creating new FAISS index...")
    texts = []
    metadatas = []
    for c in chunks:
        text = f"{c.title}\n{c.text}" if c.title else c.text
        texts.append(text)
        metadatas.append({"source_id": doc_id})

    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)

    faiss_path = os.path.join(output_dir, "faiss_index")
    vectorstore.save_local(faiss_path)
    logger.info(f"FAISS index saved at: {faiss_path}")
    return vectorstore

def load_or_create_faiss_index(faiss_path: str, embeddings: OpenAIEmbeddings) -> Optional[FAISS]:
    if faiss_index_exists(faiss_path):
        logger.info(f"Loading existing FAISS index from {faiss_path}")
        # allow_dangerous_deserialization for LC >= 0.2.x
        return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    else:
        return None

def interactive_query_with_sources_from_file(vectorstore: FAISS, index_dir: str, document_limit: int):
    path = os.path.join(index_dir, "indexed_ids.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            source_ids = json.load(f)
    else:
        source_ids = []

    if source_ids:
        print("\nIndexed documents (source_ids):")
        for sid in source_ids:
            print(f" - {sid}")
    else:
        print("\nNo documents found in the index file.")

    print(f"\nEnter a query (type 'exit' or 'quit' to stop). Will retrieve up to {document_limit} docs.")
    while True:
        q = input("> ")
        if q.lower() in ["exit", "quit"]:
            break
        results = vectorstore.similarity_search(q, k=document_limit)
        for i, res in enumerate(results, 1):
            source_id = res.metadata.get("source_id", "unknown")
            print(f"\nResult {i} (source_id={source_id}):\n{res.page_content}\n{'-'*40}")

# ---------------- Pipeline Orchestration ----------------

def main(input_pattern: str, faiss_dir: str, log_level: str = "INFO"):
    global logger
    logger = build_logger(log_level)

    faiss_dir_p = Path(faiss_dir)
    ensure_dir(faiss_dir_p)
    ensure_dir(faiss_dir_p / "images")
    ensure_dir(faiss_dir_p / "artifacts")

    faiss_path = str(faiss_dir_p / "faiss_index")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = load_or_create_faiss_index(faiss_path, embeddings)

    indexed_ids = load_indexed_ids(faiss_dir)

    files = glob.glob(input_pattern)
    if not files:
        logger.error(f"No files matched pattern: {input_pattern}")
        sys.exit(1)

    pipeline_metrics: Dict[str, Any] = {
        "files": len(files),
        "images_saved": 0,
        "images_bytes": 0,
        "llm_calls": 0,
        "llm_chunks": 0,
        "emb_time_sec": 0.0,
        "semantic_initial_chunks": 0,
        "semantic_final_chunks": 0,
        "processing_sec": 0.0,
    }

    t_pipeline = time.time()

    for filepath in files:
        doc_id = extract_id_from_filename(filepath)
        logger.info(
            f"Processing file '{Colors.BOLD}{filepath}{Colors.RESET}' "
            f"doc_id='{Colors.BOLD}{doc_id}{Colors.RESET}'"
        )

        if doc_id in indexed_ids:
            logger.info(f"Document id '{Colors.BOLD}{doc_id}{Colors.RESET}' already indexed. Skipping.")
            continue

        # Stage 0: read
        text = read_md(filepath)

        # Stage 1: extract images + replace
        images_dir = faiss_dir_p / "images"
        text_replaced, img_metrics = replace_base64_images(text, doc_id, images_dir)

        # Persist artifact for transparency
        extracted_md_path = Path(filepath).with_suffix("")  # drop .md suffix
        extracted_md_path = faiss_dir_p / "artifacts" / f"{Path(extracted_md_path).name}_images_extracted.md"
        write_text(extracted_md_path, text_replaced)
        logger.info(f"Images-extracted MD written: {Colors.BOLD}{extracted_md_path}{Colors.RESET}")

        pipeline_metrics["images_saved"] += img_metrics["count"]
        pipeline_metrics["images_bytes"] += img_metrics["total_bytes"]

        # Stage 2+3: semantic chunking
        sem_chunks, sem_metrics = semantic_chunking(text_replaced, embeddings)
        pipeline_metrics["emb_time_sec"] += sem_metrics["embed_time_sec"]
        pipeline_metrics["semantic_initial_chunks"] += sem_metrics["initial_chunks"]
        pipeline_metrics["semantic_final_chunks"] += sem_metrics["merged_chunks"]

        # Stage 4: LLM shaping
        chunks, llm_metrics = process_chunks(sem_chunks, faiss_dir_p / "artifacts", doc_id)
        pipeline_metrics["llm_calls"] += llm_metrics["llm_calls"]
        pipeline_metrics["llm_chunks"] += llm_metrics["llm_chunks"]

        if not chunks:
            logger.error(f"No chunks created for file {filepath}. Skipping.")
            continue

        # Stage 5: FAISS
        if vectorstore:
            add_documents_to_faiss(vectorstore, chunks, doc_id)
        else:
            base_dir = faiss_dir
            vectorstore = create_faiss_index(chunks, base_dir, doc_id)

        vectorstore.save_local(faiss_path)
        logger.info(f"FAISS index updated: {Colors.BOLD}{faiss_path}{Colors.RESET}")

        indexed_ids.add(doc_id)
        save_indexed_ids(faiss_dir, indexed_ids)

    pipeline_metrics["processing_sec"] = time.time() - t_pipeline

    # ---- Summary ----
    logger.info(
        f"{Colors.BOLD}{Colors.BLUE}Pipeline summary:{Colors.RESET}\n"
        f"  files_processed     : {pipeline_metrics['files']}\n"
        f"  images_saved        : {pipeline_metrics['images_saved']} "
        f"({pipeline_metrics['images_bytes']} bytes)\n"
        f"  emb_time_sec        : {pipeline_metrics['emb_time_sec']:.2f}\n"
        f"  semantic_chunks     : initial={pipeline_metrics['semantic_initial_chunks']} "
        f"→ final={pipeline_metrics['semantic_final_chunks']}\n"
        f"  llm_calls           : {pipeline_metrics['llm_calls']}\n"
        f"  llm_chunks_created  : {pipeline_metrics['llm_chunks']}\n"
        f"  total_time_sec      : {pipeline_metrics['processing_sec']:.2f}"
    )

    if not vectorstore:
        logger.error("No FAISS index available after processing files. Exiting.")
        sys.exit(1)

    # Optional interactive search
    try:
        interactive_query_with_sources_from_file(vectorstore, faiss_dir, document_limit=5)
    except KeyboardInterrupt:
        print()

# ---------------- Entrypoint ----------------

if __name__ == "__main__":

    print(".:: Markdown file to FAISS database v4 ::.")
    print("\nGimme a markdown with images and i return a faiss directory with images\n")

    if not (3 <= len(sys.argv) <= 4):
        logger.error("Usage: python langchain_llm_chunker.py <input_pattern.md> <faiss_directory> [LOG_LEVEL]")
        sys.exit(1)

    pattern = sys.argv[1]
    outdir = sys.argv[2]
    lvl = sys.argv[3] if len(sys.argv) == 4 else "INFO"
    main(pattern, outdir, lvl)
