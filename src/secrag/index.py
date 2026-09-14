import json
import logging
import uuid
from dataclasses import asdict

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from secrag.chunking import WINDOW, bge_spans, fixed_chunks, structural_chunks
from secrag.config import DATA_DIR, EMBED_MODEL, QDRANT_URL

CHUNKS_DIR = DATA_DIR / "chunks"
log = logging.getLogger("secrag.index")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    spans = bge_spans()
    docs = [json.loads(p.read_text()) for p in sorted((DATA_DIR / "sections").glob("*.json"))]
    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(url=QDRANT_URL)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    for strategy, chunker in [("fixed", fixed_chunks), ("structural", structural_chunks)]:
        chunks = [c for doc in docs for c in chunker(doc, spans)]
        oversized = sum(len(spans(c.text)) > WINDOW for c in chunks)
        log.info(
            "%s: %d chunks from %d filings, %d over %d tokens",
            strategy,
            len(chunks),
            len(docs),
            oversized,
            WINDOW,
        )
        # BM25 is built from this file at query time; Qdrant only holds the dense side.
        with open(CHUNKS_DIR / f"{strategy}.jsonl", "w") as f:
            f.writelines(json.dumps(asdict(c)) + "\n" for c in chunks)

        vectors = model.encode_document(
            [c.text for c in chunks],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        name = f"chunks_{strategy}"
        if client.collection_exists(name):
            client.delete_collection(name)
        client.create_collection(
            name,
            vectors_config=models.VectorParams(
                size=vectors.shape[1], distance=models.Distance.COSINE
            ),
        )
        client.create_payload_index(name, "ticker", models.PayloadSchemaType.KEYWORD)
        client.create_payload_index(name, "fiscal_year", models.PayloadSchemaType.INTEGER)
        client.upload_points(
            name,
            (
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                    vector=v.tolist(),
                    payload=asdict(c),
                )
                for c, v in zip(chunks, vectors, strict=True)
            ),
            wait=True,
        )
        log.info("%s: %d points indexed", name, client.count(name).count)


if __name__ == "__main__":
    main()
