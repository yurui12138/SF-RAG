"""Public SF-RAG indexing and answer-generation interface."""

import json
from pathlib import Path

from sf_retriever import SFRetriever
from structure_index import INDEX_VERSION, StructureIndexer


class SFRAG:
    def __init__(self, config, index_path="output/sf_index.json", retrieval_config=None):
        self.index_path = Path(index_path)
        options = {
            "alpha": 0.5, "beta": 0.8, "sections": 2, "paths": 3,
            "hops": 3, "entity_threshold": 0.7, "token_budget": 2048,
            "rerank_candidates": 64, "top_n": 64,
            "embedding_model": config.embedding_model,
        }
        if retrieval_config:
            options.update(retrieval_config)
        self.indexer = StructureIndexer(config, threshold=0.70, segment_tokens=512)
        self.retriever = SFRetriever(config, options, str(self.index_path))

    async def build_index(self, source="files"):
        """Index MinerU *_content_list.json files in the SF-RAG format."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists():
            corpus = json.loads(self.index_path.read_text(encoding="utf-8"))
            if not isinstance(corpus.get("papers"), dict):
                raise ValueError("Invalid SF-RAG index")
        else:
            corpus = {"papers": {}}
        indexed = {paper["__meta__"]["source_folder"]: (title, paper["__meta__"])
                   for title, paper in corpus["papers"].items()}
        for folder in sorted(Path(source).iterdir()):
            if not folder.is_dir():
                continue
            existing = indexed.get(folder.name)
            if existing and (existing[1].get("generator_model") == self.indexer.config.model
                             and existing[1].get("index_version") == INDEX_VERSION
                             and existing[1].get("segment_tokens") == self.indexer.segment_tokens):
                continue
            paper = await self.indexer.build(folder)
            paper["__meta__"].update(generator_model=self.indexer.config.model,
                                     index_version=INDEX_VERSION,
                                     segment_tokens=self.indexer.segment_tokens)
            if existing:
                del corpus["papers"][existing[0]]
            corpus["papers"][paper["__meta__"]["title"]] = paper
            temporary = self.index_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(corpus, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
            temporary.replace(self.index_path)
        return list(corpus["papers"])

    def list_papers(self):
        if not self.index_path.exists():
            return []
        return list(json.loads(self.index_path.read_text(encoding="utf-8"))["papers"])

    async def answer(self, question, papers, multi_hop=False):
        """Multi-hop decomposition is opt-in; multi-paper queries are transformed per paper."""
        if not papers:
            raise ValueError("Select at least one indexed paper")
        return await self.retriever.answer(question, papers, multi_hop=multi_hop)
