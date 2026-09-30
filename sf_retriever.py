"""Three-mode SF-RAG retrieval over structure-fidelity indexes."""

import asyncio
import json
import math
import re

import requests
import tiktoken
from openai import AsyncOpenAI

from sf_prompts import decompose_question, generate_answer, transform_question
from structure_index import INDEX_VERSION, token_count

def cosine(a, b):
    numerator = sum(x * y for x, y in zip(a, b))
    denominator = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return numerator / denominator if denominator else 0.0


class SFRetriever:
    def __init__(self, config, retrieval_config, index_path="output/sf_index.json"):
        self.config = config
        self.retrieval_config = retrieval_config
        self.llm = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self.index_path = index_path
        self.alpha = retrieval_config.get("alpha", 0.5)
        self.beta = retrieval_config.get("beta", 0.8)
        self.section_limit = retrieval_config.get("sections", 2)
        self.path_limit = retrieval_config.get("paths", 3)
        self.budget = retrieval_config.get("token_budget", 2048)
        self.embedding_model = retrieval_config.get("embedding_model", "text-embedding-3-small")

    async def _call_llm(self, prompt, max_tokens, temperature):
        for attempt in range(3):
            try:
                response = await self.llm.chat.completions.create(
                    model=self.config.model, messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens, temperature=temperature)
                return response.choices[0].message.content.strip()
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(2)

    async def rerank_contents(self, query, items):
        payload = {
            "model": self.config.rerank_model, "query": query,
            "documents": [re.sub(r"\s+", " ", item.strip()) for item in items],
            "top_n": min(len(items), self.retrieval_config.get("top_n", 64)),
            "return_documents": False,
        }
        response = await asyncio.to_thread(
            requests.post, self.config.rerank_url, json=payload,
            headers={"Authorization": f"Bearer {self.config.rerank_api_key}"},
            timeout=60)
        response.raise_for_status()
        return response.json().get("results", [])

    async def _embed(self, texts):
        if not texts:
            return []
        output = []
        for offset in range(0, len(texts), 16):
            batch = texts[offset:offset + 16]
            response = await self.llm.embeddings.create(model=self.embedding_model, input=batch)
            output.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
        return output

    async def _section_scores(self, query, sections):
        prompt = (
            "Score how well the conventional role of each academic section title matches "
            "the question intent, independently of its content. Return ONLY a JSON object "
            'mapping each provided path to a number in [0,1].\n'
            f"Question: {query}\nSections: {json.dumps({p: s['title'] for p, s in sections.items()})}"
        )
        try:
            response = await self._call_llm(prompt, 1200, 0)
            semantic = json.loads(response)
            if not isinstance(semantic, dict):
                semantic = {}
        except (ValueError, TypeError):
            semantic = {}
        encoder = tiktoken.get_encoding("cl100k_base")
        chunks, ranges = [], []
        for path in sections:
            descendants = [g for child_path, section in sections.items()
                           if child_path == path or (
                               path.count("/") > 1 and child_path.startswith(path + "/"))
                           for g in section["groups"]]
            text = " ".join(g["content"] for g in descendants).strip() or sections[path]["title"]
            tokens = encoder.encode(text)
            start = len(chunks)
            for offset in range(0, len(tokens), 6000):
                part = tokens[offset:offset + 6000]
                chunks.append((encoder.decode(part), len(part)))
            ranges.append((start, len(chunks)))
        # The dense section signal is the body embedding, never a mixture with summaries.
        vectors = await self._embed([query] + [chunk for chunk, _ in chunks])
        scores = {}
        for i, path in enumerate(sections):
            try:
                alignment = max(0, min(1, float(semantic.get(path, 0))))
            except (ValueError, TypeError):
                alignment = 0
            start, end = ranges[i]
            if end - start == 1:
                section_vector = vectors[start + 1]
            else:
                # A single encoder call cannot accept an arbitrarily long section.
                total = sum(chunks[j][1] for j in range(start, end))
                section_vector = [
                    sum(vectors[j + 1][k] * chunks[j][1] for j in range(start, end)) / total
                    for k in range(len(vectors[0]))
                ]
            scores[path] = self.alpha * alignment + (
                1 - self.alpha) * cosine(vectors[0], section_vector)
        return scores

    async def _score_segments(self, query, groups):
        vectors = await self._embed([query] + [g["content"] for g in groups]
                                    + [g.get("summary") or g["content"] for g in groups])
        count = len(groups)
        for i, group in enumerate(groups):
            group["score"] = self.beta * cosine(vectors[0], vectors[i + 1]) + (
                1 - self.beta) * cosine(vectors[0], vectors[count + i + 1])
            group["segment_cost"] = max(1, group.get("token_cost") or token_count(group["content"]))
            group["cost"] = max(1, token_count(
                self.format_context(group.get("paper_title", ""), [group]) + "\n\n") + 1)
        # Cross-encoder refines ordering after dual-channel dense scoring.
        if groups:
            candidates = sorted(groups, key=lambda g: g["score"], reverse=True)[
                :self.retrieval_config.get("rerank_candidates", 64)]
            results = await self.rerank_contents(query, [g["content"] for g in candidates])
            for rank in results:
                index = rank.get("index")
                if isinstance(index, int) and 0 <= index < len(candidates):
                    candidates[index]["rerank_score"] = rank["relevance_score"]
            groups.sort(key=lambda g: (g.get("rerank_score", float("-inf")), g["score"]),
                        reverse=True)
        return groups

    @staticmethod
    def _unique_groups(paper, selected_paths, mode):
        found = {}
        for section in paper.values():
            if not isinstance(section, dict) or "path" not in section:
                continue
            path = section["path"]
            if selected_paths is not None and not any(
                path == prefix or (prefix.count("/") > 1 and path.startswith(prefix + "/"))
                for prefix in selected_paths
            ):
                continue
            for group in section.get("groups", []):
                if not group.get("content"):
                    continue
                index = group.get("global_index")
                key = (path, index if index is not None else group["content"])
                found[key] = dict(group, path=path)
        return sorted(found.values(), key=lambda g: g.get("global_index") or 0)

    def _choose(self, groups, mode, budget):
        if mode == "path":
            # A segment is a leaf under its heading chain. Thus U(p) contains
            # that leaf's S_i / c_i; section headings have no segment score.
            ranked = sorted(groups, key=lambda g: (
                g["score"] / max(1, g.get("segment_cost", g["cost"])),
                g.get("rerank_score", float("-inf"))), reverse=True)
            beam = [(0.0, 0, ())]
            width = 3
            for group in ranked:
                if group["score"] <= 0:
                    continue
                expanded = list(beam)
                for value, spent, chosen in beam:
                    if len(chosen) < self.path_limit and spent + group["cost"] <= budget:
                        expanded.append((value + group["score"] /
                                         max(1, group.get("segment_cost", group["cost"])),
                                         spent + group["cost"], chosen + (group,)))
                beam = sorted(expanded, key=lambda state: (
                    state[0], -state[1]), reverse=True)[:width]
            selected = max(beam, key=lambda state: state[0])[2]
            return sorted(selected, key=lambda g: g.get("global_index") or 0)
        ranked = sorted(groups, key=lambda g: (
            g.get("rerank_score", float("-inf")),
            g["score"] / max(1, g.get("segment_cost", g["cost"]))
        ), reverse=True)
        selected, seen = [], set()
        for group in ranked:
            key = (group["path"], group.get("global_index"), group["content"])
            if key not in seen and group["cost"] <= budget:
                seen.add(key)
                selected.append(group)
                budget -= group["cost"]
        selected.sort(key=lambda g: g.get("global_index") or 0)
        return selected

    async def retrieve(self, query, paper, budget=None):
        meta = paper.get("__meta__")
        if not isinstance(meta, dict):
            raise ValueError("Index lacks SF-RAG structure metadata; rebuild from MinerU blocks")
        mode = meta.get("mode", "ordered")
        if mode not in ("path", "section", "ordered"):
            mode = "ordered"
        title = meta.get("title") or next(
            (v["path"].strip("/").split("/")[0] for v in paper.values()
             if isinstance(v, dict) and "path" in v), "Unknown")
        sections = {v["path"]: v for v in paper.values()
                    if isinstance(v, dict) and "path" in v
                    and (v["path"] != "/" + title or v.get("groups"))}
        selected_paths = None
        if mode != "ordered" and sections:
            scores = await self._section_scores(query, sections)
            selected_paths = sorted(scores, key=scores.get, reverse=True)[:self.section_limit]
        else:
            mode = "ordered"
        groups = self._unique_groups(paper, selected_paths, mode)
        if not groups:
            return mode, []
        for group in groups:
            group["paper_title"] = title
        await self._score_segments(query, groups)
        allowance = self.budget if budget is None else budget
        selected = self._choose(groups, mode, allowance)
        while selected and token_count("\n\n".join(
                self.format_context(title, [g]) for g in selected)) > allowance:
            selected.pop()
        return mode, selected

    @staticmethod
    def format_context(paper_title, contexts):
        return "\n".join(
            f"Paper: {paper_title}\nPath: {g['path']}\n"
            f"Summary: {g.get('summary') or 'No summary'}\nContent: {g['content']}"
            for g in contexts
        )

    async def _steps(self, query, paper, budget):
        response = await self._call_llm(decompose_question(query), 500, 0)
        try:
            steps = json.loads(response)
        except ValueError:
            import ast
            try:
                steps = ast.literal_eval(response)
            except (ValueError, SyntaxError):
                steps = [query]
        if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
            steps = [query]
        contexts, entities, stalled, mode = [], [], 0, "ordered"
        for step in (steps or [query])[:self.retrieval_config.get("hops", 3)]:
            if "{previous_entity}" in step:
                step = step.replace("{previous_entity}", ", ".join(entities))
            elif entities:
                step = f"{step}\nPreviously identified entities: {', '.join(entities)}"
            mode, new = await self.retrieve(step, paper, budget)
            for group in new:
                if not any(g["path"] == group["path"] and g["content"] == group["content"]
                           for g in contexts):
                    contexts.append(group)
                    budget -= group["cost"]
            if budget <= 0:
                break
            try:
                prompt = (
                    "Identify only explicit entities that answer this subquery in the evidence. "
                    "Return ONLY a JSON array of objects with fields entity (string) and "
                    "confidence (number in [0,1]); return [] when none are supported.\n"
                    f"Subquery: {step}\nEvidence: {self.format_context('', new)}"
                )
                extracted = json.loads(await self._call_llm(prompt, 200, 0))
            except (ValueError, TypeError):
                extracted = []
            threshold = self.retrieval_config.get("entity_threshold", 0.7)
            additions = [e["entity"] for e in extracted
                         if isinstance(e, dict) and isinstance(e.get("entity"), str)
                         and isinstance(e.get("confidence"), (int, float))
                         and e["confidence"] >= threshold and e["entity"] not in entities]
            entities.extend(additions)
            stalled = 0 if additions else stalled + 1
            if stalled == 2:
                break
        return mode, sorted(contexts, key=lambda g: g.get("global_index") or 0)

    async def answer(self, query, selected_papers, multi_hop=False):
        with open(self.index_path, encoding="utf-8") as stream:
            corpus = json.load(stream)["papers"]
        missing = [title for title in selected_papers if title not in corpus]
        if missing:
            raise ValueError(f"Papers not indexed: {missing}")
        stale = [title for title in selected_papers if
                 corpus[title].get("__meta__", {}).get("generator_model") != self.config.model
                 or corpus[title].get("__meta__", {}).get("index_version") != INDEX_VERSION]
        if stale:
            raise ValueError(f"Index was built with a different generator or algorithm; "
                             f"rebuild these papers before answering: {stale}")
        if len(selected_papers) > 1:
            transformed = await self._call_llm(transform_question(query), 200, 0)
        else:
            transformed = query
        budget = self.budget
        evidence, modes = [], {}
        for title in selected_papers:
            if title not in corpus:
                continue
            paper = corpus[title]
            allowance = budget // max(1, len(selected_papers) - len(modes))
            if multi_hop:
                mode, contexts = await self._steps(transformed, paper, allowance)
            else:
                mode, contexts = await self.retrieve(transformed, paper, allowance)
            modes[title] = mode
            evidence.extend(dict(g, paper=title) for g in contexts)
            budget -= sum(g["cost"] for g in contexts)
        context = "\n\n".join(self.format_context(g["paper"], [g]) for g in evidence)
        while evidence and token_count(generate_answer(query, context)) > self.budget:
            evidence.pop()
            context = "\n\n".join(self.format_context(g["paper"], [g]) for g in evidence)
        answer = await self._call_llm(generate_answer(query, context or "No relevant content"),
                                      1000, 0.3)
        return {"answer": answer, "contexts": evidence, "strategy": modes, "error": None}
