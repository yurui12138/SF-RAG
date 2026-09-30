"""Offline regression checks for structure routing and budget selection."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sf_rag import SFRAG
from sf_retriever import SFRetriever
from structure_index import INDEX_VERSION, StructureIndexer, split_blocks, token_count
from sf_prompts import generate_answer


CONFIG = SimpleNamespace(api_key="test", base_url="https://example.invalid",
                         model="gpt-4o-mini", rerank_model="BAAI/bge-reranker-v2-m3",
                         embedding_model="text-embedding-3-small")


class StructureTests(unittest.IsolatedAsyncioTestCase):
    async def _build(self, blocks):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "paper_content_list.json").write_text(json.dumps(blocks), encoding="utf-8")
            indexer = StructureIndexer(CONFIG)
            indexer._relation = AsyncMock(side_effect=lambda previous, candidate, stack:
                                          (indexer._rule(previous, candidate, stack), 0.6, True))
            indexer.llm.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Summary"))]))
            return await indexer.build(folder)

    async def test_three_modes_and_fallback(self):
        title = {"type": "text", "text": "Paper", "text_level": 1}
        body = {"type": "text", "text": "Evidence from the body."}
        cases = [
            ([title, {"type": "text", "text": "1 Introduction", "text_level": 1},
              body, {"type": "text", "text": "1.1 Background", "text_level": 2}, body], "path"),
            ([title, {"type": "text", "text": "1 Introduction", "text_level": 1}, body],
             "section"),
            ([title, body, body], "ordered"),
        ]
        for blocks, expected in cases:
            with self.subTest(expected=expected):
                result = await self._build(blocks)
                self.assertEqual(result["__meta__"]["mode"], expected)
                self.assertTrue(any(g["content"] for section in result.values()
                                    if "groups" in section for g in section["groups"]))
                if expected != "ordered":
                    self.assertTrue(result["__meta__"]["decisions"][0]["fallback"])

    def test_segment_bound_and_paragraphs(self):
        paragraphs = ["A paragraph. " * 60, "Separate paragraph."]
        chunks = split_blocks(paragraphs, 100)
        self.assertTrue(all(token_count(c) <= 100 for c in chunks))
        self.assertIn("Separate paragraph.", chunks[-1])

    async def test_global_reconciliation_rejects_missing_numbered_parent(self):
        blocks = [{"text": "Paper", "text_level": 1},
                  {"text": "1 Introduction", "text_level": 1},
                  {"text": "2.1 Orphan", "text_level": 1},
                  {"text": "2 Results", "text_level": 1},
                  {"text": "Evidence."}]
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "paper_content_list.json").write_text(json.dumps(blocks), encoding="utf-8")
            indexer = StructureIndexer(CONFIG)
            indexer._relation = AsyncMock(return_value=("child", 0.99, False))
            indexer.llm.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Summary"))]))
            paper = await indexer.build(folder)
        self.assertEqual(paper["__meta__"]["mode"], "section")
        self.assertEqual(paper["__meta__"]["decisions"][1]["reconciliation"], "numbering parent")
        self.assertIn("2.1 Orphan", " ".join(paper["Paper/1 Introduction"]["content"]))

    async def test_page_furniture_needs_repetition_and_margin(self):
        blocks = [{"text": "Paper", "text_level": 1, "page_idx": 0},
                  {"text": "1 Introduction", "text_level": 1, "page_idx": 0}]
        pages = [{"page_idx": 0, "para_blocks": []}]
        for page in (1, 2, 3):
            blocks.extend([{"text": "Running Head", "text_level": 1, "page_idx": page},
                           {"text": "Body text", "page_idx": page}])
            pages.append({"page_idx": page, "page_size": [100, 100], "para_blocks": [
                {"type": "title", "bbox": [5, 2, 80, 10], "lines": [
                    {"spans": [{"type": "text", "content": "Running Head"}]}]}]})
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "paper_content_list.json").write_text(json.dumps(blocks), encoding="utf-8")
            Path(folder, "paper_middle.json").write_text(json.dumps({"pdf_info": pages}), encoding="utf-8")
            indexer = StructureIndexer(CONFIG)
            indexer._relation = AsyncMock(side_effect=lambda previous, candidate, stack:
                                          (indexer._rule(previous, candidate, stack), 0.6, True))
            indexer.llm.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Summary"))]))
            paper = await indexer.build(folder)
        self.assertEqual(paper["__meta__"]["mode"], "section")
        self.assertEqual(len(paper["__meta__"]["decisions"]), 1)
        self.assertNotIn("Running Head", " ".join(paper["Paper/1 Introduction"]["content"]))
        self.assertIn("Body text", " ".join(paper["Paper/1 Introduction"]["content"]))

    async def test_missing_title_does_not_consume_first_section(self):
        paper = await self._build([
            {"text": "An unmarked paper title"},
            {"text": "1 Introduction", "text_level": 1},
            {"text": "Introduction body."},
            {"text": "1.1 Background", "text_level": 2},
            {"text": "Background body."},
        ])
        self.assertEqual(paper["__meta__"]["mode"], "path")
        self.assertIn(f"{paper['__meta__']['title']}/1 Introduction", paper)

    async def test_typography_and_nearby_prose_promote_missing_heading(self):
        paper = await self._build([
            {"text": "Paper", "text_level": 1},
            {"text": "Experimental Protocol", "font_size": 18, "bold": True},
            {"text": "We evaluate the approach across many different datasets.",
             "font_size": 11},
        ])
        self.assertEqual(paper["__meta__"]["mode"], "section")
        self.assertIn("Paper/Experimental Protocol", paper)

    async def test_unmarked_abstract_precedes_headings_in_index(self):
        paper = await self._build([
            {"text": "Paper", "text_level": 1},
            {"text": "Abstract evidence, otherwise inaccessible."},
            {"text": "1 Methods", "text_level": 1},
            {"text": "Methods body."},
        ])
        self.assertEqual(paper["Paper"]["groups"][0]["global_index"], 0)
        self.assertEqual(paper["Paper/1 Methods"]["groups"][0]["global_index"], 1)

    def test_fallback_uses_layout_indent(self):
        previous = {"text": "Overview", "text_level": 1, "page_idx": 0, "bbox": [10, 20, 60, 30]}
        candidate = {"text": "Details", "text_level": 1, "page_idx": 0, "bbox": [30, 40, 70, 50]}
        self.assertEqual(StructureIndexer._rule(previous, candidate, [previous]), "child")

    async def test_unsupported_nested_heading_uses_section_mode(self):
        blocks = [{"text": "Paper", "text_level": 1},
                  {"text": "Overview", "text_level": 1},
                  {"text": "Details", "text_level": 1},
                  {"text": "Evidence."}]
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "paper_content_list.json").write_text(json.dumps(blocks), encoding="utf-8")
            indexer = StructureIndexer(CONFIG)
            indexer._relation = AsyncMock(return_value=("child", 0.95, False))
            indexer.llm.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Summary"))]))
            paper = await indexer.build(folder)
        self.assertEqual(paper["__meta__"]["mode"], "section")
        self.assertEqual(paper["__meta__"]["decisions"][1]["reconciliation"],
                         "unsupported nesting")
        self.assertIn("Paper/Details", paper)

    async def test_unnumbered_heading_returns_to_earlier_ancestor(self):
        blocks = [{"text": "Paper", "text_level": 1}]
        blocks.extend({"text": name, "text_level": level}
                      for name, level in (("Main", 1), ("Part", 2), ("Detail", 3),
                                          ("Fine detail", 4), ("Next main", 1)))
        paper = await self._build(blocks)
        self.assertEqual(paper["__meta__"]["mode"], "path")
        self.assertIn("Paper/Next main", paper)
        self.assertEqual(paper["__meta__"]["decisions"][-1]["ancestor_depth"], 1)

    async def test_llm_ancestor_depth_is_retained(self):
        indexer = StructureIndexer(CONFIG)
        indexer.llm.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "child": 0.02, "sibling": 0.02, "ancestor_sibling": 0.94,
                "invalid": 0.02, "ancestor_depth": 1})))]))
        stack = [{"text": "Main"}, {"text": "Part"}, {"text": "Detail"}]
        candidate = {"text": "Next main"}
        relation, confidence, fallback = await indexer._relation(stack[-1], candidate, stack)
        self.assertEqual((relation, confidence, fallback), ("ancestor_sibling", 0.94, False))
        self.assertEqual(candidate["_ancestor_depth"], 1)

    def test_long_heading_reaches_validation(self):
        heading = {"text": "1.1 " + "Detailed methods " * 12, "text_level": 2}
        self.assertTrue(StructureIndexer._candidate(heading))
        parent = {"text": "Methods", "text_level": 1}
        self.assertEqual(StructureIndexer._rule(parent, heading, [parent]), "child")

    async def test_repeated_real_headings_keep_separate_paths_and_order(self):
        blocks = [{"text": "Paper", "text_level": 1}]
        for i in range(3):
            blocks.extend([{"text": "Study", "text_level": 1, "page_idx": i},
                           {"text": f"Evidence {i}.", "page_idx": i}])
        paper = await self._build(blocks)
        self.assertEqual(paper["__meta__"]["mode"], "section")
        paths = ["Paper/Study", "Paper/Study [2]", "Paper/Study [3]"]
        self.assertEqual([paper[path]["title"] for path in paths], ["Study"] * 3)
        self.assertEqual([paper[path]["groups"][0]["global_index"] for path in paths],
                         [0, 1, 2])


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.retriever = SFRetriever(CONFIG, {"alpha": 0.5, "beta": 0.8, "sections": 2,
                                            "paths": 1, "token_budget": 90})
        self.retriever._section_scores = AsyncMock(return_value={
            "/Paper/A": 0.8, "/Paper/B": 0.7})

        async def score(query, groups):
            for group in groups:
                group["score"] = 0.8 if group["path"].endswith("A") else 0.1
                group["cost"] = 40
            return groups
        self.retriever._score_segments = AsyncMock(side_effect=score)

    async def test_paths_budget_and_document_order(self):
        paper = {"__meta__": {"mode": "path", "title": "Paper"}}
        for path, start in (("A", 0), ("B", 2)):
            paper[path] = {"path": "/Paper/" + path, "title": path,
                           "content": [path],
                           "groups": [{"content": path + str(i), "summary": path,
                                       "global_index": start + i} for i in range(2)]}
        mode, selected = await self.retriever.retrieve("question", paper)
        self.assertEqual(mode, "path")
        self.assertEqual([g["global_index"] for g in selected], [0])
        self.assertLessEqual(sum(g["cost"] for g in selected), 90)

    async def test_ordered_mode_skips_section_selection(self):
        paper = {"__meta__": {"mode": "ordered", "title": "Paper"},
                 "Body": {"path": "/Paper", "groups": [
                     {"content": "evidence", "global_index": 0}]}}
        mode, selected = await self.retriever.retrieve("question", paper)
        self.assertEqual(mode, "ordered")
        self.assertEqual(len(selected), 1)
        self.retriever._section_scores.assert_not_awaited()

    async def test_root_body_is_a_selectable_section(self):
        paper = {"__meta__": {"mode": "section", "title": "Paper"},
                 "root": {"path": "/Paper", "title": "Paper",
                          "groups": [{"content": "Unheaded abstract evidence.",
                                      "summary": "Abstract", "global_index": 0}]},
                 "section": {"path": "/Paper/Methods", "title": "Methods",
                             "groups": [{"content": "Methods body.",
                                         "summary": "Methods", "global_index": 1}]}}
        self.retriever._section_scores = AsyncMock(
            return_value={"/Paper": 0.9, "/Paper/Methods": 0.1})
        mode, selected = await self.retriever.retrieve("abstract evidence?", paper)
        self.assertEqual(mode, "section")
        self.assertIn("Unheaded abstract evidence.", [g["content"] for g in selected])
        self.retriever._section_scores.assert_awaited_once()

    async def test_rejects_indexes_without_sf_metadata(self):
        with self.assertRaisesRegex(ValueError, "rebuild"):
            await self.retriever.retrieve("question", {"Old": {"path": "/Paper/Old"}})

    async def test_single_level_uses_section_scoring_without_paths(self):
        paper = {"__meta__": {"mode": "section", "title": "Paper"},
                 "A": {"path": "/Paper/A", "title": "A", "content": ["evidence"],
                       "groups": [{"content": "evidence", "summary": "summary",
                                   "global_index": 0}]}}
        mode, selected = await self.retriever.retrieve("question", paper)
        self.assertEqual(mode, "section")
        self.assertEqual(len(selected), 1)
        self.retriever._section_scores.assert_awaited_once()

    def test_path_selection_limits_segment_leaves(self):
        groups = [{"path": "/Paper/A", "content": str(i), "score": score,
                   "cost": 30, "global_index": i} for i, score in enumerate([0.9, 0.1, 0.8])]
        selected = self.retriever._choose(groups, "path", 90)
        self.assertEqual([g["global_index"] for g in selected], [0])

    def test_child_leaf_does_not_require_unrelated_parent_segment(self):
        groups = [
            {"path": "/Paper/A", "content": "parent", "score": 0.8,
             "cost": 10, "global_index": 0},
            {"path": "/Paper/A/B", "content": "child", "score": 0.9,
             "cost": 10, "global_index": 1},
        ]
        selected = self.retriever._choose(groups, "path", 20)
        self.assertEqual([g["content"] for g in selected], ["child"])

    def test_shared_ancestor_is_selected_only_once(self):
        self.retriever.path_limit = 2
        groups = [
            {"path": "/Paper/A", "content": "shared", "score": 1.0,
             "cost": 10, "global_index": 0},
            {"path": "/Paper/A/B", "content": "left", "score": 0.9,
             "cost": 10, "global_index": 1},
            {"path": "/Paper/A/C", "content": "right", "score": 0.9,
             "cost": 10, "global_index": 2},
        ]
        selected = self.retriever._choose(groups, "path", 30)
        self.assertEqual([g["content"] for g in selected], ["shared", "left"])

    def test_path_density_uses_fused_score_over_segment_cost(self):
        groups = [
            {"path": "/Paper/A", "content": str(i), "score": score,
             "cost": 10, "global_index": i} for i, score in enumerate([0.1, 1.0])
        ] + [
            {"path": "/Paper/B", "content": str(i), "score": 0.45,
             "cost": 10, "global_index": i + 2} for i in range(4)
        ]
        selected = self.retriever._choose(groups, "path", 20)
        self.assertEqual([g["path"] for g in selected], ["/Paper/A"])

    def test_reranker_does_not_redefine_path_score(self):
        groups = [
            {"path": "/Paper/A", "content": "a", "score": 0.4,
             "rerank_score": 1.0, "cost": 10, "global_index": 0},
            {"path": "/Paper/B", "content": "b", "score": 0.5,
             "rerank_score": 0.0, "cost": 10, "global_index": 1},
        ]
        self.assertEqual(self.retriever._choose(groups, "path", 10)[0]["path"], "/Paper/B")

    def test_each_segment_can_end_a_root_to_leaf_path(self):
        groups = [{"path": "/Paper/A", "content": str(i), "score": score,
                   "cost": 10, "global_index": i}
                  for i, score in enumerate([0.8, -0.7, 0.1])]
        self.assertEqual([g["global_index"] for g in
                          self.retriever._choose(groups, "path", 30)], [0])

    def test_later_segment_can_be_retrieved_without_earlier_prefix(self):
        groups = [{"path": "/Paper/A", "content": "unrelated",
                   "score": 0.01, "cost": 50, "global_index": 0},
                  {"path": "/Paper/A", "content": "late evidence",
                   "score": 0.9, "cost": 10, "global_index": 1}]
        self.assertEqual([g["content"] for g in
                          self.retriever._choose(groups, "path", 10)], ["late evidence"])

    async def test_final_context_is_within_evidence_budget(self):
        paper = {"__meta__": {"mode": "ordered", "title": "Very Long Paper Title"},
                 "Body": {"path": "/Very Long Paper Title", "groups": [
                     {"content": "Long evidence paragraph with several words.",
                      "summary": "A summary.", "global_index": 0}]}}
        async def optimistic_score(query, groups):
            for group in groups:
                group.update(score=0.9, cost=1)
        self.retriever._score_segments = AsyncMock(side_effect=optimistic_score)
        _, selected = await self.retriever.retrieve("question", paper, budget=10)
        self.assertEqual(selected, [])

    async def test_answer_checks_full_prompt_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            index_path = Path(folder, "index.json")
            index_path.write_text(json.dumps({"papers": {
                "Paper": {"__meta__": {"mode": "ordered", "title": "Paper",
                                       "generator_model": CONFIG.model,
                                       "index_version": INDEX_VERSION}}
            }}), encoding="utf-8")
            self.retriever.index_path = str(index_path)
            self.retriever.budget = 100
            self.retriever.retrieve = AsyncMock(return_value=("ordered", [
                {"path": "/Paper", "content": "evidence " * 60, "summary": "summary",
                 "cost": 1, "global_index": 0}]))
            self.retriever._call_llm = AsyncMock(return_value="Answer")
            result = await self.retriever.answer("What happened?", ["Paper"])
        context = "\n\n".join(self.retriever.format_context(g["paper"], [g])
                               for g in result["contexts"])
        self.assertLessEqual(token_count(generate_answer("What happened?", context)), 100)

    async def test_answer_rejects_stale_generator_index(self):
        with tempfile.TemporaryDirectory() as folder:
            index_path = Path(folder, "index.json")
            index_path.write_text(json.dumps({"papers": {
                "Paper": {"__meta__": {"generator_model": "another-model",
                                       "index_version": INDEX_VERSION}}
            }}), encoding="utf-8")
            self.retriever.index_path = str(index_path)
            with self.assertRaisesRegex(ValueError, "rebuild"):
                await self.retriever.answer("Question?", ["Paper"])

    def test_raw_segment_cost_drives_density_not_prompt_cost(self):
        groups = [
            {"path": "/Paper/A", "content": "A", "score": 0.8,
             "segment_cost": 20, "cost": 21, "global_index": 0},
            {"path": "/Paper/B", "content": "B", "score": 0.6,
             "segment_cost": 5, "cost": 40, "global_index": 1},
        ]
        self.assertEqual(self.retriever._choose(groups, "path", 40)[0]["path"], "/Paper/B")

    async def test_dual_channel_beta_fusion(self):
        group = {"path": "/Paper/A", "content": "raw", "summary": "overview"}
        self.retriever._embed = AsyncMock(return_value=[
            [1, 0], [1, 0], [0, 1]])
        self.retriever.rerank_contents = AsyncMock(return_value=[
            {"index": 0, "relevance_score": 0.9}])
        await self.retriever._score_segments("query", [group])
        self.assertAlmostEqual(group["score"], 0.8)

    async def test_section_scoring_includes_late_content(self):
        sections = {"/Paper/A": {"title": "A", "groups": [
            {"content": "early " * 7000 + "late evidence " * 500,
             "summary": "discourse summary"}]}}
        self.retriever._call_llm = AsyncMock(return_value='{}')

        async def embed(texts):
            self.assertGreaterEqual(len(texts), 3)
            self.assertTrue(any("late evidence" in text for text in texts[2:]))
            self.assertTrue(all(token_count(text) <= 6000 for text in texts[1:]))
            self.assertTrue(all("discourse summary" not in text for text in texts[1:]))
            return [[1, 0]] + [[0, 1]] + [[1, 0]] * (len(texts) - 2)

        self.retriever._embed = AsyncMock(side_effect=embed)
        scores = await self.retriever._section_scores("late evidence", sections)
        self.assertGreater(scores["/Paper/A"], 0)

    async def test_short_section_embeds_body_once_without_summary(self):
        sections = {"/Paper/A": {"title": "A", "groups": [
            {"content": "Only body text.", "summary": "Only summary text."}]}}
        self.retriever._call_llm = AsyncMock(return_value='{}')
        self.retriever._embed = AsyncMock(return_value=[[1, 0], [1, 0]])
        await SFRetriever._section_scores(self.retriever, "body?", sections)
        self.retriever._embed.assert_awaited_once_with(["body?", "Only body text."])

    async def test_multihop_passes_entities_without_placeholder(self):
        self.retriever._call_llm = AsyncMock(side_effect=[
            '["Find the method", "Where was it evaluated?"]',
            '[{"entity": "Method X", "confidence": 0.95}]',
            "[]",
        ])
        self.retriever.retrieve = AsyncMock(side_effect=[
            ("section", [{"path": "/Paper/A", "content": "Method X", "cost": 10,
                          "global_index": 0}]),
            ("section", []),
        ])
        await self.retriever._steps("question", {}, 80)
        second_query = self.retriever.retrieve.await_args_list[1].args[0]
        self.assertIn("Method X", second_query)


class InterfaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_index_and_default_single_query(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "files")
            source.mkdir()
            (source / "paper").mkdir()
            rag = SFRAG(CONFIG, index_path=str(Path(directory, "sf_index.json")))
            rag.indexer.build = AsyncMock(return_value={
                "__meta__": {"title": "Paper", "source_folder": "paper", "mode": "ordered",
                             "generator_model": CONFIG.model, "index_version": INDEX_VERSION,
                             "segment_tokens": 512}})
            self.assertEqual(await rag.build_index(source), ["Paper"])
            self.assertEqual(rag.list_papers(), ["Paper"])
            self.assertEqual(await rag.build_index(source), ["Paper"])
            rag.indexer.build.assert_awaited_once()
            rag.retriever.answer = AsyncMock(return_value={"answer": "Evidence"})
            await rag.answer("Question?", ["Paper"])
            rag.retriever.answer.assert_awaited_once_with(
                "Question?", ["Paper"], multi_hop=False)

    async def test_switch_generator_rebuilds_index(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "files")
            (source / "paper").mkdir(parents=True)
            index_path = Path(directory, "sf_index.json")
            index_path.write_text(json.dumps({"papers": {"Old title": {
                "__meta__": {"source_folder": "paper", "generator_model": "old-model",
                             "index_version": INDEX_VERSION, "segment_tokens": 512}}}}),
                                  encoding="utf-8")
            rag = SFRAG(CONFIG, index_path=str(index_path))
            rag.indexer.build = AsyncMock(return_value={
                "__meta__": {"title": "New title", "source_folder": "paper"}})
            self.assertEqual(await rag.build_index(source), ["New title"])
            rag.indexer.build.assert_awaited_once()
            self.assertEqual(json.loads(index_path.read_text())["papers"]["New title"]
                             ["__meta__"]["generator_model"], CONFIG.model)


if __name__ == "__main__":
    unittest.main()
