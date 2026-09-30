# SF-RAG

`sf_rag.py` exposes the structure-fidelity indexing and retrieval interface.
Input folders must contain MinerU's ordered `*_content_list.json` output. Keep
the original PDFs and layout artifacts when reproducing PDF conversion; the
indexer reads the typed content list directly and uses `*_middle.json` layout
information when available. Repeatable page furniture is removed only when
it is confirmed across at least three page margins. Uncertain heading candidates
remain in the converted body.

```powershell
python demo_sf_rag.py "What is the research motivation?" --paper "Paper title" --build
```

Repeat `--paper` to synthesize evidence across documents. Add `--multi-hop`
only for the optional decomposition extension; it is disabled by default.
The `--build` flag creates `output/sf_index.json`; omit it after indexing.

The index records heading decisions and chooses `path`, `section`, or `ordered`
retrieval based on the recovered outline. Retrieval fuses section role and
body similarity (`alpha=0.5`), then segment and summary similarities
(`beta=0.8`), with cross-encoder reranking and a global context token budget.
Segments are bounded to 512 tokens; selection retains at most two sections
and three root-to-segment paths. Each segment is a leaf under its recovered
section hierarchy; other segments in that section are separate leaves, not
mandatory ancestors. The path density is the fused segment score divided by
its raw segment token cost. A bounded beam considers paths in decreasing
density and retains at most three high-density paths under the context budget
(which counts path, summary, and text). Selected leaves are
presented in document order; continuity is favored by section selection but
not enforced when later segments have stronger evidence.
The cross-encoder refines ordering but does not replace the stated fused score.
An unheaded front-matter segment remains eligible for section selection.

For section selection, the dense signal embeds the section body, not the
structure-anchored summaries; summaries contribute to segment scoring and
answer context. This follows the explicit section-similarity equation in the
method: the separate statement that summaries also participate in section
selection requires clarification in the manuscript. Sections beyond the
embedding input limit are split into 6000-token parts and their vectors are
token-weighted, an approximation to a full-body embedding.
Answer generation checks the complete prompt against the configured token
budget. Index metadata records the generator and index version; running `--build`
rebuilds stale indexes when either changes. Answers reject stale indexes until
they are rebuilt. Existing indexes created before this change need `--build`.

Set `GPT_4o_mini.api_key`, `.base_url`, `.model`, `.rerank_url`,
`.rerank_api_key`, and `.rerank_model` in `.env`. The optional
`GPT_4o_mini.embedding_model` defaults to `text-embedding-3-small`.
Run offline checks with `python -m unittest test_sf`.
