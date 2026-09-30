"""Build a structure-fidelity index from MinerU's ordered content blocks."""

import json
import logging
import re
import statistics
from collections import Counter
from pathlib import Path

import pysbd
import tiktoken
from openai import AsyncOpenAI

from sf_prompts import summarize

log = logging.getLogger(__name__)
NUMBER = re.compile(r"^(\d+(?:\.\d+)*)[.\s]+")
CONVENTIONAL = re.compile(
    r"^(abstract|introduction|background|related work|methods?|methodology|"
    r"experiments?|results?|discussion|conclusion|references|acknowledg(?:e)?ments)$",
    re.I,
)
RELATIONS = ("child", "sibling", "ancestor_sibling", "invalid")
INDEX_VERSION = 2


def token_count(text):
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def block_text(block):
    parts = [block.get("text", "")]
    for field in ("table_caption", "img_caption", "table_body",
                  "table_footnote", "img_footnote"):
        value = block.get(field, "")
        parts.extend(value if isinstance(value, list) else [value])
    return "\n".join(str(part).strip() for part in parts if part).strip()


def normalized_text(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def layout_blocks(folder, blocks):
    """Attach available MinerU coordinates and typography to converted blocks."""
    middle = list(Path(folder).glob("*_middle.json"))
    if len(middle) != 1:
        return blocks
    try:
        pages = json.loads(middle[0].read_text(encoding="utf-8"))["pdf_info"]
    except (OSError, ValueError, KeyError):
        return blocks
    positions = {}
    for page in pages:
        for item in page.get("para_blocks", []):
            spans = [span for line in item.get("lines", [])
                     for span in line.get("spans", []) if span.get("type") == "text"]
            text = " ".join(span.get("content", "") for span in spans)
            if text and item.get("bbox"):
                positions.setdefault((page["page_idx"], normalized_text(text)), []).append(
                    (item, page.get("page_size"), spans))
    enriched = []
    for block in blocks:
        block = dict(block)
        key = (block.get("page_idx"), normalized_text(block.get("text", "")))
        if positions.get(key):
            item, block["page_size"], spans = positions[key].pop(0)
            block["bbox"] = item["bbox"]
            block["layout_title"] = item.get("type") == "title"
            if "font_size" not in block:
                sizes = [span.get("font_size", span.get("size")) for span in spans]
                sizes = [size for size in sizes if isinstance(size, (int, float))]
                if sizes:
                    block["font_size"] = max(sizes)
            if "bold" not in block:
                block["bold"] = bool(item.get("bold") or any(
                    span.get("bold") or "bold" in str(span.get("font_weight", "")).lower()
                    for span in spans))
        enriched.append(block)
    return enriched


def split_blocks(blocks, limit=512):
    """Keep paragraphs/list items together when possible; split oversized ones by sentence."""
    segmenter = pysbd.Segmenter(language="en", clean=False)
    units = []
    for block in blocks:
        text = block.strip()
        if not text:
            continue
        if token_count(text) <= limit:
            units.append(text)
            continue
        for sentence in segmenter.segment(text):
            sentence = sentence.strip()
            if token_count(sentence) <= limit:
                units.append(sentence)
            else:
                tokens = tiktoken.get_encoding("cl100k_base").encode(sentence)
                units.extend(
                    tiktoken.get_encoding("cl100k_base").decode(tokens[i:i + limit])
                    for i in range(0, len(tokens), limit)
                )
    chunks, current = [], []
    for unit in units:
        if current and token_count("\n".join(current + [unit])) > limit:
            chunks.append("\n".join(current))
            current = []
        current.append(unit)
    if current:
        chunks.append("\n".join(current))
    return chunks


class StructureIndexer:
    def __init__(self, config, threshold=0.70, segment_tokens=512):
        self.config = config
        self.threshold = threshold
        self.segment_tokens = segment_tokens
        self.llm = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    @staticmethod
    def _candidate(block):
        text = block.get("text", "").strip()
        return bool(text and (
            block.get("text_level") is not None
            or NUMBER.match(text)
            or CONVENTIONAL.fullmatch(text)
            or block.get("layout_title")
            or block.get("heading_cue")
        ))

    @staticmethod
    def _title_index(blocks):
        first = next((i for i, block in enumerate(blocks)
                      if block.get("type", "text") == "text"
                      and block.get("text", "").strip()), None)
        if first is None:
            return None
        block = blocks[first]
        text = block["text"].strip()
        if (block.get("text_level") == 1 and not NUMBER.match(text)
                and not CONVENTIONAL.fullmatch(text)
                and block.get("page_idx", 0) == 0):
            return first
        return None

    @staticmethod
    def _normalize_candidates(blocks):
        """Promote short, typographically marked lines followed by body prose."""
        sizes = [b["font_size"] for b in blocks if b.get("text_level") is None
                 and not b.get("layout_title")
                 and isinstance(b.get("font_size"), (int, float))]
        body_size = statistics.median(sizes) if sizes else None
        for i, block in enumerate(blocks):
            if StructureIndexer._candidate(block):
                continue
            text = block.get("text", "").strip()
            size = block.get("font_size")
            next_body = next((b for b in blocks[i + 1:i + 4] if block_text(b)), None)
            typography = (isinstance(size, (int, float)) and body_size is not None
                          and size > body_size * 1.15) or (
                              bool(block.get("bold")) and isinstance(size, (int, float))
                              and (body_size is None or size >= body_size))
            standalone = (block.get("type", "text") == "text" and text
                          and len(text) <= 120 and not text.endswith((".", ":", ";", "?"))
                          and next_body is not None
                          and next_body.get("text_level") is None
                          and len(next_body.get("text", "")) > len(text))
            if typography and standalone:
                block["heading_cue"] = True

    @staticmethod
    def _ancestor_depth(candidate, stack):
        number = NUMBER.match(candidate["text"])
        if number:
            return len(number.group(1).split("."))
        level = candidate.get("text_level")
        if level is not None:
            for depth in range(len(stack) - 1, 0, -1):
                if stack[depth - 1].get("text_level") == level:
                    return depth
        if CONVENTIONAL.fullmatch(candidate["text"].strip()):
            return 1
        box = candidate.get("bbox")
        if box:
            for depth in range(len(stack) - 1, 0, -1):
                ancestor = stack[depth - 1]
                if ancestor.get("page_idx") == candidate.get("page_idx") and ancestor.get("bbox"):
                    if abs(box[0] - ancestor["bbox"][0]) <= 15:
                        return depth
        return 1

    @staticmethod
    def _rule(previous, candidate, stack):
        number = NUMBER.match(candidate["text"])
        earlier = NUMBER.match(previous["text"]) if previous else None
        level = candidate.get("text_level")
        old_level = previous.get("text_level") if previous else None
        if (len(candidate["text"]) >= 160 and not (number or candidate.get("layout_title"))
                and not (isinstance(level, int) and isinstance(old_level, int)
                         and level > old_level)):
            return "invalid"
        if previous is None and (number or candidate.get("text_level") is not None
                                 or CONVENTIONAL.fullmatch(candidate["text"])
                                 or candidate.get("layout_title") or candidate.get("heading_cue")):
            return "sibling"
        if number and earlier:
            depth = len(number.group(1).split("."))
            old_depth = len(earlier.group(1).split("."))
            if depth > old_depth:
                return ("child" if depth == old_depth + 1 and
                        number.group(1).startswith(earlier.group(1) + ".") else "invalid")
            return "sibling" if depth == old_depth else "ancestor_sibling"
        if level is not None and old_level is not None:
            if level > old_level:
                return "child"
            if level < old_level:
                return "ancestor_sibling"
        if previous and candidate.get("page_idx") == previous.get("page_idx"):
            current_box, old_box = candidate.get("bbox"), previous.get("bbox")
            if current_box and old_box:
                indent = current_box[0] - old_box[0]
                if indent > 15:
                    return "child"
                if indent < -15:
                    return "ancestor_sibling"
        if CONVENTIONAL.fullmatch(candidate["text"]):
            return "sibling" if len(stack) <= 1 else "ancestor_sibling"
        return "sibling" if ((level is not None and level == old_level)
                             or candidate.get("heading_cue")) else "invalid"

    async def _relation(self, previous, candidate, stack):
        prompt = (
            "Validate the next academic heading relative to the preceding heading and "
            "earlier confirmed ancestors. Return ONLY JSON with probabilities for "
            "child, sibling, ancestor_sibling (sibling of an earlier ancestor), invalid. "
            "All four probabilities must be in [0,1] and sum to 1. When "
            "ancestor_sibling is selected, also return ancestor_depth: the 1-based "
            "depth of the earlier ancestor's sibling (1 is top-level).\n"
            f"Ancestors with depths: {[(i + 1, h['text']) for i, h in enumerate(stack)]}\n"
            f"Previous: {previous['text'] if previous else '(paper title)'}\n"
            f"Next: {candidate['text']}\n"
            f"MinerU levels: {previous.get('text_level') if previous else None}, "
            f"{candidate.get('text_level')}; pages: "
            f"{previous.get('page_idx') if previous else None}, {candidate.get('page_idx')}; "
            f"positions: {previous.get('bbox') if previous else None}, {candidate.get('bbox')}; "
            f"preceding body text: {candidate.get('context_before', '')[:180]}; "
            f"following body text: {candidate.get('context_after', '')[:180]}"
        )
        try:
            response = await self.llm.chat.completions.create(
                model=self.config.model, messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0,
            )
            output = json.loads(response.choices[0].message.content)
            scores = {key: output[key] for key in RELATIONS}
            if (not isinstance(output, dict) or
                    set(output) - set(RELATIONS) - {"ancestor_depth"} or
                    any(not isinstance(scores[k], (int, float)) or not 0 <= scores[k] <= 1
                        for k in RELATIONS) or abs(sum(scores.values()) - 1) > 0.02):
                raise ValueError("Invalid heading relation distribution")
            relation = max(RELATIONS, key=scores.get)
            confidence = scores[relation]
            reported_depth = output.get("ancestor_depth")
            if relation == "ancestor_sibling" and (
                    type(reported_depth) is not int or not 1 <= reported_depth < len(stack)):
                raise ValueError("Invalid ancestor depth")
            if relation == "ancestor_sibling":
                candidate["_ancestor_depth"] = reported_depth
        except Exception as exc:
            log.warning("Heading validation unavailable: %s", exc)
            relation, confidence = "invalid", 0.0
        fallback = confidence < self.threshold
        if fallback:
            relation = self._rule(previous, candidate, stack)
            candidate.pop("_ancestor_depth", None)
        if relation == "ancestor_sibling" and "_ancestor_depth" not in candidate:
            candidate["_ancestor_depth"] = self._ancestor_depth(candidate, stack)
        return relation, confidence, fallback

    @staticmethod
    def _reconcile(blocks, decisions):
        """Rebuild paths from accepted decisions, checking numbering and parent constraints."""
        headings, stack, numbers, labels = {}, [], [], []
        path_counts = Counter()
        for decision in decisions:
            if decision["relation"] == "invalid":
                continue
            index = decision["index"]
            candidate = blocks[index]
            match = NUMBER.match(candidate["text"].strip())
            number = tuple(int(part) for part in match.group(1).split(".")) if match else None
            relation = decision["relation"]
            depth = (len(number) if number else
                     len(stack) + 1 if relation == "child" else
                     decision.get("ancestor_depth", 1) if relation == "ancestor_sibling" else
                     max(1, len(stack)))
            if depth > len(stack) + 1:
                decision["relation"], decision["reconciliation"] = "invalid", "missing parent"
                continue
            if number and depth > 1:
                parent_number = numbers[depth - 2]
                if parent_number != number[:-1]:
                    decision["relation"], decision["reconciliation"] = "invalid", "numbering parent"
                    continue
            if number and depth <= len(numbers):
                earlier = numbers[depth - 1]
                if earlier and number <= earlier:
                    decision["relation"], decision["reconciliation"] = "invalid", "numbering order"
                    continue
            if number and relation == "child" and depth <= len(stack):
                decision["reconciliation"] = "numbering overrides relation"
            if depth > 1 and not number:
                parent = stack[depth - 2]
                typed = (isinstance(parent.get("text_level"), int)
                         and isinstance(candidate.get("text_level"), int)
                         and candidate["text_level"] > parent["text_level"])
                sized = (isinstance(parent.get("font_size"), (int, float))
                         and isinstance(candidate.get("font_size"), (int, float))
                         and candidate.get("heading_cue")
                         and candidate["font_size"] < parent["font_size"])
                positioned = (
                    candidate.get("page_idx") == parent.get("page_idx")
                    and candidate.get("bbox") and parent.get("bbox")
                    and candidate["bbox"][0] - parent["bbox"][0] > 15)
                if not (typed or sized or positioned):
                    depth = 1
                    decision["relation"], decision["reconciliation"] = (
                        "sibling", "unsupported nesting")
            stack = stack[:depth - 1] + [candidate]
            numbers = numbers[:depth - 1] + [number]
            parent_labels = labels[:depth - 1]
            name = candidate["text"].strip().replace("/", "／")
            key = tuple(parent_labels + [name])
            path_counts[key] += 1
            labels = parent_labels + [name if path_counts[key] == 1
                                      else f"{name} [{path_counts[key]}]"]
            headings[index] = labels
            decision["depth"] = depth
        levels = {len(path) for path in headings.values()}
        mode = "path" if any(depth > 1 for depth in levels) else (
            "section" if headings else "ordered")
        return headings, mode

    async def build(self, folder):
        files = list(Path(folder).glob("*_content_list.json"))
        if len(files) != 1:
            raise ValueError(f"Expected one MinerU content list in {folder}")
        blocks = layout_blocks(folder, json.loads(files[0].read_text(encoding="utf-8")))
        self._normalize_candidates(blocks)
        title_index = self._title_index(blocks)
        title = blocks[title_index]["text"].strip() if title_index is not None else Path(folder).name
        heading_candidates = [i for i, b in enumerate(blocks)
                              if i != title_index and self._candidate(b)]
        # Recurrence alone is not enough: real headings may have the same name.
        frequency = Counter((normalized_text(b.get("text", "")), b.get("page_idx"))
                            for b in blocks if b.get("text", "").strip())
        repeated_pages = Counter({text: len({page for name, page in frequency if name == text})
                                  for text, _ in frequency})
        furniture = {
            i for i, block in enumerate(blocks) if (
                i != title_index and block.get("text", "").strip()
                and repeated_pages[normalized_text(blocks[i]["text"])] >= 3
                and blocks[i].get("bbox") and blocks[i].get("page_size")
                and (blocks[i]["bbox"][1] < 0.12 * blocks[i]["page_size"][1]
                     or blocks[i]["bbox"][3] > 0.88 * blocks[i]["page_size"][1])
                and not CONVENTIONAL.fullmatch(blocks[i]["text"].strip()))
        }
        heading_candidates = [i for i in heading_candidates if i not in furniture]
        stack, decisions = [], []
        previous = None
        for index in heading_candidates:
            candidate = dict(blocks[index])
            candidate["context_before"] = next(
                (block_text(b) for b in reversed(blocks[max(0, index - 3):index])
                 if b.get("text_level") is None and block_text(b)), "")
            candidate["context_after"] = next(
                (block_text(b) for b in blocks[index + 1:index + 4]
                 if b.get("text_level") is None and block_text(b)), "")
            relation, confidence, fallback = await self._relation(previous, candidate, stack)
            if relation == "invalid":
                decisions.append({"index": index, "relation": relation, "confidence": confidence,
                                  "fallback": fallback})
                continue
            if relation == "child":
                depth = min(len(stack) + 1, 6)
            elif relation == "sibling":
                depth = max(1, len(stack))
            else:
                depth = candidate.get("_ancestor_depth",
                                      self._ancestor_depth(candidate, stack))
            stack = stack[:depth - 1] + [candidate]
            previous = candidate
            decisions.append({"index": index, "relation": relation, "confidence": confidence,
                              "fallback": fallback, "depth": depth,
                              "ancestor_depth": depth if relation == "ancestor_sibling" else None})

        headings, mode = self._reconcile(blocks, decisions)
        sections = {}
        current_path = title
        content = {title: []}
        runs = []
        # Preserve every converted body block, including uncertain heading candidates.
        for i, block in enumerate(blocks):
            text = block_text(block)
            if not text or i == title_index or i in furniture:
                continue
            if i in headings:
                current_path = title + "/" + "/".join(headings[i])
                sections[current_path] = block["text"].strip()
                content.setdefault(current_path, [])
            else:
                content.setdefault(current_path, []).append(text)
                if not runs or runs[-1][0] != current_path:
                    runs.append((current_path, []))
                runs[-1][1].append(text)
        if mode == "ordered":
            content = {title: [block_text(b) for i, b in enumerate(blocks)
                               if i not in furniture and block_text(b)]}
            runs = [(title, content[title])]

        result = {"__meta__": {"mode": mode, "title": title, "source_folder": Path(folder).name,
                               "decisions": decisions,
                               "index_version": INDEX_VERSION, "generator_model": self.config.model,
                               "segment_tokens": self.segment_tokens}}
        for path in content:
            result[path] = {"path": "/" + path, "title": sections.get(path, title),
                            "groups": [], "content": []}
        ordinal = 0
        previous_summaries = {}
        for path, paragraphs in runs:
            for chunk in split_blocks(paragraphs, self.segment_tokens):
                prompt = summarize(title, " > ".join(path.split("/")),
                                   chunk, previous_summaries.get(path, "<SECTION_START>"))
                response = await self.llm.chat.completions.create(
                    model=self.config.model, messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                summary = response.choices[0].message.content.strip()
                result[path]["groups"].append({
                    "content": chunk, "summary": summary, "global_index": ordinal,
                    "token_cost": token_count(chunk), "special_type": "content"})
                result[path]["content"].append(chunk)
                previous_summaries[path] = summary
                ordinal += 1
        return result
