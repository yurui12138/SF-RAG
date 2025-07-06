import json
import re
import asyncio
import aiofiles
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import pysbd
from collections import defaultdict
import logging
from openai import AsyncOpenAI
from config import Config
from promote import generate_images_summary_promote, generate_summary_promote_with_previous_summary, \
    generate_summary_promote, get_true_title

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/retrieval.log'),
        logging.StreamHandler()
    ]
)
logging.getLogger().setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

class PaperTreeBuilder:
    MAX_TITLE_DISPLAY_LENGTH = 10
    MAX_CHILDREN_FOR_SPECIAL_NODES = 3
    CHUNK_SIZE = 5

    def __init__(self, config: Config):
        self.config = config
        self.segmenter = pysbd.Segmenter(language="en", clean=True)
        self.llm = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self.cache_dir = Path("cache/image_summaries")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = 10

    async def _generate_images_summary(self, img_title: str, img_path: str, max_retries: int = 3) -> str:
        cache_key = hashlib.md5(f"{img_title}:{img_path}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.txt"
        if cache_file.exists():
            async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                return (await f.read()).strip()
        for attempt in range(max_retries):
            try:
                response = await self.llm.responses.create(
                    model=self.config.model_vl,
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": generate_images_summary_promote(img_title)},
                            {"type": "input_image", "image_url": img_path}
                        ]
                    }],
                    timeout=30
                )
                summary = response.output_text.strip()
                async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
                    await f.write(summary)
                return summary
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed for image summary (title: {img_title}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    return f"图表：{img_title}（摘要生成失败）"

    def segment_and_merge_text(self, text: str) -> List[str]:
        sentences = self.segmenter.segment(text)
        merged_sentences = []
        buffer = []
        for sentence in sentences:
            cleaned = sentence.strip()
            buffer.append(cleaned)
            if cleaned.endswith('.'):
                merged_sentences.append(' '.join(buffer))
                buffer = []
        if buffer:
            merged_sentences.append(' '.join(buffer))
        return merged_sentences

    async def _generate_summary(self, content: str, parent_titles: List[str], previous_summary: str = None) -> str:
        parent_titles_str = " > ".join(parent_titles)
        if previous_summary:
            prompt = generate_summary_promote_with_previous_summary(parent_titles, parent_titles_str, previous_summary, content)
        else:
            prompt = generate_summary_promote(parent_titles, parent_titles_str, content)
        response = await self.llm.chat.completions.create(
            model=self.config.model,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    async def _load_figures_from_json(self, json_file_path: str, parent_path: str) -> List[Dict]:
        async with aiofiles.open(json_file_path, 'r', encoding='utf-8') as json_file:
            json_data = json.loads(await json_file.read())
        figures = []
        summary_tasks = []
        for item in json_data:
            img_path = item.get("img_path", "")
            img_title = " ".join(item.get("img_title", []))
            summary_tasks.append(self._generate_images_summary(img_title, img_path))
        for i in range(0, len(summary_tasks), self.batch_size):
            batch_tasks = summary_tasks[i:i + self.batch_size]
            summaries = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for idx, (item, summary) in enumerate(zip(json_data[i:i + self.batch_size], summaries)):
                img_path = item.get("img_path", "")
                img_title = " ".join(item.get("img_title", []))
                if isinstance(summary, Exception):
                    summary = f"Figure: {img_title} (summary generation failed)"
                figures.append({
                    "content": f"{img_title}",
                    "summary": summary,
                    "path": parent_path,
                    "url": img_path,
                    "chapter_depth": len(parent_path.split('/')) + 1 if parent_path else 1,
                    "special_type": "figure"
                })
        return figures

    async def _process_content(self, sentences: List[str], node: Dict, parent_titles: List[str], group_counter: int) -> Tuple[List[Dict], int]:
        groups = []
        if not sentences:
            return groups, group_counter
        chunks = [sentences[i:i + self.CHUNK_SIZE] for i in range(0, len(sentences), self.CHUNK_SIZE)]
        previous_summary = None
        for chunk in chunks:
            content = ' '.join(chunk)
            summary = await self._generate_summary(content, parent_titles, previous_summary)
            previous_summary = summary
            group_data = {
                "content": content,
                "summary": summary,
                "global_index": group_counter,
                "path": node["path"],
                "chapter_depth": len(node["path"].split('/')) - 1,
            }
            groups.append(group_data)
            group_counter += 1
        return groups, group_counter

    async def update_headings_in_md(self, lines: List[str]) -> List[str]:
        logger.critical("Section Hierarchy Extraction.")
        heading_lines = [(i, line.strip()) for i, line in enumerate(lines) if line.lstrip().startswith('#')]
        if not heading_lines:
            logger.info("No headings found in the document.")
            return lines
        logger.critical("Section Hierarchy Successfully Extracted.")

        headings = [line for _, line in heading_lines]

        logger.critical("Filter and Rearrange Headings.")
        try:
            response = await self.llm.chat.completions.create(
                model=self.config.model,
                messages=[
                    {'role': 'system',
                     'content': 'You are a helpful assistant. Return updated headings in the same order as provided.'},
                    {'role': 'user', 'content': get_true_title(headings)}
                ]
            )
            response_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._apply_fallback_headings(lines, heading_lines, logger)

        updated_headings = [line.strip() for line in response_text.split('\n') if line.strip()]

        if len(updated_headings) != len(headings):
            logger.warning(
                f"LLM returned {len(updated_headings)} headings, expected {len(headings)}. Applying fallback.")
            return self._apply_fallback_headings(lines, heading_lines, logger)

        for (idx, _), new_heading in zip(heading_lines, updated_headings):
            lines[idx] = new_heading + '\n'

        logger.critical("Headings updated successfully.")
        return lines

    def _apply_fallback_headings(self, lines: List[str], heading_lines: List[tuple], logger: logging.Logger) -> List[
        str]:
        for idx, (line_idx, original_heading) in enumerate(heading_lines):
            if idx == 0:
                lines[line_idx] = original_heading + '\n'
            else:
                lines[line_idx] = '##' + original_heading.lstrip('#') + '\n'
        logger.critical("Fallback headings applied: first as title, others as level-2.")
        return lines

    async def parse_md_to_structure(self, md_content: str, json_file_path: str) -> Dict:
        structure = {"groups": []}
        current_path = []
        current_node = {"path": ""}
        current_content = []
        in_references = False
        group_counter = 0
        tasks = []
        lines = await self.update_headings_in_md(md_content.split('\n'))
        title = next((item for item in lines if item.startswith("# ")), None).split('#')[-1].strip()
        figures_task = asyncio.create_task(self._load_figures_from_json(json_file_path, f"/{title}/All figures and tables"))
        logger.critical("Contextual Summary Generation.")
        for line in lines:
            header_match = re.match(r'^(#+)\s*(.*)', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                if current_content:
                    if in_references:
                        for ref_line in '\n'.join(current_content).strip().split('\n'):
                            if ref_line.strip():
                                group_data = {
                                    "content": ref_line.strip(),
                                    "summary": "",
                                    "global_index": group_counter,
                                    "path": current_node["path"],
                                    "chapter_depth": len(current_node["path"].split('/')) - 1,
                                    "special_type": "reference"
                                }
                                structure["groups"].append(group_data)
                                group_counter += 1
                    else:
                        sentences = self.segment_and_merge_text('\n'.join(current_content))
                        parent_titles = current_path[:-1] + [title] if current_path else [title]
                        tasks.append(self._process_content(sentences, current_node, parent_titles, group_counter))
                current_content = []
                while len(current_path) >= level:
                    current_path.pop()
                current_path.append(title)
                current_node = {"path": "/" + "/".join(current_path)}
                if 'references' in title.lower() and len(title.lower().split(' ')) < 3:
                    in_references = True
                elif in_references and level <= len(current_path):
                    in_references = False
            else:
                if line.strip() and not line.startswith('!['):
                    current_content.append(line)
        if current_content:
            if in_references:
                for ref_line in '\n'.join(current_content).strip().split('\n'):
                    if ref_line.strip():
                        group_data = {
                            "content": ref_line.strip(),
                            "summary": "",
                            "global_index": group_counter,
                            "path": current_node["path"],
                            "chapter_depth": len(current_node["path"].split('/')) - 1,
                            "special_type": "reference"
                        }
                        structure["groups"].append(group_data)
                        group_counter += 1
            else:
                sentences = self.segment_and_merge_text('\n'.join(current_content))
                parent_titles = current_path[:-1] + [current_node["path"].split('/')[-1]] if current_path else ["Root"]
                tasks.append(self._process_content(sentences, current_node, parent_titles, group_counter))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.critical("Contextual Summary Generation Completed.")
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error processing content: {result}")
                continue
            groups, new_counter = result
            for group in groups:
                structure["groups"].append(group)
            group_counter = max(group_counter, new_counter)
        figures = await figures_task
        if figures:
            for figure in figures:
                figure_without_index = {k: v for k, v in figure.items() if k != "global_index"}
                structure["groups"].append(figure_without_index)
                group_counter += 1
        logger.critical("Paper Subtree Construction Completed.")
        return structure

    async def save_paper_tree_to_dict(self, structure: Dict) -> Dict:
        paper_tree_dict = {}
        path_to_content = defaultdict(list)
        path_to_groups = defaultdict(list)
        ordered_paths = []
        seen_paths = set()
        for group in structure["groups"]:
            path = group["path"]
            if group.get("chapter_depth", 0) > 0:
                content = group["content"]
                current_path = path
                path_to_content[current_path].append(content)
                group_data = {
                    "content": group["content"],
                    "summary": group.get("summary", ""),
                    "special_type": group.get("special_type", "content")
                }
                if group.get("special_type") not in ("reference", "figure"):
                    group_data["global_index"] = group["global_index"]
                if group.get("special_type") == "figure":
                    group_data["url"] = group["url"]
                path_to_groups[current_path].append(group_data)
                if current_path not in seen_paths:
                    ordered_paths.append(current_path)
                    seen_paths.add(current_path)
                path_parts = current_path.strip('/').split('/')
                for i in range(1, len(path_parts)):
                    parent_path = '/' + '/'.join(path_parts[:i])
                    if parent_path not in seen_paths:
                        ordered_paths.append(parent_path)
                        seen_paths.add(parent_path)
        for path in ordered_paths:
            title = path.split('/')[-1]
            own_content = path_to_content.get(path, [])
            own_groups = path_to_groups.get(path, [])
            subchapter_content = []
            subchapter_groups = []
            for sub_path in ordered_paths:
                if sub_path != path and sub_path.startswith(path + '/'):
                    sub_groups = path_to_groups.get(sub_path, [])
                    if not any(g.get("special_type") in ("reference", "figure") for g in sub_groups):
                        subchapter_content.extend(path_to_content.get(sub_path, []))
                        subchapter_groups.extend(sub_groups)
            combined_content = own_content + subchapter_content
            combined_groups = own_groups + subchapter_groups
            reindexed_groups = []
            current_index = 0
            for group in combined_groups:
                new_group = group.copy()
                if group.get("special_type") != "figure":
                    new_group["global_index"] = current_index
                    current_index += 1
                reindexed_groups.append(new_group)
            paper_tree_dict[title] = {
                "path": path,
                "content": combined_content,
                "groups": reindexed_groups
            }
        return paper_tree_dict

    async def build_paper_tree(self, root_directory: str) -> Dict:
        from data_clean import data_clean
        file_processor = data_clean(root_directory=root_directory, config=self.config)
        structure = await self.parse_md_to_structure(file_processor.docs_text, file_processor.json_file_path)
        paper_tree_dict = await self.save_paper_tree_to_dict(structure)
        return paper_tree_dict
