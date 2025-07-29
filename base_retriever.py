import asyncio
import re
from typing import Tuple

import requests
from openai import AsyncOpenAI
from promote import find_most_relevant_section
import json
from typing import List, Dict, Optional
import logging
from config import Config

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

class BaseRetriever:
    def __init__(self, config: Config, retrieval_config: Dict):
        self.config = config
        self.llm = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self.sections: List[Tuple[str, str, int]] = []
        self.groups: List[Dict] = []
        self.paper_title: Optional[str] = None
        self.title_path: Optional[str] = None
        self.paper_data_map: Dict[str, Dict] = {}
        self.retrieval_config = retrieval_config

    def fix_third_level_paths(self, data):
        for key1, value1 in data.items():
            for key2, value2 in value1.items():
                for key3, value3 in value2.items():
                    if isinstance(value3, dict) and 'path' in value3:
                        original_path = value3['path']
                        expected_prefix = f"/{key2}"

                        if not original_path.startswith(expected_prefix):
                            parts = original_path.split('/', 2)
                            new_path = f"{expected_prefix}/{parts[2]}" if len(parts) > 2 else expected_prefix
                            value3['path'] = new_path

    async def load_data(self, selected_papers: List[str]) -> bool:
        try:
            with open("output/all_papers.json", 'r', encoding='utf-8') as file:
                queries_dict = json.load(file)
            if not queries_dict:
                logger.error("Empty JSON file")
                return False

            self.fix_third_level_paths(queries_dict)

            queries_dict = self._filter_papers(queries_dict, selected_papers)
            logger.info("Loaded output/all_papers.json")

            self._reset_data()  
            return self._process_paper_data(queries_dict)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load JSON: {e}")
            return False

    def _filter_papers(self, queries_dict: Dict, selected_papers: List[str]) -> Dict:
        if not queries_dict.get("Papers_Root"):
            return {"Papers_Root": {}}

        papers = {}
        for paper_title in selected_papers:
            paper_content = queries_dict["Papers_Root"].get(paper_title, {})
            for item in paper_content.values():
                if 'path' in item:
                    path = item['path']
                    target_prefix = f"/{paper_title}/"
                    if not path.startswith(target_prefix):
                        parts = path.strip('/').split('/', 1)
                        new_path = f"/{paper_title}/{parts[1]}" if len(parts) > 1 else f"/{paper_title}"
                        item['path'] = new_path
            papers[paper_title] = paper_content

        return {"Papers_Root": papers}

    def _reset_data(self):
        self.sections = []
        self.groups = []
        self.paper_title = None
        self.title_path = None
        self.paper_data_map = {}


    async def rerank_contents(self, query: str, items: List[str]) -> List[Dict]:
        payload = {
            "model": self.config.rerank_model,
            "query": query,
            "documents": [self.clean_text(item) for item in items],
            "top_n": self.retrieval_config["top_n"],
            "return_documents": False,
            "max_chunks_per_doc": 1024,
            "overlap_tokens": 80
        }
        headers = {
            "Authorization": f"Bearer {self.retrieval_config['rerank_api_token']}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(self.retrieval_config["rerank_api_url"], json=payload, headers=headers)
            response.raise_for_status()
            results = response.json().get("results", [])
            filtered_results = [r for r in results if r.get("relevance_score", 0) >= self.retrieval_config["min_relevance_score"]]
            return filtered_results or results[:self.retrieval_config["top_n"]]
        except requests.RequestException as e:
            logger.error(f"Rerank API request failed: {e}")
            raise

    def _process_paper_data(self, queries_dict: Dict) -> bool:
        min_path_length = float('inf')
        is_multi_paper = "Papers_Root" in queries_dict and isinstance(queries_dict["Papers_Root"], dict)  

        if is_multi_paper:
            logger.info("Processing all_papers.json format")
            for paper_title, paper_data in queries_dict["Papers_Root"].items():
                if not isinstance(paper_data, dict):
                    logger.warning(f"Invalid paper data for {paper_title}, skipping")
                    continue
                self.paper_data_map[paper_title] = paper_data
                min_path_length = self._extract_sections_and_groups(paper_data, min_path_length)
            self.paper_title = "Multiple Papers"
        else:
            logger.info("Processing single paper JSON format")

        self.sections.sort(key=lambda x: (x[2], x[1]))
        if not self.paper_title:
            self.paper_title = "Unknown Paper"
            self.title_path = "/Unknown hospitalizations"
        if not self.sections:
            logger.error("No valid sections loaded")
            return False

        logger.critical(f"Loaded {len(self.sections)} sections and {len(self.groups)} groups")
        return True

    def _extract_sections_and_groups(self, data: Dict, min_path_length: float) -> float:
        min_length = min_path_length
        for title, info in data.items():
            if not title or not isinstance(title, str) or not isinstance(info, dict):
                continue
            path = info.get("path", "")
            if not path or not path.startswith('/'):
                continue
            path_fragments = path.strip('/').split('/')
            path_length = len(path_fragments)
            if path_length < min_length:
                min_length = path_length
                self.paper_title = title
                self.title_path = path
            depth = len(path_fragments) - 1
            self.sections.append((title, path, depth))
            for group in info.get("groups", []):
                group_data = {
                    "path": path,
                    "content": group.get("content", ""),
                    "summary": group.get("summary", ""),
                    "special_type": group.get("special_type", "content"),
                    "global_index": group.get("global_index", None),
                    "section_depth": depth,
                    "url": group.get("url", "") if group.get("special_type") == "figure" else ""
                }
                self.groups.append(group_data)
        return min_length

    async def find_most_relevant_section(self, question: str, paper_title: Optional[str] = None) -> Optional[str]:
        if not self.sections:
            return None
        logger.critical(f"Extract the subtree of the paper <{paper_title}>.")
        if paper_title:
            relevant_sections = [
                (title, path, depth)
                for title, path, depth in self.sections
                if path.startswith(f"/{paper_title}/")
            ]
            relevant_sections.append((paper_title, f"/{paper_title}", 0))
        else:
            relevant_sections = self.sections
        if not relevant_sections:
            return None
        logger.critical(f"Paper subtree of <{paper_title}> extraction completed.")
        section_list = [f"(depth={depth}) {path}" for _, path, depth in relevant_sections]
        logger.critical(f"Select the section most relevant to question <{question}>.")
        prompt = find_most_relevant_section(paper_title or self.paper_title, section_list, question)
        try:
            response = await self._call_llm(
                prompt=prompt,
                max_tokens=50,
                temperature=0.3
            )
            selected_path = response.strip()
            for _, path, _ in relevant_sections:
                if path.split('/')[-1] in selected_path or selected_path in path.split('/')[-1]:
                    logger.critical(f"The title most relevant to question <{question}> is <{path.split('/')[-1]}>")
                    return path.split('/')[-1]
            logger.critical(f"The title most relevant to question <{question}> is  <{paper_title}>")
            return paper_title
        except Exception as e:
            section_titles = []
            for _, path, depth in relevant_sections:
                section_titles.append(path)
            paper_title_index = await self.rerank_contents(question, section_titles)
            top_title = section_titles[paper_title_index[0]['index']]
            logger.error(f"Error in title relevance judgment: {e}")
            return top_title

    async def _call_llm(self, prompt: str, max_tokens: int, temperature: float) -> str:
        for attempt in range(3):
            try:
                response = await self.llm.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    raise
        raise Exception("Max retries reached for LLM call")

    @staticmethod
    def clean_text(text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip())
