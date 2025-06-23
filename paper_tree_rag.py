import time
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Optional
import json
import logging
from config import Config

from paper_tree_builder import PaperTreeBuilder
from global_retriever import GlobalRetriever
from local_retriever import LocalRetriever

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


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        # logger.info("Creating a new event loop in main thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

class PaperTreeRAG:
    def __init__(self, config: Config, retrieval_config: Optional[Dict] = None):
        self.config = config
        self.retrieval_config = {
            "rerank_api_url": self.config.rerank_url,
            "rerank_api_token": self.config.rerank_api_key,
            "top_n": 3,
            "min_relevance_score": 0.1,
            "max_retries": 3,
            "retry_delay": 2,
        }
        if retrieval_config:
            self.retrieval_config.update(retrieval_config)
        self.paper_tree_builder = PaperTreeBuilder(config)
        self.global_retriever = GlobalRetriever(config)
        self.local_retriever = LocalRetriever(config, self.retrieval_config)
        self.processed_folders = self.load_processed_folders()
        self.all_papers_path = "output/all_papers.json"

    @staticmethod
    def load_processed_folders(output_dir: str = "cache") -> set:
        processed_file = Path(output_dir) / "processed_folders.json"
        if processed_file.exists():
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except Exception as e:
                logger.error(f"Failed to load processed_folders.json: {e}")
        return set()

    async def save_processed_folders(self, output_dir: str = "cache") -> None:
        processed_file = Path(output_dir) / "processed_folders.json"
        try:
            async with aiofiles.open(processed_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(list(self.processed_folders), ensure_ascii=False, indent=4))
            logger.info(f"Saved processed folders to {processed_file}")
        except Exception as e:
            logger.error(f"Failed to save processed_folders.json: {e}")

    async def build_paper_tree(self, root_directory: str = "./files", output_dir: str = "output") -> None:
        start_time = time.time()
        files_path = Path(root_directory)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        all_papers_file = Path(self.all_papers_path)
        all_papers_json = {"Papers_Root": {}} if not all_papers_file.exists() else json.load(open(all_papers_file, 'r', encoding='utf-8'))

        for folder in files_path.iterdir():
            if folder.is_dir() and folder.name not in self.processed_folders:
                logger.critical(f"Processing new folder: {folder.name}")
                try:
                    paper_data = await self.paper_tree_builder.build_paper_tree(str(folder))
                    if not paper_data:
                        logger.warning(f"No data returned for {folder.name}, skipping")
                        continue
                    for title, data in paper_data.items():
                        if data.get("path"):
                            paper_title = data["path"].split('/')[1]
                            all_papers_json["Papers_Root"][paper_title] = paper_data
                            self.processed_folders.add(folder.name)
                            await self.save_processed_folders("cache")
                            logger.critical(f"Processed folder {folder.name}")
                            break
                    else:
                        logger.warning(f"No valid path found for {folder.name}, skipping")
                except Exception as e:
                    logger.error(f"Failed to process folder {folder.name}: {e}")

        async with aiofiles.open(self.all_papers_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(all_papers_json, ensure_ascii=False, indent=4))
        logger.critical("Paper Tree Construction Completed.")
        logger.critical(f"Paper trees built in {time.time() - start_time:.2f} seconds")

    def list_papers(self) -> List[str]:
        try:
            with open(self.all_papers_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return list(data.get("Papers_Root", {}).keys())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load papers: {e}")
            return []

    async def select_papers(self, retrieval_type: str) -> List[str]:
        paper_titles = self.list_papers()
        if not paper_titles:
            print("No papers available")
            return []
        for i, title in enumerate(paper_titles, 1):
            print(f"{i}. {title}")
        while True:
            try:
                if retrieval_type == "cross":
                    print(f"{len(paper_titles) + 1}. All papers")
                    selected_nums_input = input(
                        f"Select papers to retrieve (enter numbers separated by commas, or {len(paper_titles) + 1} for all papers): "
                    )
                    selected_nums = [int(num.strip()) for num in selected_nums_input.split(",") if num.strip()]
                    if len(paper_titles) + 1 in selected_nums:
                        return paper_titles
                    if all(1 <= num <= len(paper_titles) for num in selected_nums):
                        return [paper_titles[num - 1] for num in selected_nums]
                    else:
                        print("Invalid input, please enter valid numbers.")
                        continue
                elif retrieval_type == "single":
                    selected_num_input = input("Select a paper to retrieve, enter the paper number: ")
                    selected_num = int(selected_num_input.strip())
                    if 1 <= selected_num <= len(paper_titles):
                        return [paper_titles[selected_num - 1]]
                    else:
                        print("Invalid input, please enter a valid number.")
                        continue
            except ValueError:
                print("Invalid input, please enter numbers.")
                continue

    async def global_retrieval(self, question: str, retrieval_type: str) -> str | None:
        selected_papers = await self.select_papers(retrieval_type)
        if not selected_papers:
            print("No valid paper selected")
            return None
        start_time = time.time()
        try:
            answer = await self.global_retriever.global_retrieval(question, retrieval_type, selected_papers, self.all_papers_path)
            logger.critical(f"Global retrieval for question '{question}' completed in {time.time() - start_time:.2f} seconds")
            return answer.get('answer', 'No answer')
        except Exception as e:
            logger.error(f"Global retrieval failed: {e}")
            return None

    async def local_retrieval(self, query: str, retrieval_type: str, pro: bool = True) -> str:
        selected_papers = await self.select_papers(retrieval_type)
        if not selected_papers:
            print("No valid paper selected")
            return ''
        start_time = time.time()
        try:
            result = await self.local_retriever.local_retrieval(query, retrieval_type, selected_papers, pro)
            logger.critical(f"Local retrieval for query '{query}' (pro={pro}) completed in {time.time() - start_time:.2f} seconds")
            return result.get('answer', 'No answer')
        except Exception as e:
            logger.error(f"Local retrieval failed: {e}")
            return 'Local retrieval failed'
