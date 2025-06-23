from promote import generate_answer, transform_question
from typing import List, Optional, Dict
import logging
from base_retriever import BaseRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/retrieval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GlobalRetriever(BaseRetriever):
    async def generate_answer(self, question: str, relevant_title: str, paper_title: Optional[str] = None) -> Dict:
        result = {"answer": None, "content": [], "strategy": "", "error": None}
        logger.critical(f"Extract the content of section <{relevant_title}>")
        path = f"/{paper_title}/{relevant_title}"
        if paper_title:
            content = []
            path = None
            for g in self.groups:
                if g["path"].startswith(f"/{paper_title}") and g["path"].endswith(relevant_title):
                    content.append(g["content"])
                    if path is None:
                        path = g["path"]
        else:
            content = []
            path = None
            for g in self.groups:
                if g["path"].endswith(relevant_title):
                    content.append(g["content"])
                    if path is None:
                        path = g["path"]

        logger.critical(f"Section content extraction is complete.")

        context_text_first = [
            'Retrieval results contain: target paper title, question-relevant section title with corresponding content, and its section path.',
            f'target paper title:{paper_title}',
            f'For question <{question}>, the most relevant section is <{relevant_title}> (located at path: {path}).',
            f'the content of <{relevant_title}> section:']
        context_text = context_text_first + content

        logger.critical(f"Generating answer...")
        prompt = generate_answer(question, "\n".join(context_text) or "No relevant content")
        try:
            result['answer'] = await self._call_llm(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3
            )
            result['content'] = content
            logger.critical(f"The answer has been generated.")
            return result
        except Exception as e:
            logger.error(f"Error in answer generation: {e}")
            return {}

    async def cross_paper_retrieval(self, question: str, selected_papers: List[str]) -> Dict:
        result = {"answer": None, "content": [], "strategy": "", "error": None}
        if not self.paper_data_map:
            return {}
        if list(self.paper_data_map.keys()) == selected_papers:
            paper_titles = list(self.paper_data_map.keys())
        else:
            paper_titles = [p for p in selected_papers if p in self.paper_data_map]
        if not paper_titles:
            return {}

        logger.critical(f"Transform question <{question}>.")
        decompose_prompt = transform_question(question)
        try:
            transformed_query = await self._call_llm(
                prompt=decompose_prompt,
                max_tokens=500,
                temperature=0.3
            )
            logger.critical(f"The transformed question is <{transformed_query}>.")
        except Exception as e:
            logger.error(f"Failed to decompose question: {e}")
            return {}

        combined_sub_contents_parts = []
        for paper_title in paper_titles:
            logger.critical(f"Process query <{transformed_query}> with respect to Paper <{paper_title}>.")
            relevant_chapter = await self.find_most_relevant_section(transformed_query, paper_title)
            if not relevant_chapter:
                continue
            logger.critical(f"Extract the content of section <{relevant_chapter}>")
            paper_data = self.paper_data_map[paper_title]
            if relevant_chapter in paper_data:
                groups = paper_data[relevant_chapter].get("groups", [])
                content = [g["content"] for g in groups]
                content_str = "\n".join(content) if content else "No relevant content"
                chapter_desc = {"All content of this paper" if relevant_chapter == paper_title else relevant_chapter}
                part = f"target paper title: {paper_title}\nquestion-relevant section title with section path: {chapter_desc}\nThe content of {chapter_desc} section: {content_str}"
                combined_sub_contents_parts.append(part)
            logger.critical(f"Section content extraction is complete.")

        context_text_first = [
            'Retrieval results include: titles of relevant papers, titles of question-related section along with their content, and the hierarchical section paths for each extracted section content from multiple source papers.']
        combined_sub_contents_parts = context_text_first + combined_sub_contents_parts

        logger.critical(f"Generating answer...")
        final_prompt = generate_answer(question, "\n".join(combined_sub_contents_parts) or "No relevant content")
        try:
            result['answer'] = await self._call_llm(
                prompt=final_prompt,
                max_tokens=1000,
                temperature=0.3
            )
            result['content'] = combined_sub_contents_parts
            logger.critical(f"The answer has been generated.")
            return result
        except Exception as e:
            logger.error(f"Failed to generate final answer: {e}")
            return {}

    async def global_retrieval(self, question: str, retrieval_type: str, selected_papers: List[str], json_path: str = "output/all_papers.json") -> Dict:
        await self.load_data(selected_papers)
        if retrieval_type == "single":
            if len(selected_papers) != 1:
                logger.error("For single-paper retrieval, exactly one paper should be selected.")
                return {}
            paper_title = selected_papers[0]
            relevant_title = await self.find_most_relevant_section(question, paper_title)
            if not relevant_title:
                return {}
            return await self.generate_answer(question, relevant_title, paper_title)
        elif retrieval_type == "cross":
            return await self.cross_paper_retrieval(question, selected_papers)
        else:
            logger.error("Invalid retrieval type. Must be 'single' or 'cross'.")
            return {}
