from typing import List, Dict, Optional
import json
import requests
import logging
from config import Config
from promote import (
    generate_answer, decompose_question, entity_extraction, transform_question
)
import ast
from base_retriever import BaseRetriever
from focused_rag import FocusedRAG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/retrieval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LocalRetriever(BaseRetriever):
    def __init__(self, config: Config, retrieval_config: Dict):
        super().__init__(config, retrieval_config)
        self.retrieval_config = retrieval_config
        # Initialize FocusedRAG with configurable token budget
        token_budget = retrieval_config.get('token_budget', 2000)
        self.focused_rag = FocusedRAG(
            rerank_function=self.rerank_contents,
            token_budget=token_budget
        )
        self.use_focused_rag = retrieval_config.get('use_focused_rag', True)

    def get_relevant_summaries(self, selected_group: Dict, is_summary_retrieval: bool = False) -> str:
        path = selected_group["path"]
        if selected_group.get("special_type") == "figure":
            return self.clean_text(selected_group.get("summary", ""))
        global_index = selected_group.get("global_index", float('inf')) if is_summary_retrieval else None
        relevant_groups = [
            g for g in self.groups
            if g["path"] == path and g.get("summary") and (not is_summary_retrieval or g.get("global_index", float('inf')) < global_index)
        ]
        relevant_groups.sort(key=lambda x: x.get("global_index", 0))
        return "\n".join(self.clean_text(g["summary"]) for g in relevant_groups) or ""

    def merge_contexts(self, content_contexts: List[Dict], summary_contexts: List[Dict]) -> List[Dict]:
        content_to_context = {}
        for ctx in content_contexts + summary_contexts:
            content = ctx["content"]
            if content not in content_to_context or ctx["relevance_score"] > content_to_context[content]["relevance_score"]:
                content_to_context[content] = ctx
        merged_contexts = list(content_to_context.values())
        merged_contexts.sort(key=lambda x: x["relevance_score"], reverse=True)
        return merged_contexts

    async def process_retrieval(self, query: str, relevant_title: str, item_type: str) -> List[Dict]:
        if item_type in ("reference", "figure"):
            return await self.process_reference_figure_retrieval(query, item_type)
        items, item_to_group = [], {}
        key = "summary" if item_type == "summary" else "content"
        for group in self.groups:
            if group.get("special_type") != "reference" and group.get(key) and relevant_title in group.get("path", ""):
                cleaned_item = self.clean_text(group[key])
                if cleaned_item:
                    items.append(cleaned_item)
                    item_to_group[cleaned_item] = group
        if not items:
            return []
        rerank_results = await self.rerank_contents(query, items)
        contexts = []
        for result in rerank_results[:self.retrieval_config["top_n"]]:
            item = items[result["index"]]
            group = item_to_group[item]
            contexts.append({
                "path": group["path"],
                "content": self.clean_text(group["content"]),
                "summaries": self.clean_text(group["summary"]),
                "relevance_score": result["relevance_score"]
            })
        return contexts

    async def process_reference_figure_retrieval(self, query: str, item_type: str) -> List[Dict]:
        items, item_to_group = [], {}
        content_key = "summary" if item_type == "figure" else "content"
        for group in self.groups:
            if group.get("special_type") == item_type and group.get(content_key):
                cleaned_content = self.clean_text(group[content_key])
                if cleaned_content:
                    items.append(cleaned_content)
                    item_to_group[cleaned_content] = group
        if not items:
            return []
        rerank_results = await self.rerank_contents(query, items)
        if not rerank_results:
            return []
        top_result = rerank_results[0]
        content = items[top_result["index"]]
        group = item_to_group[content]
        result = {
            "path": group["path"],
            "content": group["content"] if top_result["relevance_score"] > 0.1 else "No relevant literature available",
            "summaries": content if item_type == "figure" else "",
            "relevance_score": top_result["relevance_score"]
        }
        if item_type == "figure":
            result["url"] = group.get("url", "")
        return [result]

    async def decompose_question(self, query: str) -> List[str]:
        logger.critical(f"Decompose question <{query}> into several sub-queries.")
        prompt = decompose_question(query)
        try:
            response = await self._call_llm(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            try:
                try:
                    sub_queries = ast.literal_eval(response)
                    if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                        logger.critical(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
                        return sub_queries
                    raise ValueError("Invalid sub-queries format")
                except:
                    response = response.split(',')
                    return response
            except:
                response = response.split(',')
                return response
        except (SyntaxError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to decompose question: {e}")
            return []

    async def generate_final_answer(self, sub_query_results: List, original_question: str) -> Optional[str]:
        context_text_first = ['Given the complexity of the original question, we decomposed it into simpler sub-questions to facilitate content retrieval. The retrieval results below include: each sub-question, corresponding target paper titles, relevant fragments with summary, and fragment section paths.']
        context_text = context_text_first + sub_query_results
        prompt = generate_answer(original_question, '\n'.join(context_text) or "No relevant content")
        try:
            return await self._call_llm(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Insufficient information to generate answers."

    def _build_context_map(self, sub_query_results: List[Dict]) -> Dict:
        content_map = {}
        for result in sub_query_results:
            query = result['query']
            for ctx in result.get("contexts", []):
                content = ctx['content']
                if content not in content_map:
                    content_map[content] = {'path': ctx['path'], 'summaries': ctx['summaries'] or '无', 'queries': []}
                if query not in content_map[content]['queries']:
                    content_map[content]['queries'].append(query)
        return content_map

    async def cross_paper_retrieval(self, query: str, selected_papers: List[str], turbo: bool) -> Dict:
        result = {"answer": None, "contexts": [], "strategy": "cross_paper", "error": None}
        try:
            await self.load_data(selected_papers)
            if turbo:
                return await self._turbo_cross_retrieval(query, selected_papers)
            return await self._non_turbo_cross_retrieval(query, selected_papers)
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Cross-paper retrieval failed: {e}")
        return result

    async def _turbo_cross_retrieval(self, query: str, selected_papers: List[str]) -> Dict:
        result = {"answer": None, "contexts": [], "strategy": "cross_paper", "error": None}
        try:
            sub_query = await self._call_llm(
                prompt=transform_question(query),
                max_tokens=500,
                temperature=0.3
            )
            logger.critical(f"The transformed question is <{sub_query}>.")
        except Exception as e:
            logger.error(f"Failed to decompose question: {e}")
            result["error"] = "Failed to decompose question"
            return result
        sub_queries = await self.decompose_question(sub_query)
        if not sub_queries:
            result["error"] = "Failed to decompose question"
            return result
        sub_query_results = []
        for paper_title in selected_papers:
            await self.load_data([paper_title])
            sub_query_results.extend(await self._process_sub_queries(sub_queries, paper_title))
        result["contexts"] = sub_query_results
        logger.critical(f"Generating answer...")
        if result["contexts"]:
            result["answer"] = await self.generate_final_answer(sub_query_results, query)
        else:
            result["error"] = "No valid contexts retrieved across papers"
        logger.critical(f"The answer has been generated.")
        return result

    async def _non_turbo_cross_retrieval(self, query: str, selected_papers: List[str]) -> Dict:
        logger.critical(f"Transform question <{query}>.")
        result = {"answer": None, "contexts": [], "strategy": "cross_paper", "error": None}
        if not self.paper_data_map:
            return result
        paper_titles = [p for p in selected_papers if p in self.paper_data_map]
        if not paper_titles:
            return result
        try:
            sub_query = await self._call_llm(
                prompt=transform_question(query),
                max_tokens=500,
                temperature=0.3
            )
            logger.critical(f"The transformed question is <{sub_query}>.")
        except Exception as e:
            logger.error(f"Failed to decompose question: {e}")
            result["error"] = "Failed to decompose question"
            return result
        combined_sub_contents_parts = []
        for paper_title in paper_titles:
            sub_query_results = []
            sub_query_results.append(
                f"The following retrieval results are based on the target paper <{paper_title}>, presenting content fragments and their summaries retrieved for each decomposed sub-question, along with the section path of each fragment.")
            logger.critical(f"Process query <{sub_query}> with respect to Paper <{paper_title}>.")
            relevant_section = await self.find_most_relevant_section(sub_query, paper_title)
            if not relevant_section:
                continue
            logger.critical(f"Retrieve relevant fragments from Section <{relevant_section}>.")
            contexts = await self.retrieve_contexts(sub_query, relevant_section, paper_title)
            sub_query_results = sub_query_results + [
                f"### fragment {i}:\n**Content**:\n{self.clean_text(ctx['content'])}\n**Summary of the fragment**:\n{ctx['summaries'] or 'No summary'}\n**Section path of the fragment**: {ctx['path']}"
                for i, ctx in enumerate(contexts, 1)
            ]
            combined_sub_contents_parts.append("\n".join(sub_query_results))
            logger.critical(f"The relevant fragments have been retrieved.")
        if not combined_sub_contents_parts:
            result["error"] = "No relevant content found"
            return result
        logger.critical(f"All sub-questions have been processed.")
        logger.critical(f"Generating answer...")
        final_prompt = generate_answer(query, combined_sub_contents_parts)
        try:
            result["answer"] = await self._call_llm(
                prompt=final_prompt,
                max_tokens=1000,
                temperature=0.3
            )
        except Exception as e:
            logger.error(f"Failed to generate final answer: {e}")
            result["error"] = "Failed to generate final answer"
        logger.critical(f"The answer has been generated.")
        return result

    async def _process_sub_queries(self, sub_queries: List[str], paper_title: str) -> List[Dict]:
        sub_query_results = []
        previous_entities = []
        current_sub_query_results = []
        sub_query_results.append(f"The following retrieval results are based on the target paper <{paper_title}>, presenting content fragments and their summaries retrieved for each decomposed sub-question, along with the section path of each fragment.")

        for i, sub_query in enumerate(sub_queries):
            logger.critical(f"Retrieve Sub-question <{sub_query}> from Paper <{paper_title}>.")
            if "{previous_entity}" in sub_query:
                if not previous_entities:
                    logger.warning("No previous entities found for substitution")
                    relevant_title = await self.find_most_relevant_section(sub_query, paper_title)
                    contexts = await self.retrieve_contexts(sub_query, relevant_title, paper_title)
                    seen_contents = set()
                    sub_query_prompt = [f'For question <{sub_query}>, the most relevant section is <{relevant_title}> (located at path: {contexts[0]["path"].split(relevant_title)[0] + relevant_title}).',
                    f'In Section <{relevant_title}>, the most relevant fragments related to question <{sub_query}> and their summaries are as follows:']
                    current_sub_query_results = [
                        f"### fragment {i}:\n**Content**:\n{self.clean_text(ctx['content'])}\n**Summary of the fragment**:\n{ctx['summaries'] or 'No summary'}"
                        for i, ctx in enumerate(contexts, 1) if
                        self.clean_text(ctx['content']) not in seen_contents and not seen_contents.add(
                            self.clean_text(ctx['content']))
                    ]
                    sub_query_results = sub_query_results + sub_query_prompt + current_sub_query_results
                else:
                    for entity in previous_entities:
                        path = None
                        current_query = sub_query.replace("{previous_entity}", entity)
                        if "reference" in current_query:
                            relevant_title = 'Reference'
                            path = '/' + paper_title + '/' + 'Reference'
                        elif "figure" in current_query:
                            relevant_title = 'All figures and tables'
                            path = '/' + paper_title + '/' + 'All figures and tables'
                        else:
                            relevant_title = await self.find_most_relevant_section(current_query, paper_title)
                        contexts = await self.retrieve_contexts(current_query, relevant_title, paper_title)
                        seen_contents = set()
                        sub_query_prompt = [f'For question <{current_query}>, the most relevant section is <{relevant_title}> (located at path: {path or contexts[0]["path"].split(relevant_title)[0] + relevant_title}).',
                                     f'In Section <{relevant_title}>, the most relevant fragments related to question <{current_query}> and their summaries are as follows:']
                        current_sub_query_results = [
                            f"### fragment {i}:\n**Content**:\n{self.clean_text(ctx['content'])}\n**Summary of the fragment**:\n{ctx['summaries'] or 'No summary'}"
                            for i, ctx in enumerate(contexts, 1) if
                            self.clean_text(ctx['content']) not in seen_contents and not seen_contents.add(
                                self.clean_text(ctx['content']))
                        ]
                        sub_query_results = sub_query_results + sub_query_prompt + current_sub_query_results
            else:
                relevant_title = await self.find_most_relevant_section(sub_query, paper_title)
                contexts = await self.retrieve_contexts(sub_query, relevant_title, paper_title)
                seen_contents = set()
                sub_query_prompt = [f'For question <{sub_query}>, the most relevant section is <{relevant_title}> (located at path: {contexts[0]["path"].split(relevant_title)[0] + relevant_title}).',
                             f'In Section <{relevant_title}>, the most relevant fragments related to question <{sub_query}> and their summaries are as follows:']
                current_sub_query_results = [
                    f"### fragment {i}:\n**Content**:\n{self.clean_text(ctx['content'])}\n**Summary of the fragment**:\n{ctx['summaries'] or 'No summary'}"
                    for i, ctx in enumerate(contexts, 1) if
                    self.clean_text(ctx['content']) not in seen_contents and not seen_contents.add(
                        self.clean_text(ctx['content']))
                ]
                sub_query_results = sub_query_results + sub_query_prompt + current_sub_query_results

            if i + 1 < len(sub_queries) and "{previous_entity}" in sub_queries[i + 1]:
                previous_entities = await self._extract_entities(sub_query, current_sub_query_results) or previous_entities

        logger.critical(f"All sub-questions in Paper <{paper_title}> have been retrieved.")
        return sub_query_results

    async def retrieve_contexts(self, query: str, relevant_title: Optional[str], paper_title: Optional[str] = None) -> List[Dict]:
        if not relevant_title:
            relevant_title = await self.find_most_relevant_section(query, paper_title)
            if not relevant_title:
                return []
        if "reference" in query.lower() or "figure" in query.lower():
            item_type = "reference" if "reference" in query.lower() else "figure"
            special_contexts = await self.process_retrieval(query, relevant_title, item_type)
            return special_contexts
        
        # FocusedRAG Enhanced Retrieval
        if self.use_focused_rag:
            return await self.focused_rag_retrieve_contexts(query, relevant_title, paper_title)
        
        # Original retrieval method
        return self.merge_contexts(
            await self.process_retrieval(query, relevant_title, "content"),
            await self.process_retrieval(query, relevant_title, "summary")
        )
    
    async def focused_rag_retrieve_contexts(self, query: str, relevant_title: str, paper_title: Optional[str] = None) -> List[Dict]:
        """
        FocusedRAG-enhanced context retrieval with section localization and density-guided selection.
        
        Args:
            query: User query
            relevant_title: Primary relevant section title
            paper_title: Paper title for filtering
            
        Returns:
            List of context dictionaries with enhanced precision
        """
        logger.critical(f"Using FocusedRAG for enhanced retrieval")
        
        # Gather candidate sections from the paper
        candidate_sections = []
        for group in self.groups:
            if paper_title and not group["path"].startswith(f"/{paper_title}/"):
                continue
            
            if group.get("special_type") == "reference":
                continue
            
            # Extract section information
            path = group["path"]
            section_title = path.split('/')[-1]
            content = group.get("content", "")
            summary = group.get("summary", "")
            
            if not content:
                continue
            
            # Check if we already have this section
            existing = None
            for sec in candidate_sections:
                if sec['title'] == section_title and sec['path'] == path:
                    existing = sec
                    break
            
            if existing:
                # Append content to existing section
                existing['content'] += " " + content
                if summary:
                    existing['summary'] += " " + summary
            else:
                # New section
                candidate_sections.append({
                    'title': section_title,
                    'path': path,
                    'content': content,
                    'summary': summary,
                    'groups': [group]
                })
        
        if not candidate_sections:
            logger.warning("No candidate sections found for FocusedRAG")
            return []
        
        # Apply FocusedRAG retrieval pipeline
        try:
            focused_result = await self.focused_rag.retrieve(
                query=query,
                sections=candidate_sections,
                token_budget=self.retrieval_config.get('token_budget', 2000)
            )
            
            # Convert FocusedRAG results to context format
            contexts = []
            for sentence_data in focused_result['sentences']:
                content = sentence_data['content']
                score = sentence_data['relevance_score']
                
                # Find source section for this sentence
                source_section = None
                for section in focused_result['sections']:
                    if content in section.get('content', ''):
                        source_section = section
                        break
                
                if source_section:
                    contexts.append({
                        "path": source_section['path'],
                        "content": content,
                        "summaries": source_section.get('summary', ''),
                        "relevance_score": score
                    })
            
            logger.critical(f"FocusedRAG retrieved {len(contexts)} high-quality contexts")
            return contexts
            
        except Exception as e:
            logger.error(f"FocusedRAG retrieval failed: {e}, falling back to original method")
            # Fallback to original method
            return self.merge_contexts(
                await self.process_retrieval(query, relevant_title, "content"),
                await self.process_retrieval(query, relevant_title, "summary")
            )

    async def _extract_entities(self, sub_query: str, contexts: List) -> List[str]:
        if not contexts:
            return []
        context_text = "\n".join(self.clean_text(ctx.split('**Content**:')[-1].split('**Summary of the fragment**:')[0]) for ctx in contexts)
        prompt = entity_extraction(sub_query, context_text)
        response = await self._call_llm(
            prompt=prompt,
            max_tokens=100,
            temperature=0.3
        )
        try:
            entities = ast.literal_eval(response)
            logger.critical(f"Extracted entities: {entities}")
            return entities
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            response = response.split(',')
            return response

    async def local_retrieval(self, query: str, retrieval_type: str, selected_papers: List[str], turbo: bool = True) -> Dict:
        if retrieval_type == "cross":
            return await self.cross_paper_retrieval(query, selected_papers, turbo)
        if retrieval_type != "single":
            logger.error("Invalid retrieval type. Must be 'single' or 'cross'.")
            return {}

        result = {"answer": None, "contexts": [], "strategy": "", "error": None}
        try:
            await self.load_data(selected_papers)
            if turbo:
                return await self._turbo_local_retrieval(query, selected_papers)
            return await self._non_turbo_local_retrieval(query, selected_papers)
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Local retrieval failed: {e}")
        return result

    async def _turbo_local_retrieval(self, query: str, selected_papers: List[str]) -> Dict:
        result = {"answer": None, "contexts": [], "strategy": "multi_hop", "error": None}
        sub_queries = await self.decompose_question(query)
        if not sub_queries:
            result["error"] = "Failed to decompose question"
            return result
        sub_query_results = await self._process_sub_queries(sub_queries, selected_papers[0])

        logger.critical(f"All sub-questions have been processed.")
        logger.critical(f"Generating answer...")
        result["contexts"] = sub_query_results
        if result["contexts"]:
            result["answer"] = await self.generate_final_answer(sub_query_results, query)
        else:
            result["error"] = "No valid contexts retrieved"
        logger.critical(f"The answer has been generated.")
        return result

    async def _non_turbo_local_retrieval(self, query: str, selected_papers: List[str]) -> Dict:
        result = {"answer": None, "contexts": [], "strategy": "", "error": None}
        strategy = "reference" if "reference" in query.lower() else "figure" if "figure" in query.lower() else "merged"
        result["strategy"] = strategy
        relevant_title = await self.find_most_relevant_section(query, selected_papers[0])
        if not relevant_title:
            result["error"] = "No relevant Section found"
            return result
        logger.critical(f"Retrieve relevant fragments from Section <{relevant_title}>.")
        result["contexts"] = await self.retrieve_contexts(query, relevant_title)
        logger.critical(f"The relevant fragments have been retrieved.")
        logger.critical(f"Generating answer...")
        seen_contents = set()
        context_text = [
            f"### fragment {i}:\n**Content**:\n{self.clean_text(ctx['content'])}\n**Summary of the fragment**:\n{ctx['summaries'] or 'No summary'}"
            for i, ctx in enumerate(result["contexts"], 1) if
            self.clean_text(ctx['content']) not in seen_contents and not seen_contents.add(
                self.clean_text(ctx['content']))
        ]
        context_text_first = ['Retrieval results contain: target paper title, question-relevant content fragments with corresponding summary, and its section path.',
                              f'target paper title:{selected_papers[0]}',
                              f'For question <{query}>, the most relevant section is <{relevant_title}> (located at path: {result["contexts"][0]['path'].split(relevant_title)[0] + relevant_title}).',
                              f'In Section <{relevant_title}>, the most relevant fragments related to question <{query}> and their summaries are as follows:']
        context_text = context_text_first + context_text
        prompt = generate_answer(query, "\n".join(context_text) or "No relevant content")
        result["answer"] = await self._call_llm(prompt=prompt, max_tokens=500, temperature=0.3)
        logger.critical(f"The answer has been generated.")
        return result
