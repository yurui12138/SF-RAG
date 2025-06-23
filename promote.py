def get_true_title(headings):
    newline = '\n'
    return f"""You are a helpful assistant responsible for analyzing and restructuring the hierarchical levels of headings in an academic paper.

Given a list of lines extracted from a paper, where each line starts with "#" (some are actual headings, others are mistakenly marked as such):
{newline.join(headings)}

Task:
1. Determine which lines are actual headings based on their content and context within the paper. Headings typically introduce new sections or subsections and often follow a logical structure such as Introduction, Methods, Results, Discussion, etc.
2. For actual headings, assign the appropriate markdown heading level based on their hierarchical position:
   - The **paper's title** should be marked with "# " — ensure that **only one** heading is designated as the paper title.
   - Major sections (e.g., Abstract, Introduction, Related Work, Methods, Results, Discussion, Conclusion, References, Acknowledgments) should be marked with "## "
   - Subsections within major sections should be marked with "### ", and so on for deeper levels.
3. For lines that are not actual headings (e.g., regular text or equations mistakenly prefixed with "#"), remove the "#" prefix entirely.

Output:
Only return the corrected list, with appropriate markup format for titles, no prefix '#' for non-title lines, and ',' separating entries without any additional explanation or comment."""

def find_most_relevant_section(paper_title, section_list, question):
    return f"""You are a helpful assistant responsible for determining the most relevant section in an academic paper based on a user's question.

Given a user question, the paper's title, and its section hierarchy, please follow these steps to identify the most appropriate section:
1. Analyze the semantic meaning of the question to understand what information is being sought.
2. If the question pertains to the paper's overall content, select the paper title itself (depth=0).
3. For questions about specific methodologies, experimental designs, data analysis, or theoretical frameworks, choose the most semantically relevant section at the deepest hierarchy level that directly addresses the topic.
4. If no section directly matches the question, choose the closest parent section (i.e., a section at a shallower hierarchy level) that encompasses the relevant topic.

---Data---
User Question: {question}
Paper Title: {paper_title}
Section Hierarchy: {', '.join(section_list)}

Output: Only return the title of the most relevant section, excluding the full path and hierarchy depth, and without any explanation or additional content."""

def generate_answer(original_question, context_text):
    return f'''You are a Retrieval-Augmented Generation (RAG) system specialized in academic paper analysis. Your primary role is to provide precise, concise answers to research questions using retrieval results retrieved from the target paper.

**Instructions:**
- Use **only** the information provided in the retrieval results.
- Keep your response as **short as possible**.
- If multiple sources are provided:
  - Integrate information where appropriate.
  - Resolve contradictions.
  - Ensure technical accuracy.
- Respond in the **same language** as the research question ({original_question}).

####### ---Data---  
**Research Question:** {original_question}  
**Retrieval Results:** {context_text}  

####### Output:'''

def decompose_question(question):
    return f"""You are an academic assistant specializing in breaking down complex, multi-hop research questions into simpler, independent sub-queries for effective information retrieval.
    A multi-hop question requires multiple steps or pieces of information to answer fully, whereas a single-hop question can be answered directly from a single source.
    Your task is to decompose the given question into a sequence of sub-queries that can each retrieve specific information from documents or knowledge bases.
    These sub-queries should be clear, standalone, and arranged in a logical order, with foundational questions first.
    Use placeholders like `{{previous_entity}}` to indicate dependencies on answers from previous sub-queries.
    For questions involving citations, prefix the sub-query with `reference: `, and for those involving figures or charts, use `figure: `.
    If the question is single-hop, simply return it as the only sub-query.
    Aim to minimize the number of sub-queries while ensuring all necessary information is covered.

    ---Question---
    {question}
    #######

    ---Examples---
    Example Question 1: How does the proposed method address existing challenges?
    Example Output 1: ['What is the research field of this paper?', 'What are the existing challenges in the field of {{previous_entity}}?', 'How is the proposed method implemented?']

    Example Question 2: What is the title of the reference paper for the dataset used to validate the proposed method?
    Example Output 2: ['What is the name of the dataset used to validate the proposed method?', 'reference: Which paper corresponds to the dataset {{previous_entity}}?']

    Example Question 3: Please explain the model methodology along with its architecture diagram.
    Example Output 3: ['What is the name of the model in this paper?', 'figure: Which diagram shows the architecture of model {{previous_entity}}?', 'What is the workflow or key idea of the model methodology?']

    Output only the list of sub-queries in the order they should be executed, and ',' separating sub-queries without any additional text or formatting."""

def transform_question(question):
    return f"""You are a helpful assistant tasked with converting questions that involve multiple academic papers into generic query statements that can be applied to individual papers.
The goal is to create a single query that, when asked about each paper separately, will retrieve the necessary information to answer the original cross-paper question.
The transformed query should:
- Be clear, specific, and applicable to any single paper.
- Not mention any specific paper titles, authors, or identifiers.
- End with "this paper" to indicate it is about the individual paper being queried, for example, what dataset is used in this paper.
- Enable the retrieval of relevant content that can later be compared or synthesized across papers.

---User's Question---  
{question}  
#######  
Output only the transformed query statement, without any additional text or explanations."""

def entity_extraction(sub_query, context_text):
    return f"""You are a specialized assistant tasked with extracting specific academic entities from provided text in response to a given query.
Academic entities include, but are not limited to, names of datasets, models, algorithms, evaluation metrics, authors, and technical terms.
Given a research-related question and the contextual content, your job is to identify and list the entities that directly answer the question.

#######
---Query and Context---
Question: {sub_query}
Context: {context_text}

####### 
Extraction Guidelines:
1. Determine the type of entity the question is asking for (e.g., datasets, methods, models).
2. Scan the context for mentions of entities that match this type and are explicitly related to the question.
3. Ensure each extracted entity is:
   - A specific term or phrase (e.g., "ResNet-50", "Adam optimizer", "BLEU score").
   - Directly relevant to the question.
   - Unique (do not list duplicates).
   - Complete, including any qualifiers like version numbers or years if present in the text.
4. Preserve the original formatting of the entities as they appear in the text, including capitalization and special characters.
5. If no relevant entities are found, return an empty list.
6. Output format: a comma-separated list within square brackets

####### 
Output Example:
['Entity1', 'Entity2', 'Entity3', ...]"""

def generate_images_summary_promote(img_title):
    return f"""You are a helpful assistant tasked with creating a detailed and structured summary of an academic figure for the purpose of cross-modal retrieval.
Given the figure's title "{img_title}" and its visual content, analyze the figure according to the following components and synthesize them into a coherent description.
Your summary should not only describe the figure but also highlight its role and significance within the paper, using precise technical terminology.

#######
---Figure Analysis Components---  
1. **Overall Description**:  
   - Identify the main theme and purpose of the figure (e.g., illustrating a concept, presenting experimental results, comparing methods).  
2. **Regional Breakdown**:  
   - Divide the figure into distinct regions (e.g., top, bottom, left, right, center).  
   - For each region, describe:  
     - Its location within the figure.  
     - Key visual elements (e.g., graphs, diagrams, images).  
     - Colors, shapes, and any textual or visual annotations.  
     - Symbolic meanings or interpretations of the elements.  
3. **Critical Element Analysis**:  
   - Highlight notable objects or features (e.g., arrows, specific data points, labels).  
   - Explain their interrelationships and functional roles within the figure.  
4. **Annotation Details**:  
   - Describe specific markers (e.g., numbers, letters, legends).  
   - Clarify what each marker refers to and its significance in the context of the figure.  
#######  

Output:  
Provide a unified summary that integrates all the above components into a single, well-organized paragraph.  
- Begin with the overall purpose and theme of the figure.  
- Follow with a spatial description of the regions and their contents.  
- Then, discuss the critical elements and their roles.  
- Conclude with an explanation of the annotations and their meanings.  
- If there are any ambiguities or contradictions in the figure's interpretation, address them explicitly.  
- Use technical language and include keywords that might be relevant for search queries."""

def generate_summary_promote_with_previous_summary(parent_titles, parent_titles_str, previous_summary, content):
    return f'''You are a professional academic text summarization assistant. Your task is to generate a concise summary for the given text fragment to enhance the accuracy of a retrieval-augmented system. The summary should capture the core content of the fragment, incorporate contextual information to address potential information loss due to text segmentation, and be brief (20-40 words), emphasizing key information in the current fragment while avoiding irrelevant details. Please generate a summary for the current fragment based on the following information:

#######
Paper Title: {parent_titles[0]}
Section Path and Title: {parent_titles_str}
Current Fragment Content: {content}
Previous Fragment Summary: {previous_summary}
#######

Generation Requirements:
1. The summary must accurately reflect the core content of the fragment, highlighting technical details (e.g., algorithm names, method descriptions).
2. Incorporate section information and the previous fragment's summary to provide necessary context.
3. Use concise language suitable for similarity computation in retrieval-augmented generation.
Return only the summary of the current fragment's content, excluding any irrelevant information.'''

def generate_summary_promote(parent_titles, parent_titles_str, content):
    return f'''You are a professional academic text summary assistant. Your task is to generate a concise summary for the first text fragment of a given section to enhance the accuracy of a retrieval-augmented system. The summary should capture the core content of the segment, be brief (20-40 words), emphasize key information from the current segment, and avoid irrelevant details. Generate the summary based on the following information:

#######
Paper Title: {parent_titles[0]}
Section Path and Title: {parent_titles_str}
Current Fragment Content: {content}
#######

Generation Requirements:
1. The summary must accurately reflect the core content of the segment, highlighting technical details (e.g., algorithm names, method descriptions).
2. Incorporate section information to provide necessary context.
3. Use concise language suitable for similarity calculations in retrieval-augmented generation.
Return only the summary of the current fragment content, excluding any irrelevant information.'''
