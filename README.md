# PT-RAG

PT-RAG is a Retrieval-Augmented Generation (RAG) system designed specifically for academic papers. It enables users to build a "paper tree" from a directory of paper files (e.g., PDFs) and answer various types of questions based on single or multiple papers. The system supports both global (macro) and local (detail) retrieval and can handle complex queries, such as multi-hop questions and figure-related inquiries.

## Features

- **Build a paper tree**: Construct a paper tree from a directory containing academic paper files (e.g., PDFs).
- **Answer questions**: Respond to questions based on single or multiple papers.
- **Support for multiple question types**:
  - Macro questions (global retrieval)
  - Detail questions (local retrieval)
  - Multi-hop questions
  - Figure-related questions
- **Handle single and cross-paper queries**: For single-paper questions, the system automatically identifies the relevant paper.

## Installation

To use PT-RAG, ensure the following requirements are met:

- **Python 3.12**

You can install the PT-RAG package by cloning the repository. For example:

```bash
git clone https://github.com/your-repo/pt-rag.git
cd pt-rag
```

*Note: Replace `https://github.com/your-repo/pt-rag.git` with the actual repository URL if applicable.*

### Dependency Installation

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Environment Variable Configuration

Create a `.env` file in the project root directory and populate it with the following environment variables:

```env
GPT_4o_mini.api_key=<your_api_key>
GPT_4o_mini.base_url=https://vip.apiyi.com/v1
GPT_4o_mini.model=gpt-4o-mini
GPT_4o_mini.model_vl=gpt-4o-mini
GPT_4o_mini.rerank_url=https://api.siliconflow.cn/v1/rerank
GPT_4o_mini.rerank_api_key=<your_rerank_api_key>
GPT_4o_mini.rerank_model=BAAI/bge-reranker-v2-m3
GPT_4o_mini.picture_bed_url=<your_picture_bed_url>
```

Replace `<your_api_key>`, `<your_rerank_api_key>`, and `<your_picture_bed_url>` with your actual API keys and URL.

## Processing PDF Files

Before building the paper tree, you need to process PDF files using MinerU. Follow these steps:

1. **Install MinerU**:

   Refer to the [MinerU GitHub repository](https://github.com/opendatalab/mineru) for installation instructions.

2. **Parse PDF Files**:

   Use the following command to parse all PDF files in the `pdf_files` directory and save the results to the `files` directory:

   ```bash
   magic-pdf -p pdf_files -o files -m auto
   ```

   After parsing, the `files` directory will contain all processed files.

## Usage

Follow these steps to use PT-RAG:

### 1. Import Necessary Modules and Initialize Configuration

```python
from config import Config
from paper_tree_rag import PaperTreeRAG, always_get_an_event_loop

# Set up environment
config = Config.from_env(prefix="GPT_4o_mini")
loop = always_get_an_event_loop()
```

Ensure that the required environment variables are correctly set in the `.env` file (as described in the "Environment Variable Configuration" section).

### 2. Instantiate PT-RAG and Build the Paper Tree

```python
# Instantiate PT-RAG
processor = PaperTreeRAG(config)

# Build paper tree from processed files
loop.run_until_complete(processor.build_paper_tree("./files"))
```

This step builds the paper tree using the parsed files in the `files` directory. For a demonstration of this step, refer to the <font size=1>**Paper Tree Building Demo Video**</font>.

https://github.com/user-attachments/assets/a62e2127-0d60-4db5-9aa4-044da9cd1594

### 3. Answer Questions and Test PT-RAG Performance

PT-RAG supports various question types and retrieval methods. Below are examples of how to use the system for different questions, each accompanied by a corresponding demo video.

#### Example 1: Macro Question on a Single Paper

```python
result = loop.run_until_complete(
    processor.global_retrieval(
        'Please briefly summarize the main components of the '
        'Transformer model and their functions',
        'single'
    )
)
print(f"Single Paper Global Retrieval Answer: \n{result}\n")
```

**Demo Video**: 

https://github.com/user-attachments/assets/df026bab-7b94-4a52-9d10-b14d36420bd1

#### Example 2: Detail Question on a Single Paper

```python
result = loop.run_until_complete(
    processor.local_retrieval(
        'How is the embedding layer designed and implemented '
        'in the Vision Transformer?',
        'single',
        False
    )
)
print(f"Single Paper Local Retrieval Answer (pro=False): \n{result}\n")
```

**Demo Video**: 

https://github.com/user-attachments/assets/fef549bc-5519-46d0-ad12-2d43166457e9

#### Example 3: Multi-hop Question on a Single Paper

```python
result = loop.run_until_complete(
    processor.local_retrieval(
        'What datasets were used for pre-training ViT, '
        'and what are the titles of their original papers?',
        'single',
        True
    )
)
print(f"Single Paper Local Retrieval Answer (pro=True): \n{result}\n")
```

**Demo Video**: 

https://github.com/user-attachments/assets/6ec6310a-baa9-4c4f-905f-ebc8011bd69c

#### Example 4: Figure-Related Question on a Single Paper

```python
result = loop.run_until_complete(
    processor.local_retrieval(
        'What is the main difference between the models of '
        'the two stages on the left and right sides of Figure 1?',
        'single',
        True
    )
)
print(f"Single Paper Local Retrieval Answer (pro=True): \n{result}\n")
```

**Demo Video**: 

https://github.com/user-attachments/assets/4befa596-2f1d-4f3e-99ce-c62364fa22ef

#### Example 5: Macro Question on Multiple Papers

```python
result = loop.run_until_complete(
    processor.global_retrieval(
        'Please analyze the relationships between '
        'the methods proposed in these three papers',
        'cross'
    )
)
print(f"Cross-Paper Global Retrieval Answer: \n{result}\n")
```

**Demo Video**: 

https://github.com/user-attachments/assets/a167851d-4577-4269-90a2-fa9440378094

#### Example 6: Detail Question on Multiple Papers

```python
result = loop.run_until_complete(
    processor.local_retrieval(
        'Please compare and contrast the embedding modules in '
        'BERT and ViT',
        'cross',
        False
    )
)
print(f"Cross-Paper Local Retrieval Answer (pro=False): \n{result}\n")
```

**Demo Video**:

https://github.com/user-attachments/assets/45df602a-9b77-4907-9e26-25d34032d27a

#### Example 7: Multi-hop Question on Multiple Papers

```python
result = loop.run_until_complete(
    processor.local_retrieval(
        'What datasets were utilized during the pre-training phases '
        'of BERT and ViT, respectively?',
        'cross',
        True
    )
)
print(f"Cross-Paper Local Retrieval Answer (pro=True): \n{result}\n")
```

**Demo Video**:

https://github.com/user-attachments/assets/e36b830e-72ec-4270-8396-898b261c1eea

*Note: Ensure that the demo video files (e.g., `paper_tree_building_demo.mp4`, `question_1_macro_single_demo.mp4`, etc.) are placed in the project root directory or a designated video directory for user reference.*

## Running the Test Script

To test the PT-RAG system, run the `test.py` script, which includes code for building the paper tree and answering various questions. Before running the script, ensure that the `files` directory contains the parsed PDF files.

```bash
python test.py
```

The script executes the following three stages:
1. **Initialization**: Import modules and configure the environment.
2. **Build Paper Tree**: Construct the paper tree using the parsed files in the `files` directory (see `paper_tree_building_demo.mp4`).
3. **Test Performance**: Evaluate PT-RAG's retrieval and answering capabilities using different question types (see demo videos for each question).
