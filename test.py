from config import Config
from paper_tree_rag import PaperTreeRAG, always_get_an_event_loop

# Set up environment
config = Config.from_env(prefix="GPT_4o_mini")
loop = always_get_an_event_loop()

# Instantiate PT-RAG
processor = PaperTreeRAG(config)

# Build paper tree
# loop.run_until_complete(processor.build_paper_tree("./files"))


# # Answer questions
#
# Question 1
# Single paper: Macro question based on "Attention Is All You Need"
# result = loop.run_until_complete(
#     processor.global_retrieval(
#         'Please briefly summarize the main components of the '
#         'Transformer model and their functions',
#         'single'))
# print(f"Globe Retrieval Answer for single: \n{result}\n")
#
#
# Question 2
# Single paper: Detail question based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
# result = loop.run_until_complete(
#     processor.local_retrieval(
#         'How is the embedding layer designed and implemented '
#         'in the Vision Transformer?',
#         'single', False))
# print(f"Local Retrieval Answer for single (pro=False): \n{result}\n")
#
#
# Question 3
# Single paper: Multi-hop question based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
# result = loop.run_until_complete(
#     processor.local_retrieval(
#         'What datasets were used for pre-training ViT, '
#         'and what are the titles of their original papers?',
#         'single', True))
# print(f"Local Retrieval Answer for single (pro=True): \n{result}\n")
#
#
# Question 4
# Single paper: Figure question based on "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
# result = loop.run_until_complete(
#     processor.local_retrieval(
#         'What is the main difference between the models of '
#         'the two stages on the left and right sides of Figure 1?',
#         'single', True))
# print(f"Local Retrieval Answer for cross (pro=True): \n{result}\n")
#
#
# Question 5
# Multiple papers: Macro question based on <three processed papers>
# result = loop.run_until_complete(
#     processor.global_retrieval(
#         'Please analyze the relationships between '
#         'the methods proposed in these three papers',
#         'cross'))
# print(f"Globe Retrieval Answer for cross: \n{result}\n")
#
#
# Question 6
# Multiple papers: Detail question based on <BERT> and <Vision Transformer>
# result = loop.run_until_complete(
#     processor.local_retrieval(
#         'Please compare and contrast the embedding modules in '
#         'BERT and ViT',
#         'cross', False))
# print(f"Local Retrieval Answer for cross (pro=False): \n{result}\n")
#
#
# Question 7
# Multiple papers: Multi-hop question based on <BERT> and <Vision Transformer>
# result = loop.run_until_complete(
#     processor.local_retrieval(
#         'What datasets were utilized during the pre-training phases '
#         'of BERT and ViT, respectively? ',
#         'cross', True))
# print(f"Local Retrieval Answer for cross (pro=True): \n{result}\n")

