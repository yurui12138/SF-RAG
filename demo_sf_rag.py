"""Example: build the MinerU-backed index and answer a paper question."""

import argparse
import asyncio

from config import Config
from sf_rag import SFRAG


async def main():
    parser = argparse.ArgumentParser(description="Run SF-RAG over MinerU paper blocks")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--paper", action="append", required=True,
                        help="Indexed paper title; repeat for multi-document synthesis")
    parser.add_argument("--multi-hop", action="store_true",
                        help="Enable optional query decomposition (disabled by default)")
    parser.add_argument("--build", action="store_true",
                        help="Index new folders under files/ before answering")
    args = parser.parse_args()

    rag = SFRAG(Config.from_env())
    if args.build:
        await rag.build_index()
    result = await rag.answer(args.question, args.paper, multi_hop=args.multi_hop)
    print(result["answer"])


if __name__ == "__main__":
    asyncio.run(main())
