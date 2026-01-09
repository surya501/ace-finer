#!/usr/bin/env python3
"""Main entry point for ACE Framework."""

import argparse
import asyncio
import logging
import os
from dotenv import load_dotenv

from data.pipeline import stream_finer, batch_samples
from playbook.store import PlaybookStore
from agents.llm import LLMClient
from agents.generator import Generator
from agents.reflector import Reflector
from agents.curator import Curator
from runner import run_batch
from metrics import MetricsLogger
from guardrails import Guards, GuardError

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="ACE Framework for FiNER-139")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Maximum samples to process (default: all)")
    p.add_argument("--batch-size", type=int, default=20,
                   help="Batch size for processing")
    p.add_argument("--max-concurrent", type=int, default=10,
                   help="Maximum concurrent API calls")
    p.add_argument("--curation-freq", type=int, default=100,
                   help="Run curation every N samples")
    p.add_argument("--checkpoint-freq", type=int, default=1000,
                   help="Checkpoint every N samples")
    p.add_argument("--max-cost", type=float, default=5.0,
                   help="Maximum API cost in dollars")
    p.add_argument("--model", type=str, default="openai/gpt-oss-20b",
                   help="Model to use (default: openai/gpt-oss-20b)")
    return p.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Check for API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY not found in environment")
        return

    log.info(f"Starting ACE Framework")
    log.info(f"  Model: {args.model}")
    log.info(f"  Max samples: {args.max_samples or 'unlimited'}")
    log.info(f"  Batch size: {args.batch_size}")
    log.info(f"  Max cost: ${args.max_cost}")

    # Initialize guards first (need cost callback)
    guards = Guards(max_cost=args.max_cost)

    # Initialize LLM with cost callback
    llm = LLMClient(
        api_key=api_key,
        model=args.model,
        on_cost=guards.record_cost  # Wire cost tracking
    )

    # Initialize components
    store = PlaybookStore(path="./playbook_db")
    generator = Generator(llm)
    reflector = Reflector(llm)
    curator = Curator(store, llm)
    metrics = MetricsLogger(window=100)

    step = 0
    samples_iter = stream_finer(limit=args.max_samples)

    try:
        for batch in batch_samples(samples_iter, args.batch_size):
            # Process batch concurrently
            states = await run_batch(
                batch, store, generator, reflector, guards,
                max_concurrent=args.max_concurrent
            )

            # Update metrics and check guards
            for state in states:
                step += 1
                metrics.record(state)
                guards.check_all(step)

            # Curate periodically
            if step % args.curation_freq < args.batch_size:
                report = await curator.run()
                log.info(f"Curation: {report}")

            # Checkpoint periodically
            if step % args.checkpoint_freq < args.batch_size:
                store.checkpoint(f"./checkpoints/checkpoint_{step}")
                metrics.save(f"./checkpoints/metrics_{step}.json")
                log.info(f"Checkpoint saved at step {step}")

            # Log progress
            log.info(f"{metrics.summary()} | cost=${guards.total_cost:.4f}")

    except GuardError as e:
        log.error(f"Guard triggered: {e}")
        store.checkpoint("./checkpoints/checkpoint_error")
        metrics.save("./checkpoints/metrics_error.json")
        raise

    except KeyboardInterrupt:
        log.info("Interrupted, saving checkpoint...")
        store.checkpoint("./checkpoints/checkpoint_interrupted")
        metrics.save("./checkpoints/metrics_interrupted.json")

    finally:
        # Final summary
        log.info(f"Final: {metrics.summary()}")
        log.info(f"Total cost: ${guards.total_cost:.4f}")
        log.info(f"Rules in playbook: {store.count()}")


if __name__ == "__main__":
    asyncio.run(main())
