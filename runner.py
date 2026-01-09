"""Runner for processing samples through the ACE pipeline."""

import asyncio
from state import State
from data.pipeline import Sample
from data.labels import classify_error
from playbook.store import PlaybookStore
from agents.generator import Generator
from agents.reflector import Reflector
from guardrails import Guards


async def run_sample(
    sample: Sample,
    store: PlaybookStore,
    generator: Generator,
    reflector: Reflector,
    guards: Guards
) -> State:
    """
    Process one sample through the full pipeline.

    Steps:
    1. Retrieve relevant rules
    2. Generate predictions
    3. Evaluate against ground truth
    4. Update rule stats
    5. Reflect on errors and create new rules

    Args:
        sample: Input sample to process
        store: PlaybookStore for rule retrieval/storage
        generator: Generator for predictions
        reflector: Reflector for rule generation
        guards: Guards for safety checks

    Returns:
        Completed State object
    """
    state = State(
        sample_id=sample.id,
        sentence=sample.sentence,
        tokens=sample.tokens,
        ground_truth=sample.ner_labels
    )

    # 1. Retrieve rules
    state.retrieved_rules = store.retrieve(state.sentence, top_k=5)

    # 2. Generate predictions
    state.predictions, state.parse_failed = await generator.predict(
        state.tokens,
        state.retrieved_rules
    )

    if state.parse_failed:
        guards.record_parse_failure()
    else:
        guards.record_parse_success()

    # 3. Evaluate
    state.is_correct = (state.predictions == state.ground_truth)
    if not state.is_correct:
        state.error_type = classify_error(state.predictions, state.ground_truth)

    # 4. Update rule stats
    for rule in state.retrieved_rules:
        store.update_stats(rule.rule_id, success=state.is_correct)

    # 5. Reflect on errors
    if not state.is_correct:
        new_rule = await reflector.generate_rule(state)

        if new_rule:
            # Validate rule fixes the error
            if await reflector.validate(new_rule, state, generator):
                # Check for duplicates
                embedding = store.get_embedding(new_rule.content)
                if guards.check_duplicate(embedding):
                    store.add_rule(new_rule)
                    state.new_rule = new_rule
                    guards.record_rule_created()

    guards.record_sample()
    return state


async def run_batch(
    samples: list[Sample],
    store: PlaybookStore,
    generator: Generator,
    reflector: Reflector,
    guards: Guards,
    max_concurrent: int = 10
) -> list[State]:
    """
    Process batch with limited concurrency.

    Args:
        samples: List of samples to process
        store: PlaybookStore for rule retrieval/storage
        generator: Generator for predictions
        reflector: Reflector for rule generation
        guards: Guards for safety checks
        max_concurrent: Maximum concurrent tasks

    Returns:
        List of completed State objects
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def process(sample):
        async with sem:
            return await run_sample(sample, store, generator, reflector, guards)

    return await asyncio.gather(*[process(s) for s in samples])
