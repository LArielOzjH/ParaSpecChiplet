#!/usr/bin/env python3
"""Probe a per-cycle state-conditioned DFlash MLP-width schedule.

This script is intended for the official DFlash checkout. It duplicates the
small generation loop so the MLP schedule can change after each target
verification. It is an acceptance experiment, not a speedup implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from paraspec.official_trace import stats_to_verification_events
from paraspec.partial_mlp import reduced_gated_mlp


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    probs = torch.softmax(logits.view(-1, vocab_size) / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def install_dynamic_widths(layers: object, state: dict[str, dict[int, float]], *, draft_layers: int):
    originals = []
    for index, layer in enumerate(layers):
        mlp = layer.mlp
        original = mlp.forward

        def dynamic_forward(
            hidden_states: torch.Tensor,
            *args: object,
            _index: int = index,
            _mlp: object = mlp,
            _original: object = original,
            **kwargs: object,
        ) -> torch.Tensor:
            if args or kwargs:
                raise ValueError("dynamic width probe does not support extra MLP arguments")
            fraction = state["fractions"].get(_index, 1.0)
            if fraction >= 1.0:
                return _original(hidden_states)
            width = max(1, int(_mlp.gate_proj.weight.shape[0] * fraction))
            return reduced_gated_mlp(hidden_states, _mlp, width)

        originals.append((mlp, original))
        mlp.forward = dynamic_forward

    if len(originals) != draft_layers:
        raise ValueError("layer count does not match draft_layers")

    def restore() -> None:
        for mlp, original in originals:
            mlp.forward = original

    return restore


@torch.inference_mode()
def generate_with_selector(
    model: object,
    target: object,
    input_ids: torch.LongTensor,
    *,
    max_new_tokens: int,
    stop_token_ids: list[int],
    state: dict[str, dict[int, float]],
    threshold: int,
):
    from dflash.model import _cuda_time, extract_context_feature
    from transformers import DynamicCache

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = int(model.block_size)
    output_ids = torch.full(
        (1, max_length + block_size), model.mask_token_id,
        dtype=torch.long, device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    target_cache = DynamicCache()
    draft_cache = DynamicCache()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=target_cache,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    acceptance_lengths: list[int] = []
    schedule_names: list[str] = []
    start = num_input_tokens
    previous_acceptance = 0
    while start < max_length:
        if previous_acceptance >= threshold:
            state["fractions"] = {2: 0.5}
            schedule_names.append("layer2_half")
        else:
            state["fractions"] = {}
            schedule_names.append("uniform")
        block_output_ids = output_ids[:, start:start + block_size].clone()
        block_position_ids = position_ids[:, start:start + block_size]
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, draft_cache.get_seq_length():start + block_size],
            past_key_values=draft_cache,
            use_cache=True,
            is_causal=False,
        )[:, 1 - block_size:, :])
        draft_cache.crop(start)
        block_output_ids[:, 1:] = sample(draft_logits)
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=target_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = sample(output.logits)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1])
            .cumprod(dim=1).sum(dim=1)[0].item()
        )
        output_ids[:, start:start + acceptance_length + 1] = block_output_ids[:, :acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        target_cache.crop(start)
        acceptance_lengths.append(acceptance_length + 1)
        previous_acceptance = int(acceptance_length)
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]
        if stop_token_ids and any(token in output_ids[:, num_input_tokens:] for token in stop_token_ids):
            break
    return SimpleNamespace(
        acceptance_lengths=acceptance_lengths,
        schedule_names=schedule_names,
        num_output_tokens=int(min(start + 1, max_length) - num_input_tokens),
    )


def load_prompts(path: Path) -> tuple[str, ...]:
    prompts = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not prompts:
        raise ValueError("prompt file contains no non-empty lines")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("state-conditioned probe requires CUDA")
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path.cwd()))
    dtype = torch.bfloat16
    target = AutoModelForCausalLM.from_pretrained(str(args.target_model), dtype=dtype).eval().to(args.device)
    draft = AutoModel.from_pretrained(str(args.draft_model), trust_remote_code=True, dtype=dtype).eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.target_model))
    restore = install_dynamic_widths(draft.layers, {"fractions": {}}, draft_layers=len(draft.layers))
    state = {"fractions": {}}
    # Reinstall with the state object used by the generation loop.
    restore()
    restore = install_dynamic_widths(draft.layers, state, draft_layers=len(draft.layers))
    records: list[dict] = []
    try:
        for prompt_index, prompt in enumerate(load_prompts(args.prompts)):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
            stats = generate_with_selector(
                draft, target, input_ids, max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [],
                state=state, threshold=args.threshold,
            )
            events = stats_to_verification_events(
                request_id=f"prompt-{prompt_index}", block_size=int(draft.block_size),
                committed_tokens_per_cycle=stats.acceptance_lengths,
                draft_layers=len(draft.layers),
            )
            for event, schedule in zip(events, stats.schedule_names):
                event.update({
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "schedule": schedule,
                    "threshold": args.threshold,
                    "selector": "previous_accepted_prefix",
                    "target_model": str(args.target_model),
                    "draft_model": str(args.draft_model),
                })
                records.append(event)
    finally:
        restore()
    args.output.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(json.dumps({"events": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
