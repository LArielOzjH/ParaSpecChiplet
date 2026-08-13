import json

from paraspec.trace_io import load_acceptance_trace


def test_load_acceptance_trace_reads_only_dflash_verification_events(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"kind": "dflash_verify", "block_size": 4, "accepted_prefix": 2}),
                json.dumps({"kind": "scheduler", "accepted_prefix": 99}),
                json.dumps({"kind": "dflash_verify", "block_size": 4, "accepted_prefix": 0}),
            ]
        )
        + "\n"
    )

    trace = load_acceptance_trace(path)

    assert trace.block_size == 4
    assert trace.acceptance_lengths == (2, 0)

