from paraspec.official_trace import stats_to_verification_events


def test_stats_to_events_removes_the_target_fallback_token():
    events = stats_to_verification_events(
        request_id="r0",
        block_size=8,
        committed_tokens_per_cycle=(4, 1, 8),
        draft_layers=3,
        stage_latency_us={"draft": 10.0, "verify": 20.0},
    )

    assert [event["accepted_prefix"] for event in events] == [3, 0, 7]
    assert events[0]["kind"] == "dflash_verify"
    assert events[0]["request_id"] == "r0"
    assert events[0]["block_size"] == 8
    assert events[0]["draft_layers"] == 3
    assert events[0]["stage_latency_us"] == {"draft": 10.0, "verify": 20.0}


def test_stats_to_events_rejects_invalid_commit_lengths():
    try:
        stats_to_verification_events(
            request_id="r0",
            block_size=8,
            committed_tokens_per_cycle=(0,),
        )
    except ValueError as exc:
        assert "committed" in str(exc)
    else:
        raise AssertionError("expected invalid committed length to be rejected")
