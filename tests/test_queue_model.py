from paraspec.queue_model import analyze_schedule_queue


SCHEDULES = {
    "uniform": (2, 2, 2, 2),
    "staircase": (2, 2, 1, 1),
}


def test_queue_model_reports_coalescing_fill_and_cost():
    labels = ("uniform", "staircase", "uniform")
    arrival = analyze_schedule_queue(
        labels,
        SCHEDULES,
        batch_capacity=2,
        draft_layers=2,
        mlp_macs_per_layer=100,
        compute_macs_per_cycle=100,
        launch_cycles=1,
        scatter_cycles_per_row=0.0,
        policy="arrival",
    )
    coalesced = analyze_schedule_queue(
        labels,
        SCHEDULES,
        batch_capacity=2,
        draft_layers=2,
        mlp_macs_per_layer=100,
        compute_macs_per_cycle=100,
        launch_cycles=1,
        scatter_cycles_per_row=0.0,
        policy="coalesced",
    )

    assert arrival.batches == 2
    assert arrival.mean_batch_fill == 1.5
    assert coalesced.batches == 2
    assert coalesced.mean_schedule_groups < arrival.mean_schedule_groups
    assert coalesced.total_grouped_cycles == arrival.total_grouped_cycles

