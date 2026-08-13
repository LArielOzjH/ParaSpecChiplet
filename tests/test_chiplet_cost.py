from paraspec.chiplet_cost import (
    estimate_chiplet_cost,
    estimate_mlp_gated_cost,
    estimate_monolithic_cost,
)


def test_chiplet_cost_exposes_compute_link_and_sync_components():
    result = estimate_chiplet_cost(
        depth_by_position=(3, 3, 2, 2),
        macs_per_layer=100,
        compute_macs_per_cycle=100,
        activation_bytes_per_position=64,
        link_bytes_per_cycle=128,
        synchronization_cycles=2,
    )

    assert result.compute_cycles == 10.0
    assert result.link_cycles == 2.0
    assert result.synchronization_cycles == 2.0
    assert result.total_cycles == 14.0


def test_chiplet_cost_models_shared_lower_layers_multicast_and_router_overhead():
    result = estimate_chiplet_cost(
        depth_by_position=(3, 3, 2, 2),
        macs_per_layer=100,
        compute_macs_per_cycle=100,
        activation_bytes_per_position=64,
        link_bytes_per_cycle=128,
        synchronization_cycles=2,
        shared_lower_depth=2,
        activation_multicast_reuse=4,
        router_cycles_per_position=0.5,
    )

    assert result.compute_cycles == 4.0
    assert result.link_cycles == 0.5
    assert result.router_cycles == 2.0
    assert result.total_cycles == 8.5


def test_monolithic_cost_accounts_for_idle_lanes_at_max_depth():
    result = estimate_monolithic_cost(
        depth_by_position=(3, 3, 2, 1),
        macs_per_layer=100,
        compute_macs_per_cycle=100,
        synchronization_cycles=2,
    )

    assert result.compute_cycles == 12.0
    assert result.total_cycles == 14.0


def test_mlp_gated_cost_keeps_attention_dense_but_scales_mlp_by_schedule():
    result = estimate_mlp_gated_cost(
        depth_by_position=(3, 3, 3, 2),
        attention_macs_per_layer=40,
        mlp_macs_per_layer=60,
        compute_macs_per_cycle=100,
        synchronization_cycles=2,
    )

    assert result.compute_cycles == 11.4
    assert result.total_cycles == 13.4
