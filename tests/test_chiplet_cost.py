from paraspec.chiplet_cost import estimate_chiplet_cost


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

