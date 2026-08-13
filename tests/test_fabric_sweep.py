from paraspec.fabric_sweep import sweep_chiplet_tradeoff


def test_tradeoff_sweep_reports_chiplet_crossover_inputs():
    points = sweep_chiplet_tradeoff(
        depth_by_position=(3, 3, 2, 1),
        base_cost_kwargs={
            "macs_per_layer": 100,
            "compute_macs_per_cycle": 100,
            "activation_bytes_per_position": 64,
            "synchronization_cycles": 2,
            "shared_lower_depth": 1,
            "router_cycles_per_position": 0.25,
        },
        link_bytes_per_cycle_values=(32, 128),
        multicast_reuse_values=(1.0, 4.0),
    )

    assert len(points) == 4
    assert points[0].chiplet_total_cycles > points[-1].chiplet_total_cycles
    assert points[0].monolithic_total_cycles == 14.0
