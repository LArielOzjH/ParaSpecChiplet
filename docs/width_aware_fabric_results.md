# Width-Aware Fabric Cost Sweep

The width-aware cost model combines measured reduced-width MLP behavior with a
conservative analytical fabric model. Qwen3-4B-DFlash dimensions are used:
hidden 2560, intermediate 9728, block size 16, five draft layers, dense
attention MAC approximation 26.2M/layer-position, and MLP MAC approximation
74.7M/layer-position.

The sweep keeps attention dense and applies a width fraction to one upper MLP
layer. The chiplet model adds 4096 activation bytes per position, multicast
reuse 4, 20 synchronization cycles, and 0.25 router cycles per position. It
uses the same aggregate compute resource as the monolithic model.

## Result

Under these conservative equal-resource assumptions, the chiplet model loses
to monolithic execution for every tested width and link bandwidth:

| Reduced width | Chiplet/monolithic at 128 B/cycle | At 1024 B/cycle |
|---:|---:|---:|
| 100% | 1.0188 | 1.0050 |
| 75% | 1.0195 | 1.0051 |
| 50% | 1.0203 | 1.0054 |
| 25% | 1.0212 | 1.0056 |

This is not evidence that chiplets can never help. It is a useful boundary:
without parallel specialization, resource overprovisioning, or substantial
multicast reuse beyond the modeled values, chiplet links and synchronization
erase the compute benefit. The primary architecture should therefore be a
grouped monolithic engine; chiplets remain a conditional extension that must
demonstrate a utilization or area/energy advantage not captured by this
equal-resource serial model.

The sweep is analytical/calibrated, not measured chiplet hardware and not an
end-to-end speedup claim. Raw data: `data/width_aware_fabric_sweep.json`.
