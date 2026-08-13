# Joint Block-Fidelity Frontier

The official MLP-only group trace is converted into a schedule frontier by
`scripts/analyze_block_frontier.py`. For each measured bypass group, the tool
records mean accepted prefix and an analytical nominal MLP-work count:

\[
  W(G)=L-|G|,
\]

where `G` is the bypassed layer set and `L=5` is the draft depth. The work
count assumes a fused implementation and is explicitly not a speedup result.

With `min_survival=1.2`, the frontier from the current Qwen3-4B trace is:

| Bypassed MLP blocks | Mean accepted prefix | Nominal MLP work |
|---|---:|---:|
| `{2}` | 1.3406 | 4/5 |
| `{}` | 1.4525 | 5/5 |

All measured multi-block bypass groups fall below the safety threshold. This
does not prove that no other fidelity schedule is possible; it says that the
currently tested all-or-nothing MLP bypass primitive is too coarse for a
larger static group. The next candidate is partial MLP fidelity (for example,
reduced precision or scaled update) plus a joint schedule table.

The selector should therefore operate on measured joint schedules:

1. reject schedules below the registered prefix-survival threshold;
2. remove schedules dominated in acceptance and nominal work;
3. group requests only when their selected schedule is compatible;
4. use dense fallback when no safe grouped schedule pays for movement and
   synchronization.

The frontier is an empirical controller input, not a claim of serving
performance. A hardware paper still needs a fused MLP implementation and an
equal-resource monolithic baseline.
