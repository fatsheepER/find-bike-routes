# 「有效轨迹」的权威口径归匹配阶段，切分阶段的四个预留列删除

九条硬剔除跨两个阶段：切分阶段做前六条，匹配阶段做依赖匹配结果的后四条。我们让 `track_match.is_valid` 成为术语表意义上的**有效轨迹**（能进入匹配的轨迹按构造已过前六条，所以这一列就是九条全通过），并把切分阶段轨迹表里 `match_rate`、`matched_length_m`、`inferred_share`、`matched_path_on_island` 四个恒为 null 的预留列删掉。留着它们的代价是任何下游读到 `tracks.is_valid` 都会拿到一个偏宽约 5% 的集合（12-21：15,527 对 14,755）而不会收到任何提示。

## Considered Options

- 匹配阶段回填那四列、原地重写切分阶段的轨迹表：得到单一权威表，但一个阶段去改写另一个阶段的产物，切分阶段的内容摘要也随之失去意义。
- 匹配阶段另出一张完整的 `tracks_matched`：下游只读一张表，但把切分阶段的每一列都复制一份，两份之间从此可能不一致。

## Consequences

- 下游读 `match_points` / `match_edges` / `match_pieces` 必须 join `track_match` 并按 `is_valid` 过滤——这三张表故意装了全部进入匹配的轨迹，因为被后四条剔掉的那些是唯一能解释这四条规则在剔什么的证据。
- 删列改变切分阶段的产物与内容摘要，需要重跑一次五天全量并刷新 `config/baselines.json`。
- `CONTEXT.md` 的「有效轨迹」条目据此修订，并写明 `tracks.is_valid` 与 `points.is_valid_track` 只表示前六条。
