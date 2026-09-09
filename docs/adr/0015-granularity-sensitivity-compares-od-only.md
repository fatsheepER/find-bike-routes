# 跨格边长的 Top-K 出行流比较无效

Status: invalidated on 2026-09-09

原决策认为出行流只依赖订单端点的区域归属，因此可以跨格边长比较 Top-K 区域对。验收后发现，150/200/300 米的备选划分会分别按区域大小和最小单元格独立重编号 `region_id`，这些编号不表示同一空间对象。现有实现直接对 `(from_region, to_region)` 编号元组求 Jaccard，没有先建立跨划分的空间对应关系，所以该数值不能回答「主要流对是否稳定」。

`data/processed/validation/granularity_topk/` 与运行摘要保留作审计痕迹，但标记为无效，不进入报告、答辩或后续判断。粒度敏感性只保留匹配点口径的 AMI、区域中位宽度、区域数和 `capped` 状态。通道流仍不跨格边长比较，因为格边长会同时改变穿越判定与够格门槛，差异不可归因。

## Consequences

- 不引用 `granularity_topk` 的 Jaccard 数值，也不据此判断格边长鲁棒性。
- 若未来确需该指标，必须先用区域几何建立跨划分对应关系，再把两侧 OD 流统一到同一组空间实体；这不属于当前作业范围。
- `data/processed/validation/` 下的备选划分永不被任何下游阶段引用；`assign-regions`、`region-profiles`、`region-sequences` 的 `--regions` 只接受冻结划分。
- `min_component_cells` 按面积等价折算并向上取整（200 米 8 格、300 米 4 格）；对齐区域尺度时只动 `markov_time`，不动这个下限——两个变量一起动就无法归因。
