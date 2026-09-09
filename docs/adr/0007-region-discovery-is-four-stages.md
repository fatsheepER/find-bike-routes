# 区域发现拆成四个阶段，行程与轨迹归属同归 assign-regions

计划把单元格流、Infomap、去抖套回写成同一步。我们拆成 `order-trips`、`grid-flow`、`regions`、`assign-regions` 四个 CLI、四套运行产物。合在一起时，改穿越、改去抖或改行程口径都要连 Infomap 一起重跑。四个阶段的输入形态（每天一张分区表，或多天合并）、执行位置（executor 上的普通 UDF，或 driver 上的 Infomap）、可复现性质（纯函数，或依赖求解器且只承诺同机同 `uv.lock`）对不上，硬塞进一个阶段只会让内容摘要失去定位。

`order-trips` 现在做，是因为 `regions` 的扫描表要算行程自环比例，而 `order-data.csv` 目前只出现在数据契约锁里。行程配对是每天的 Spark 作业，跟 driver 上跑 Infomap 不是一回事。

行程归属与轨迹归属同归 `assign-regions`。套划分是一个动作的两个对象：广播同一份 `region_cells`，轨迹走去抖后的区域序列，行程按解锁点、上锁点取区域。拆到两个阶段就无法保证用的是同一版划分，12-23 的套回与晴天的套回也会各写一套。

## Considered Options

- 一个阶段做完：notebook 就是这样，单日还过得去。五天加雨天套回之后，改穿越或改去抖都要连 Infomap 一起重跑，扫描与冻结也缠在同一次运行里。
- 行程归属放到 `order-trips`：配对当时还没有冻结划分，归属只能等 `regions` 写完再做。提前放进去会让行程阶段读一份它不生产的表。

## Consequences

- 四个阶段各自一张 `stage_counts_<stage>`，用通用漏斗形状（单位、进入、保留、拒绝），不借用切分/匹配的轨迹-点双三元组列名。`regions` 的 `source_date` 为空，产物不分区。
- `assign-regions` 把它消费的 `region_cells` 内容摘要写进运行参数。下游只问这一处。
- 代价是多三个必须先跑的前置阶段。缺分区时 fail-fast 并点名该先跑哪一个。
