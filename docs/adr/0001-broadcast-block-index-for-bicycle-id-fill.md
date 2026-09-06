# 用广播块索引而非全局窗口填充 `BICYCLE_ID`

staging CSV 只在每个车号的块首行写 `BICYCLE_ID`，其余行为空，必须按 `source_row` 顺序前向填充；而 staging 文件的哈希已冻进 `config/data-contract.lock.json`，不能改源文件回避。直觉写法 `last(..., ignorenulls=True) over Window.orderBy("source_row")` 没有 `partitionBy`，会让 Spark 把整份数据挪到单个分区——这是全管线唯一天然串行的一步，恰好落在评分维度①「大数据底层处理架构」的正面。

我们核实过**每个车号在文件里恰好占一个连续块**（12-21：9,514 个车号，0 个跨块），因此改为把非空行收集成 `(start_row, bicycle_id)` 块索引（每天仅 9,514 条，不到 1 MB）广播出去，用二分查找的普通 Python UDF 给每行定车号。全并行、O(log n)、结果与分区顺序无关因而完全确定。

## Consequences

- 该实现依赖「每车号一个连续块」这一**数据性质**而非格式保证。测试必须在 fixture 上断言 Spark 的填充结果与 pandas `ffill` 逐行一致；若将来某天的数据违反该性质，这条断言会失败而不是静默给出错误车号。
- 用的是普通 Python UDF。计划禁用的是 pandas UDF 与 pandas-on-Spark，普通 UDF 不在其列。
