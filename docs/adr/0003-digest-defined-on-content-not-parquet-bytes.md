# 「结果哈希一致」定义在内容上，不在 Parquet 字节上

计划的验收条件之一是「同一输入与配置重复运行时，阶段计数与结果哈希一致」。Parquet 的**字节不可复现**：文件数随并行度变化，压缩块边界随 task 边界变化，footer 携带写入方元数据；Spark 4 默认开启的 AQE 还会动态合并 shuffle 分区，进一步改变输出文件数。对 Parquet 文件取 sha256 当证据必然失败，且失败方式会误导人去查数据。

因此把「结果哈希」定义在内容上：每张输出表按主键全序排序、取指定列、以固定文本格式逐行喂给 sha256，写进 `artifacts/runs/<run-id>/digest.json`，同时记录各表行数与全部阶段计数。验收 = 两次运行的 `digest.json` 内容相同。

## Consequences

- 确定性必须**设计出来**而非跑出来：管线中不得出现 `sample`、`monotonically_increasing_id`，以及未先显式排序的 `collect_list` 取序。所有变换要么是行内函数，要么是按 `TRACK_ID` 分组内的确定性函数。
- AQE 与 `spark.sql.shuffle.partitions` 可以自由调整而不影响验收，因为它们只改字节布局。
- 排序列即是每张表的主键，须在 spec 中写死；换了排序列，摘要就不可比。
