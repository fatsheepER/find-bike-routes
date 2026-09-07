# 冻结划分是运行产物，不进 Git

区域划分一变，下游画像、双流矩阵、过境率、频繁序列和人工标签全部作废。把 `region_cells` 检进 Git 看起来像冻结，其实是把一次 Infomap 求解的输出当成了源码：换机器、换 `uv.lock` 里的 Infomap 版本，同一输入都可能划出另一版格归属，而 Git 里那份不会自己失效。

因此冻结划分由 `regions` 阶段写出，身份是 `region_cells` 的内容摘要。`assign-regions` 把它消费的 digest 写进运行参数；片区标签配置也携带这个 digest，不符就报错。下游只问这一处，不问 Git 里有没有一份 Parquet。

Infomap 的确定性只承诺到「同一机器、同一 `uv.lock` 两次运行逐格一致」。跨机器不承诺。对不上时的可观测事实是：重跑的 digest 与 `baselines.json` 记录不符。Infomap、leidenalg、igraph 的版本进 `environment.json`，先排除库版本再查别的。

## Considered Options

- 把 `region_cells` 检进 Git：换求解器版本或换机器时，仓库里的划分与重跑结果会默默分叉，标签配置绑的是文件而不是内容。
- 承诺跨机器逐格一致：Infomap 没有这个保证，承诺了只会把一次无法复现写成实现缺陷。

## Consequences

- `data/processed/regions/` 与别的阶段产物一样不进 Git。
- 标签配置的校验键是 digest，不是路径、不是 run-id。
- 跨机器差异表现为一次基线比对失败，而不是一次「结果哈希」口头保证。
