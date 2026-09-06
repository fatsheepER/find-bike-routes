# 验证记录

需求 → spec/ticket → 提交 → 测试 → 证据的追踪表。每次正式实验的输入哈希、参数、阶段计数、指标与结论也记在这里。

`requirements.md` 与 `design.md` 尚未建立，本文件先行，追踪链从计划条目直接指向 spec。spec 与 ticket 按仓库约定存放在 `.scratch/`（不进 git，见 `docs/agents/issue-tracker.md`），因此表中的 spec 列是本地路径。

## 追踪表

| # | 需求 | spec / ticket | 提交 | 测试 | 证据 |
|---|---|---|---|---|---|
| 1 | `docs/project-plan.md` §5.1 固化数据契约：边界唯一来源、记录全部输入的哈希与版本、建立 20 车号小样本回归集 | `.scratch/data-contract-freeze/spec.md` | `feat: Freeze the data contract with hashes, versions and a regression fixture`（分支 `data-mining`） | `tests/test_freeze_data_contract.py`（15 例，全部在 `tmp_path` 自造输入上跑，不依赖 `data/`） | `config/data-contract.lock.json`（`contract_version` `1.0`，20 条目）、`config/regression-sample.json`、`tests/fixtures/regression-sample-20201221.csv` |

## 1. 数据契约冻结

### 命令

```bash
uv run python scripts/freeze_data_contract.py                  # 扫描并写锁文件
uv run python scripts/freeze_data_contract.py --check          # 复验，漂移即非零退出
uv run python scripts/freeze_data_contract.py --check --skip-pbf   # 无 41 MB PBF 的环境
uv run python scripts/freeze_data_contract.py --extract-fixture    # 从全量重造 fixture
```

`--only ROLE`（可重复）把检查限定到 `boundary` / `config` / `fixture` / `osm_pbf` / `raw` / `staging` 中的若干个。`--only` 与 `--skip-pbf` 只对 `--check` 有效：写模式下做部分扫描会用一份残缺记录覆盖已提交的锁文件，因此脚本直接拒绝。

### 一次性人工验收（2026-09-06，本机）

在 `data/` 完整的工作区上执行，因为断言的对象是不进 git 的 `data/`，这段验收不进 pytest。

| 动作 | 结果 |
|---|---|
| 默认模式生成锁文件 | 退出 0，写出 20 个条目 |
| `--check` | 退出 0，`data contract check passed: 20 entries verified`，耗时约 2 秒 |
| `--check --skip-pbf` | 退出 0，19 个条目 |
| `--check --only staging --only boundary` | 退出 0，8 个条目 |
| `--skip-pbf`（写模式） | 退出 2 并拒绝，锁文件未被改动 |
| `--extract-fixture` | 退出 0，4,460 行、20 车号；源文件 sha256 前后一致 |

锁文件中五个关键值与 spec 的核实表逐项一致：

| 项 | 锁文件记录 |
|---|---|
| `data/raw/fujian-260901.osm.pbf` | sha256 `ad5d0304…dc09b693`，41,641,197 字节 |
| PBF 快照 | `osmosis_replication_timestamp = 2026-09-01T20:20:50Z`，序号 `772`，来源 `https://download.geofabrik.de/asia/china/fujian-updates`，`generator = osmium/1.16.0`，`sorting = Type_then_ID` |
| `config/xiamen-island.geojson` | sha256 `d29e470a…04381c70`，128,118 字节，relation `14251728` v`7`，MultiPolygon 单部件，UTM 面积 140.68 km² |
| `data/raw/weather/Weather data.csv` | sha256 `c97679d1…9806f8a8`，7,351 字节，144 行，覆盖 `2020-12-20 00:00:00` – `2020-12-25 23:00:00` |
| 电子围栏 | `data_rows` 10,522（计划 §3.1 与论文摘要的 14,071 冲突，此处以文件为准，锁文件即出处） |

### 数据规模声明的出处（对应计划 §3.1）

| 数据 | 锁文件条目 | `data_rows` |
|---|---|---|
| 轨迹点 | `data/staging/trajectory/trajectory-data-2020122{1..5}.csv` | 542,386 + 586,562 + 213,733 + 932,469 + 574,093 = 2,849,243 |
| 订单事件 | `data/staging/order/order-data.csv` | 441,350 |
| 电子围栏 | `data/staging/electronic-fence/station.csv` | 10,522 |
| 天气 | `data/raw/weather/Weather data.csv` | 144（逐小时，12-20 至 12-25） |
| 路网 | `data/raw/fujian-260901.osm.pbf` | 见上表快照标识 |
| 边界 | `config/xiamen-island.geojson` | relation 14251728 v7 |

### 小样本回归集

`config/regression-sample.json` 冻结 20 个车号共 4,460 点（当日 542,386 点、9,514 车号的 0.82%），来源为 `notebooks/trajectory_analysis.py` 的选取代码，已核实两者输出一致。冻结的理由是该代码依赖 `pandas.DataFrame.sample` 的 RNG 跨版本稳定，而 pandas 不作此保证；清单化后这项依赖消失。

`tests/fixtures/regression-sample-20201221.csv` 与 staging 文件同列同序、保留原始 `source_row`、`BICYCLE_ID` 保留只写在块首行的形态，因此管线代码读 fixture 与读全量走同一条解析路径。

清单里的 `expected_split`（297 条轨迹 / 209 条 ≥2 点 / 176 条 ≥3 点 / 88 条单点 / `BICYCLE_8697` 切出 14 条 / 方向回归用例 `BICYCLE_821_T4` 在样本内）是下一个 ticket（Spark 切分模块）的断言基线；本 ticket 的脚本不计算它，以免切分规则在两处各存一份。

### 已知边界

- 锁文件的 `tooling` 段（Python 与 pandas/numpy/pyproj/shapely/osmium/pyspark 版本、`uv.lock` 的 sha256）在 `--check` 中只作提示输出，不影响退出码：`--check` 的判据是数据漂移，不是环境升级。
- `data/staging/conversion-manifest.json` 仍由 `scripts/convert_raw_excel_to_csv.py` 生成且不进 git；锁文件登记它自身的 sha256 并交叉核对它记录的哈希，两者不一致时以重新计算的结果为准并报错。
