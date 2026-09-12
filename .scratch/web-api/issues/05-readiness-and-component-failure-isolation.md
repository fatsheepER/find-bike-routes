# 05: 报告完整就绪状态并隔离组件故障

**What to build:** 项目开发者可以通过 `GET /api/health` 判断当前 API 是否同时连接了正确的 MobilityDB 发布版、本岛边界和同一冻结划分的序列产物。一个组件故障只阻断依赖它的业务接口，健康响应提供可操作的组件状态但不泄露本机配置。

**Blocked by:** 02: 返回完整的区域间流动切片；03: 返回可复核的频繁区域序列；04: 按区域或矩形查询有效轨迹

**Status:** done

- [x] 健康检查验证数据库连接、PostGIS 与 MobilityDB 扩展、恰好一行的 `dataset_release`、可读且为 WGS84 Polygon 或 MultiPolygon 的本岛边界、两张可读的序列产物及冻结划分摘要一致性
- [x] 全部组件正常时返回 200 和 `status=ok`；任一组件异常时返回 503 和 `status=unavailable`
- [x] 响应分别列出 `database`、`extensions`、`dataset_release`、`island_boundary`、`sequences` 和 `digest_match` 状态，可返回扩展版本和摘要，但不返回 DSN、密码、本机路径、SQL 或异常堆栈
- [x] 序列产物失败时，区域、流和轨迹接口继续使用数据库；数据库失败时，区域、流、轨迹和需要核对摘要的序列接口各自返回 503
- [x] 全部接口的 422、404、500 和 503 响应都不泄露 DSN、密码、SQL、本机路径或堆栈；合法空数据保持 200，不伪装成依赖故障
- [x] 不增加单独的 liveness、readiness 或诊断路由，也不增加后台探测、重试框架或通用依赖注入层
- [x] `tests/test_api_health.py` 从 HTTP 边界覆盖全部正常、数据库不可用、缺扩展、缺发布身份、多行发布身份、边界缺失或无效、序列缺失、摘要不一致和双向故障隔离
- [x] 本票运行 `tests/test_api_regions.py`、`tests/test_api_flows.py`、`tests/test_api_sequences.py`、`tests/test_api_tracks.py`、`tests/test_api_health.py` 及直接受影响的既有测试
- [x] 所有实现票完成后，在本 feature 边界使用显式测试 DSN 运行一次完整 pytest；没有 DSN 时不自动启动 Docker，并在 Comments 中区分已运行、跳过和未覆盖的验证范围

## Comments

- Implementation: added `GET /api/health` as the only probe route. It reports safe component states for the database connection, PostGIS and MobilityDB versions, the single release identity, the RFC 7946 WGS84 island boundary, the readable default sequence products, and the frozen-partition digest match. It reuses the existing boundary and sequence readers, keeps business endpoint dependencies isolated, and adds no background probing, retries, cache, or dependency-injection layer. The flow hour query now uses FastAPI's native integer range so a valid `hour=6` reaches the database failure path instead of being rejected as 422.
- Review: the first standards review identified a single-use hour Enum, which was replaced with the native query constraint. The first spec reviews found that an explicit non-WGS84 `crs`, safe 422/404/500 coverage, and a missing default sequence ruler needed handling; each received an HTTP regression and implementation correction. Final standards and spec reviews found no remaining actionable issue.
- Test: the final ticket-named API run passed 47 tests and skipped 36. The final feature-boundary `uv run pytest -q` passed 568 tests and skipped 47 in 703.45 seconds. Ruff, mypy, compileall, `uv lock --check`, and `git diff --check` passed. `MOBILITYDB_TEST_DSN` was not set, so the real MobilityDB success, missing-extension, release-cardinality, and database-backed isolation cases were skipped; Docker was not started. A real published database HTTP performance run was not covered.
