# 管线口径参数写死在代码里，不做成配置文件

`CELL_SIZE_M`、切分四阈值、硬剔除九条的阈值、`markov_time` 等参数，我们选择放进 `src/find_bike_routes/config.py` 的 frozen dataclass 默认值，由 git 承担版本管理，而不是抽成 `config/pipeline.toml` 之类的可编辑配置。CLI 只暴露 `--dates`、`--input`、`--output`、`--run-id` 这类与口径无关的参数。

理由是这些阈值是**口径**而不是**配置**：改动其中任何一个，所有已发布的数字与结论全部作废，因此它们应当像代码一样走 diff 与 review，而不是像配置一样被随手改。配置文件还会制造「有人改了 toml 没改文档」的缝。

## Consequences

- 每次运行仍会把生效的参数序列化进 `artifacts/runs/<run-id>/params.json`，所以「这批数字是用什么参数跑的」依然自证，只是那份 json 是产物而非输入。
- 想跑参数扫描（例如 `markov_time` 网格）时必须在代码里构造参数对象、而非改文件重跑。区域发现阶段本来就要在一次运行内扫多个取值，这与该阶段的形态一致。
