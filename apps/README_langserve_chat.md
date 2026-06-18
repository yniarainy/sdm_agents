# LangServe 对话前端

## 1. 启动服务

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn apps.langserve_chat:app --host 0.0.0.0 --port 8000 --reload
```

## 2. 打开前端对话界面

浏览器访问：

- `http://127.0.0.1:8000/chat/playground`
- `http://127.0.0.1:8000/console` (推荐，参数表单 + 对话 + 实时日志)

## 3. 可用对话指令

- `帮助`
- `查看计划参数`
- `开始运行SDM`
- `状态`

## 4. 自然语言自动理解

现在支持自然语言意图识别，不必死记命令。例如：

- `请用2022年的数据给金枪鱼做分布预测，变量用sst和chl_a`
- `帮我跑一次严格GEE模式，不要回退`
- `现在进度如何`

系统会自动判断是否需要启动建模、如何设置参数。

### 推荐体验

1. 打开 `/console`
2. 左侧参数区可直接点“启动建模任务”
3. 右侧对话可自然语言下达任务
4. 下方实时日志自动跟随任务进度

## 5. 实时进度反馈

当启动任务后，会返回任务ID，例如 `job_id`，可通过：

- 任务状态：`GET /jobs/{job_id}`
- 实时日志流(SSE)：`GET /jobs/{job_id}/stream`

也可以在对话框里直接问：`现在进度如何`。

## 6. 说明

- 对话入口基于 LangChain 家族的 `langserve`。
- 执行 `开始运行SDM` 会调用当前 `config.yaml` 执行全流程。
- 产物路径请查看回复内容中的 `html_report` 与 `errors`。
- 存在点数据支持两种入口：上传文件，或不上传时按物种名称从 GBIF/OBIS 自动下载并合并去重。
- 上传文件建议包含 `species_name`、`lon`、`lat`、`year`、`month`、`date`、`source` 列；后端至少要求 `lon`、`lat`、`year`，`date/eventDate` 也可以自动解析出年份。
- 下载入口默认使用 `gbif_obis`，也可在控制台切换为 `gbif`、`obis` 或 `upload`。
- 控制台提供存在点模板下载，模板已包含 `year` 和 `month` 示例字段。
