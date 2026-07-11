from __future__ import annotations

import json
import os
import io
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langserve import add_routes

from agents.orchestrator import SDMOrchestrator
from agents.orchestrator.agent_graph import SDMAgentGraph
from agents.orchestrator.occurrence_tools import normalize_presence_dataframe

app = FastAPI(title="SDM LangServe Chat", version="1.0.0")
load_dotenv(override=True)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _build_llm():
    if not os.getenv("DEEPSEEK_API_KEY"):
        return None
    try:
        return init_chat_model(
            model="deepseek-chat",
            model_provider="deepseek",
            temperature=0.3,
        )
    except Exception:
        return None


LLM = _build_llm()
JOBS: Dict[str, Dict[str, Any]] = {}
LATEST_JOB_ID: str | None = None
JOB_LOCK = threading.Lock()
LAST_UPLOAD: Dict[str, Any] | None = None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _save_presence_points(df, original_name: str) -> Dict[str, Any]:
    safe_name = Path(original_name).stem or "presence_points"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{safe_name}_{timestamp}.csv"
    file_path = UPLOAD_DIR / file_name
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    return {
        "path": str(file_path),
        "file_name": file_name,
        "rows": int(len(df)),
    }


def _presence_points_template_csv() -> str:
    return (
        "species_name,lon,lat,year,month,date,source\n"
        "example_species,120.123,30.456,2023,6,2023-06-01,field_survey\n"
        "example_species,120.345,30.678,2023,6,2023-06-10,field_survey\n"
    )


def _to_messages(raw: Any) -> List[BaseMessage]:
    if isinstance(raw, dict):
        if "messages" in raw:
            raw = raw.get("messages")
        elif "input" in raw and isinstance(raw.get("input"), dict):
            raw = raw["input"].get("messages", [])
        else:
            raw = []

    if not isinstance(raw, list):
        return []

    out: List[BaseMessage] = []
    for item in raw:
        if isinstance(item, BaseMessage):
            out.append(item)
            continue
        if isinstance(item, dict):
            role = str(item.get("type", item.get("role", ""))).lower()
            content = str(item.get("content", ""))
            if role in {"human", "user"}:
                out.append(HumanMessage(content=content))
            elif role in {"ai", "assistant"}:
                out.append(AIMessage(content=content))
    return out


def _latest_human(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content.strip()
    return ""


def _decide_action(messages: List[BaseMessage], text: str) -> Dict[str, Any]:
    if LLM is None:
        lower = text.lower()
        if any(k in lower for k in ["开始", "运行", "建模", "预测"]):
            return {"intent": "run_model", "config_overrides": {}}
        if any(k in lower for k in ["进度", "状态", "日志"]):
            return {"intent": "check_status", "config_overrides": {}}
        return {"intent": "chat", "config_overrides": {}}

    prompt = (
        "你是 SDM 任务调度智能体。根据用户输入判定意图并返回 JSON，不要输出多余文本。"
        "JSON 格式: {\"intent\":\"run_model|check_status|show_plan|chat\",\"config_overrides\":{...},\"reply\":\"...\"}."
        "如果用户希望开始建模、预测、训练，intent=run_model。"
        "若用户询问进度或状态，intent=check_status。"
        "若用户问当前参数配置，intent=show_plan。"
        "run_model 时可提取字段: species_name,presence_points_path,presence_source_mode,occurrence_download_limit,start_date,end_date,map_resolution,use_gee,strict_gee,factors(required list)."
    )
    try:
        result = LLM.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=text),
        ])
        content = getattr(result, "content", "") or "{}"
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            return json.loads(content[start : end + 1])
    except Exception:
        pass
    return {"intent": "chat", "config_overrides": {}}


def _append_job_log(job_id: str, message: str) -> None:
    with JOB_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["logs"].append(f"[{_now()}] {message}")


def _run_job(job_id: str, overrides: Dict[str, Any]) -> None:
    _append_job_log(job_id, "任务已启动")
    with JOB_LOCK:
        JOBS[job_id]["status"] = "running"
    try:
        graph = SDMAgentGraph(
            config_path="config.yaml",
            interactive=False,
            plan_overrides=overrides,
            enable_llm=False,
        )
        result = graph.run()
        with JOB_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["finished_at"] = _now()
            JOBS[job_id]["artifacts"] = result.get("artifacts", {})
            JOBS[job_id]["step_status"] = result.get("step_status", {})
            JOBS[job_id]["errors"] = result.get("error_events", [])
        _append_job_log(job_id, "任务已完成")
    except Exception as exc:
        with JOB_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["finished_at"] = _now()
            JOBS[job_id]["error"] = str(exc)
        _append_job_log(job_id, f"任务失败: {exc}")


def _start_job(overrides: Dict[str, Any]) -> str:
    global LATEST_JOB_ID
    job_id = str(uuid.uuid4())
    with JOB_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "created_at": _now(),
            "finished_at": None,
            "overrides": overrides,
            "logs": [],
            "artifacts": {},
            "step_status": {},
            "errors": [],
            "error": None,
        }
        LATEST_JOB_ID = job_id

    t = threading.Thread(target=_run_job, args=(job_id, overrides), daemon=True)
    t.start()
    return job_id


def _job_summary(job_id: str) -> str:
    with JOB_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return "未找到任务。"
    return (
        f"任务ID: {job_id}\n"
        f"状态: {job.get('status')}\n"
        f"创建时间: {job.get('created_at')}\n"
        f"结束时间: {job.get('finished_at') or '进行中'}\n"
        f"最近日志条数: {len(job.get('logs', []))}"
    )


def _handle_chat(payload: Any) -> AIMessage:
    messages = _to_messages(payload)
    text = _latest_human(messages)
    lower = text.lower()

    if not text:
        return AIMessage(content="请输入你的需求，例如：'开始运行SDM'、'查看计划参数'、'先用GBIF和OBIS下载存在点'。")

    if any(k in lower for k in ["help", "帮助", "说明"]):
        return AIMessage(
            content=(
                "可用指令:\n"
                "1) '开始运行SDM'：按 config.yaml 启动一次完整流程\n"
                "2) '查看计划参数'：展示当前默认配置\n"
                "3) '状态'：提示你查看最近 run_summary.json 和 errors.json\n"
                "4) 如果没有上传文件，会按物种名自动从 GBIF/OBIS 下载并合并存在点"
            )
        )

    if "查看计划参数" in text or "计划" in text:
        orchestrator = SDMOrchestrator(config_path="config.yaml", interactive=False)
        plan = orchestrator._build_plan()  # noqa: SLF001
        return AIMessage(content=f"当前默认计划:\n{json.dumps(plan.__dict__, ensure_ascii=False, indent=2)}")

    decision = _decide_action(messages, text)
    intent = str(decision.get("intent", "chat")).lower()
    overrides = decision.get("config_overrides", {})
    if not isinstance(overrides, dict):
        overrides = {}

    if intent == "show_plan":
        orchestrator = SDMOrchestrator(config_path="config.yaml", interactive=False)
        plan = orchestrator._build_plan()  # noqa: SLF001
        return AIMessage(content=f"当前默认计划:\n{json.dumps(plan.__dict__, ensure_ascii=False, indent=2)}")

    if intent == "check_status" or any(k in lower for k in ["进度", "状态", "日志"]):
        if not LATEST_JOB_ID:
            return AIMessage(content="当前没有运行中的任务。你可以直接描述需求，我会自动判断是否开始建模。")
        return AIMessage(content=_job_summary(LATEST_JOB_ID))

    if intent == "run_model":
        job_id = _start_job(overrides)
        return AIMessage(
            content=(
                f"我已根据你的需求启动建模任务。\n"
                f"任务ID: {job_id}\n"
                f"实时进度: /jobs/{job_id}/stream\n"
                f"状态查询: /jobs/{job_id}\n"
                f"你也可以直接在对话里问：'现在进度如何？'"
            )
        )

    if LLM is None:
        return AIMessage(content="已收到。当前未配置可用大模型，请先设置 DEEPSEEK_API_KEY，或输入 '帮助' 使用系统命令。")

    try:
        result = LLM.invoke(messages)
        content = getattr(result, "content", "") or "已收到。"
        return AIMessage(content=str(content))
    except Exception as exc:
        return AIMessage(content=f"对话模型调用失败: {exc}")


chat_runnable = RunnableLambda(_handle_chat).with_types(
    input_type=Dict[str, Any],
    output_type=AIMessage,
)

add_routes(app, chat_runnable, path="/chat", playground_type="chat")


@app.get("/")
def health() -> dict:
    return {
        "status": "ok",
        "message": "Open /chat/playground for chat UI",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/console")
def console() -> FileResponse:
    html_file = STATIC_DIR / "dashboard.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="dashboard not found")
    return FileResponse(str(html_file))


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    with JOB_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/jobs/latest")
def get_latest_job() -> Dict[str, Any]:
    if not LATEST_JOB_ID:
        return {"status": "none", "job": None}
    with JOB_LOCK:
        job = JOBS.get(LATEST_JOB_ID)
    return {"status": "ok", "job": job}


@app.post("/jobs/start")
def start_job(overrides: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(overrides, dict):
        raise HTTPException(status_code=400, detail="overrides must be an object")
    job_id = _start_job(overrides)
    return {
        "job_id": job_id,
        "status": "queued",
        "stream": f"/jobs/{job_id}/stream",
        "detail": f"/jobs/{job_id}",
    }


@app.post("/uploads/presence-points")
async def upload_presence_points(file: UploadFile = File(...)) -> Dict[str, Any]:
    filename = file.filename or "presence_points"
    ext = Path(filename).suffix.lower()

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="空文件")

    import pandas as pd

    try:
        if ext in {".csv", ".txt", ".tsv"}:
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(io.BytesIO(raw), sep=sep)
            normalized = normalize_presence_dataframe(df, species_name="uploaded_species", source_label="upload")
        elif ext in {".geojson", ".json"}:
            geojson = json.loads(raw.decode("utf-8"))
            features = geojson.get("features", []) if isinstance(geojson, dict) else []
            rows = []
            for feature in features:
                geom = feature.get("geometry", {}) if isinstance(feature, dict) else {}
                coords = geom.get("coordinates") if isinstance(geom, dict) else None
                if not coords or len(coords) < 2:
                    continue
                props = feature.get("properties", {}) if isinstance(feature, dict) else {}
                row = dict(props)
                row["lon"] = coords[0]
                row["lat"] = coords[1]
                rows.append(row)
            if not rows:
                raise ValueError("GeoJSON 中未找到有效点要素")
            df = pd.DataFrame(rows)
            normalized = normalize_presence_dataframe(df, species_name="uploaded_species", source_label="upload")
        else:
            raise HTTPException(status_code=400, detail="仅支持 CSV/TSV/GeoJSON")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"解析失败: {exc}") from exc

    saved = _save_presence_points(normalized, filename)
    global LAST_UPLOAD
    LAST_UPLOAD = {
        "kind": "presence_points",
        "path": saved["path"],
        "file_name": saved["file_name"],
        "rows": saved["rows"],
        "uploaded_at": _now(),
    }
    preview = normalized.head(5).to_dict(orient="records")
    return {
        "status": "ok",
        "presence_points_path": saved["path"],
        "rows": saved["rows"],
        "columns": list(normalized.columns),
        "preview": preview,
        "message": "上传成功，可在启动任务时直接使用该路径。",
    }


@app.get("/uploads/latest")
def latest_upload() -> Dict[str, Any]:
    if LAST_UPLOAD is None:
        return {"status": "empty"}
        return {"status": "ok", "upload": LAST_UPLOAD}


@app.get("/uploads/presence-template")
def download_presence_template() -> Response:
    return Response(
        content=_presence_points_template_csv(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=presence_points_template.csv"},
    )


@app.get("/jobs/{job_id}/stream")
def stream_job(job_id: str) -> StreamingResponse:
    with JOB_LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="job not found")

    def event_generator():
        cursor = 0
        while True:
            with JOB_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    yield "event: error\ndata: job-not-found\n\n"
                    return
                logs = job.get("logs", [])
                status = job.get("status")

            while cursor < len(logs):
                payload = json.dumps({"log": logs[cursor]}, ensure_ascii=False)
                yield f"event: log\ndata: {payload}\n\n"
                cursor += 1

            if status in {"completed", "failed"}:
                payload = json.dumps({"status": status}, ensure_ascii=False)
                yield f"event: done\ndata: {payload}\n\n"
                return

            time.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
