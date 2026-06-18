import os
import json
import subprocess
import pandas as pd
from typing import List
from langchain_core.tools import tool

def _get_workspace_dir() -> str:
    """获取一个用于存放 R 脚本中间产物的临时目录"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace = os.path.join(base_dir, "workspace")
    os.makedirs(workspace, exist_ok=True)
    return workspace

@tool
def execute_sdm_training(
    algorithms: List[str],
    state: dict = None
) -> str:
    """
    触发 R 语言后端的 SDM 模型训练。
    
    Args:
        algorithms: 算法列表，目前 R 脚本支持 "RF" (Random Forest), "GLM"。
        
    Returns:
        JSON 格式的模型评估指标或错误日志。
    """
    if state is None or not state.get("final_dataset"):
        return json.dumps({"error": "内存中未找到 final_dataset。必须先运行 GEE 提取智能体！"})
        
    workspace = _get_workspace_dir()
    csv_path = os.path.join(workspace, "temp_training_data.csv")
    metrics_path = os.path.join(workspace, "metrics.json")
    
    # 获取 R 脚本的绝对路径
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "run_biomod2.R")
    
    try:
        # 1. 将 State 内存中的数据集落地为 CSV，供 R 读取
        df = pd.DataFrame(state["final_dataset"])
        # 删除含有 NaN 的行，否则 R 模型会报错
        df = df.dropna()
        df.to_csv(csv_path, index=False)
        
        # 2. 构建子进程命令并唤醒 R
        algos_str = ",".join(algorithms)
        command = ["Rscript", script_path, csv_path, workspace, algos_str]
        
        # 捕获 R 的标准输出和报错
        process = subprocess.run(command, capture_output=True, text=True)
        
        if process.returncode != 0:
            return json.dumps({
                "error": "R 脚本执行失败", 
                "r_log": process.stderr
            }, ensure_ascii=False)
            
        # 3. 训练成功，去读取 R 生成的 metrics.json
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
                
            # 将这些结果包装好反馈给大模型，让大模型决定下一步
            return json.dumps({
                "status": "success",
                "model_dir": workspace,
                "metrics": metrics
            }, ensure_ascii=False)
        else:
            return json.dumps({"error": "R 执行完毕但未生成 metrics.json 报告。"})
            
    except Exception as e:
        return json.dumps({"error": f"Python 调用层异常: {str(e)}"})