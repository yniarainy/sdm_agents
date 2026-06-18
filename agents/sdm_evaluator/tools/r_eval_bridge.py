import os
import json
import subprocess
from langchain_core.tools import tool

@tool
def evaluate_trained_model(state: dict = None) -> str:
    """调用 R 脚本计算 AUC、TSS，并生成变量重要性和响应曲线。"""
    workspace = state.get("model_output_dir", "./workspace")
    model_path = os.path.join(workspace, "sdm_model.rds")
    data_path = os.path.join(workspace, "temp_training_data.csv")
    
    if not os.path.exists(model_path):
        return json.dumps({"error": "未找到训练好的模型文件，请先执行训练。"})

    # 假设我们有一个 evaluate.R 脚本
    script_path = os.path.join(os.path.dirname(__file__), "../scripts/evaluate.R")
    
    try:
        # R 脚本会读取模型，进行十折交叉验证，并输出图表
        process = subprocess.run(["Rscript", script_path, model_path, data_path, workspace], capture_output=True, text=True)
        
        if process.returncode != 0:
            return json.dumps({"error": "评估失败", "log": process.stderr})
            
        # 读取 R 输出的评估结果
        with open(os.path.join(workspace, "eval_results.json"), "r") as f:
            metrics = json.load(f)
            
        return json.dumps({
            "status": "success",
            "metrics": metrics,
            "plots": f"{workspace}/response_curves.png"
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": str(e)})