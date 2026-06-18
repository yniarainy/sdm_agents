from typing import TypedDict, List, Dict, Any, Annotated
import operator
from langchain_core.messages import BaseMessage

class SDMState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    
    # ... [上文的输入、背景点、GEE 提取数据保持不变] ...
    final_dataset: List[Dict[str, Any]] # R 脚本的输入数据
    
    # === 模型训练阶段 (SDM Agent 负责) ===
    target_algorithms: List[str]    # 选用的算法，如 ["GLM", "RF", "MAXENT"]
    model_output_dir: str           # 模型和图表保存的根目录
    
    # 训练完成后的汇报字典 (存入 State 供 Evaluation Agent 读取)
    training_metrics: Dict[str, Any] 
    model_file_path: str            # .rds 或 .RData 文件的路径