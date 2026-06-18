from typing import TypedDict, List, Dict, Any, Annotated
import operator
from langchain_core.messages import BaseMessage

class SDMState(TypedDict):
    # LangGraph 必需的聊天记录
    messages: Annotated[List[BaseMessage], operator.add]
    
    # === 输入阶段 ===
    user_request: str
    presence_points: List[Dict[str, float]]  # 正样本 [{"lon": 119.5, "lat": 38.0}, ...]
    
    # === 背景点生成阶段 (BG Agent 负责) ===
    target_bg_count: int                     # 用户希望生成多少个背景点
    background_points: List[Dict[str, Any]]  # 负样本 [{"lon": 119.6, "lat": 38.1, "is_presence": 0}, ...]
    points: List[Dict[str, Any]]             # 合并后的所有点 (正+负)，交给 GEE Agent 去抽值
    
    # === GEE 提取阶段 (GEE Agent 负责) ===
    start_date: str                 
    end_date: str                   
    extracted_features: Dict[str, List[float]] 
    final_dataset: List[Dict[str, Any]]      # 包含标签和所有环境因子的最终建模宽表