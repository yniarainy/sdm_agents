from typing import TypedDict, List, Dict, Any, Annotated
import operator
from langchain_core.messages import BaseMessage

class DataFetchState(TypedDict):
    # LangGraph 必需的聊天记录，使用 operator.add 追加
    messages: Annotated[List[BaseMessage], operator.add]
    
    # ⬇️ 用户的原始输入请求
    user_request: str
    
    # ⬇️ 输入给该智能体的数据
    points: List[Dict[str, float]]  # 例如 [{"lon": 119.5, "lat": 38.0}, ...]
    start_date: str                 # "2023-08-01"
    end_date: str                   # "2023-08-31"
    
    # ⬇️ 工具在后台静默更新的数据区 (大模型不直接阅读海量数据，避免 Token 爆炸)
    extracted_features: Dict[str, List[float]] # 例如 {"sst": [28.5, 29.1, ...], "chl_a": [1.2, 1.5, ...]}
    
    # ⬇️ 最终输出给下一个智能体的合并表 (在内存中)
    final_dataset: List[Dict[str, Any]]