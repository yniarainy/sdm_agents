import os
import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END

from .state import SDMState
from .tools.r_bridge import execute_sdm_training

load_dotenv(override=True)

def build_sdm_trainer_agent():
    # 1. 实例化 DeepSeek
    llm = init_chat_model(
        model="deepseek-chat", 
        model_provider="deepseek",
        temperature=0.1
    )
    
    tools = [execute_sdm_training]
    
    # 2. 系统指令
    system_prompt = """
    你是一个物种分布模型 (SDM) 训练 Agent。
    你的任务是调用底层的 R 语言引擎来训练模型。
    
    规则：
    1. 检查用户的要求，确定使用什么算法（例如 ["RF"]）。
    2. 调用 execute_sdm_training 工具。你不必传 state 参数。
    3. 工具返回的结果中会包含模型的指标（metrics）。如果发生错误，请根据 r_log 尝试向用户解释错误原因（比如数据量太少，或者全是 NaN）。
    4. 训练成功后，简短总结一下模型表现。
    """
    
    agent_node = create_react_agent(llm, tools, state_modifier=system_prompt)
    
    # 3. State 更新节点：把模型结果写入系统内存
    def update_model_memory(state: SDMState):
        """解析工具返回的 JSON，将关键路径和指标持久化到 State 供下一个 Agent 读取"""
        for message in reversed(state["messages"]):
            if message.type == "tool":
                try:
                    result = json.loads(message.content)
                    if result.get("status") == "success":
                        return {
                            "training_metrics": result.get("metrics"),
                            "model_file_path": os.path.join(result.get("model_dir"), "sdm_model.rds")
                        }
                except:
                    continue
        return {}

    # 4. 图编译
    workflow = StateGraph(SDMState)
    workflow.add_node("llm_agent", agent_node)
    workflow.add_node("state_updater", update_model_memory)
    
    workflow.set_entry_point("llm_agent")
    workflow.add_edge("llm_agent", "state_updater")
    workflow.add_edge("state_updater", END)
    
    return workflow.compile()

sdm_trainer_app = build_sdm_trainer_agent()