import os
import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END

# 导入模块
from .state import SDMState
from .tools.spatial_tools import generate_random_pseudo_absences

load_dotenv(override=True)

def build_bg_generator_agent():
    # 1. 实例化 DeepSeek 语言模型
    llm = init_chat_model(
        model="deepseek-chat", 
        model_provider="deepseek",
        temperature=0  # 保持 0，我们需要稳定的代码调用
    )
    
    # 2. 绑定工具
    tools = [generate_random_pseudo_absences]
    
    # 3. 系统提示词
    system_prompt = """
    你是一个 SDM (物种分布模型) 背景点生成专家 Agent。
    用户的 State 内存中已经提供了 `presence_points`（物种出现点）。
    你的任务是：
    1. 根据用户的要求（如需要多少个背景点），调用 generate_random_pseudo_absences 工具。
    2. 如果用户没有指定数量，默认生成与 presence_points 数量相等的背景点 (1:1)。
    3. 获取工具返回的成功信息后，总结并结束任务。
    """
    
    agent_node = create_react_agent(llm, tools, state_modifier=system_prompt)
    
    # 4. 后处理节点：真正把生成的数据更新到内存的 State 里
    def update_state_memory(state: SDMState):
        """解析大模型对话中最后一条工具调用的返回值，把数据写入 State"""
        # 遍历最近的消息，找到我们工具生成的 JSON 字符串
        for message in reversed(state["messages"]):
            if message.type == "tool":
                try:
                    result = json.loads(message.content)
                    if result.get("status") == "success":
                        # 将生成的背景点和合并后的点集，注入到主流水线的内存中
                        return {
                            "background_points": result["data"]["background_points"],
                            "points": result["data"]["points"] # 这个将直接喂给 GEE Agent！
                        }
                except:
                    continue
        return {}

    # 5. 组装子图
    workflow = StateGraph(SDMState)
    workflow.add_node("llm_agent", agent_node)
    workflow.add_node("state_updater", update_state_memory)
    
    workflow.set_entry_point("llm_agent")
    workflow.add_edge("llm_agent", "state_updater")
    workflow.add_edge("state_updater", END)
    
    return workflow.compile()

bg_agent_app = build_bg_generator_agent()