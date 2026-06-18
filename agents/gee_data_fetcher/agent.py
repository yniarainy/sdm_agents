import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END

# 导入同级目录下的模块
from .state import DataFetchState
from .tools.gee_tools import search_marine_datasets, extract_and_save_data

# 加载环境变量 (确保根目录下的 .env 文件中配置了 DEEPSEEK_API_KEY)
load_dotenv(override=True)

def build_gee_data_agent():
    # ==========================================
    # 1. 实例化语言模型 (使用 DeepSeek)
    # ==========================================
    # init_chat_model 会自动读取系统环境变量中的 DEEPSEEK_API_KEY
    llm = init_chat_model(
        model="deepseek-chat", 
        model_provider="deepseek",
        temperature=0  # 强制设为 0，确保稳定精准地调用 GEE 工具
    )
    
    # ==========================================
    # 2. 绑定工具与提示词
    # ==========================================
    tools = [search_marine_datasets, extract_and_save_data]
    
    system_prompt = """
    你是一个专业的数据采集 Agent。
    用户的 State 内存中已经准备好了 `points`, `start_date`, `end_date`。
    你的任务是：根据用户的文字需求，调用工具完成所有因子的提取。
    
    执行步骤：
    1. 调用 search_marine_datasets 查询对应的 factor 代码。
    2. 依次调用 extract_and_save_data。你不需要传入 state 参数，系统会自动注入。
    3. 当所有要求的数据都提示“提取成功”后，向用户回复“数据采集完毕”。
    """
    
    # 3. 使用 LangGraph 预置框架构建 Agent 节点
    agent_node = create_react_agent(llm, tools, state_modifier=system_prompt)
    
    # ==========================================
    # 4. 自定义后处理节点：整合特征数据
    # ==========================================
    def finalize_dataset(state: DataFetchState):
        """当 LLM 提取完所有数据后，执行矩阵合并"""
        points = state.get("points", [])
        extracted = state.get("extracted_features", {})
        
        final_data = []
        for i, point in enumerate(points):
            row = {"lon": point["lon"], "lat": point["lat"]}
            # 遍历内存中提取完毕的所有因子列表
            for factor, values_list in extracted.items():
                if i < len(values_list):
                    row[factor] = values_list[i]
            final_data.append(row)
            
        return {"final_dataset": final_data}

    # ==========================================
    # 5. 组装最终状态图 (StateGraph)
    # ==========================================
    workflow = StateGraph(DataFetchState)
    
    # 添加节点
    workflow.add_node("llm_agent", agent_node)
    workflow.add_node("data_merger", finalize_dataset)
    
    # 定义流向
    workflow.set_entry_point("llm_agent")
    workflow.add_edge("llm_agent", "data_merger")
    workflow.add_edge("data_merger", END)
    
    # 编译图并返回
    return workflow.compile()

# 实例化应用，供主程序直接引入
gee_agent_app = build_gee_data_agent()