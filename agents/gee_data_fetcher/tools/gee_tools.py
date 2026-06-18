import os
import json
import ee
from typing import List, Dict
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.runnables import RunnableConfig

# 初始化 GEE (同上文)
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

def _get_registry() -> dict:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "references", "datasets.json"), "r", encoding="utf-8") as f:
        return json.load(f)

# 数学转换辅助函数 (同上文 _apply_math)
def _apply_math(image: ee.Image, config: dict) -> ee.Image:
    img = image
    if "scale_factor" in config:
        img = img.multiply(config["scale_factor"])
    if "offset" in config:
        img = img.add(config["offset"])
    return ee.Image(img.copyProperties(image, ['system:time_start']))

@tool
def search_marine_datasets(query: str = "") -> str:
    """查询支持的海洋环境因子字典。不确定 factor 代码时调用此工具。"""
    registry = _get_registry()
    if query:
        return json.dumps({k: v for k, v in registry.items() if query.lower() in v['description'].lower() or query.lower() in k.lower()}, ensure_ascii=False)
    return json.dumps(registry, ensure_ascii=False)

@tool
def extract_and_save_data(
    factor: str, 
    start_date: str, 
    end_date: str, 
    # 魔法参数：LangGraph 会自动把当前的 State 注入进来
    state: dict = None 
) -> str:
    """
    提取环境数据并保存到内存中。
    Args:
        factor: 环境因子代码，如 "sst"。
        start_date: 开始日期 "YYYY-MM-DD"。
        end_date: 结束日期 "YYYY-MM-DD"。
    Returns:
        执行结果状态字符串。
    """
    if state is None or "points" not in state:
        return "错误：内存中未找到输入坐标点 (points)。"
        
    points = state["points"]
    registry = _get_registry()
    
    if factor.lower() not in registry:
        return f"错误：找不到因子 '{factor}'。请先调用 search_marine_datasets 确认。"
        
    config = registry[factor.lower()]
    
    try:
        features = [ee.Feature(ee.Geometry.Point([p["lon"], p["lat"]]), {"id": i}) for i, p in enumerate(points)]
        fc = ee.FeatureCollection(features)
        
        if config.get("is_static", False):
            image = ee.Image(config["id"]).select(config["band"])
        else:
            collection = ee.ImageCollection(config["id"]).filterDate(start_date, end_date).select(config["band"])
            if collection.size().getInfo() == 0:
                return f"错误：{factor} 在所选时间内无数据，请尝试更换数据集。"
            image = collection.mean()
            
        image = _apply_math(image, config)
        
        # 批量提取
        sampled_fc = image.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=config.get("scale", 5000))
        results_list = sampled_fc.getInfo()["features"]
        
        # 提取具体的数值列表
        values = [res["properties"].get('first', res["properties"].get(config["band"], None)) for res in results_list]
        
        # 返回简短报告给 LLM，防止刷屏
        return f"成功提取了 {len(values)} 个点的 {factor} 数据，并已放入系统内存。你可以继续提取下一个因子。"
        
    except Exception as e:
        return f"GEE 提取异常: {str(e)}"