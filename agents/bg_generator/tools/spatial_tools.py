import json
import random
from typing import List, Dict
from langchain_core.tools import tool

@tool
def generate_random_pseudo_absences(
    presence_points: List[Dict[str, float]], 
    num_points: int,
    state: dict = None # 用于直接写入内存
) -> str:
    """
    根据给定的物种存在点(Presence)，计算其空间边界(Bounding Box)，
    并在该边界内随机生成指定数量的伪不存在点/背景点(Pseudo-absences)。
    
    Args:
        presence_points: 物种存在的经纬度字典列表。
        num_points: 需要生成的背景点数量。
        
    Returns:
        执行结果的文字汇报。
    """
    if not presence_points:
        return "错误：未提供存在点数据，无法生成背景点。"

    try:
        # 1. 计算存在点的包围盒 (Bounding Box)，并向外扩张 10% 作为研究区
        lons = [p["lon"] for p in presence_points]
        lats = [p["lat"] for p in presence_points]
        
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # 扩展缓冲区
        lon_buffer = (max_lon - min_lon) * 0.1 if max_lon > min_lon else 1.0
        lat_buffer = (max_lat - min_lat) * 0.1 if max_lat > min_lat else 1.0
        
        min_lon -= lon_buffer
        max_lon += lon_buffer
        min_lat -= lat_buffer
        max_lat += lat_buffer

        # 将现有存在点转换为快速查找集合，防止背景点与存在点绝对重合
        presence_set = {(round(p["lon"], 4), round(p["lat"], 4)) for p in presence_points}
        
        background_points = []
        attempts = 0
        max_attempts = num_points * 10 # 防止死循环
        
        # 2. 随机生成伪不存在点
        while len(background_points) < num_points and attempts < max_attempts:
            attempts += 1
            rand_lon = round(random.uniform(min_lon, max_lon), 4)
            rand_lat = round(random.uniform(min_lat, max_lat), 4)
            
            # 简单粗暴的剔除逻辑：不能和存在点完全重合 (实际科研中这里会用距离 Buffer 过滤)
            if (rand_lon, rand_lat) not in presence_set:
                background_points.append({
                    "lon": rand_lon, 
                    "lat": rand_lat, 
                    "is_presence": 0  # 核心标签：0代表伪不存在
                })

        # 为存在点也打上标签 1
        labeled_presences = [{"lon": p["lon"], "lat": p["lat"], "is_presence": 1} for p in presence_points]
        
        # 将合并后的数据挂载到返回信息的 JSON 字符串中，由后续节点存入 State
        # (因为有些 LangGraph 版本直接修改注入的 state 不会触发更新，最好通过节点显式返回)
        output_data = {
            "background_points": background_points,
            "points": labeled_presences + background_points
        }
        
        return json.dumps({
            "status": "success",
            "message": f"成功在包围盒 [{min_lon:.2f}, {min_lat:.2f}, {max_lon:.2f}, {max_lat:.2f}] 内生成了 {len(background_points)} 个背景点。",
            "data": output_data
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)