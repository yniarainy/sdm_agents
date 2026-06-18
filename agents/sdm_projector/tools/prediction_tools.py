import os
import ee
import json
import subprocess
from langchain_core.tools import tool

# 1. GEE 栅格下载工具 (为预测准备底图)
@tool
def download_projection_rasters(bbox: list, target_date: str, state: dict = None) -> str:
    """
    根据模型训练时使用的环境因子，从 GEE 批量下载对应的 GeoTIFF 栅格数据，用于空间预测。
    """
    workspace = state.get("model_output_dir", "./workspace")
    raster_dir = os.path.join(workspace, "projection_rasters")
    os.makedirs(raster_dir, exist_ok=True)
    
    # 获取训练时到底用了哪些因子 (比如 ["sst", "chl_a"])
    trained_factors = [k for k in state["final_dataset"][0].keys() if k not in ["lon", "lat", "is_presence"]]
    
    roi = ee.Geometry.Rectangle(bbox)
    downloaded_files = []
    
    try:
        # 这里需要调用你之前的 datasets.json 字典 (省略读取逻辑)
        registry = _get_registry() 
        
        for factor in trained_factors:
            config = registry[factor]
            # 获取那一天的均值图
            img = ee.ImageCollection(config["id"]).filterDate(target_date, target_date+"-28").select(config["band"]).mean()
            img = _apply_math(img, config)
            
            out_file = os.path.join(raster_dir, f"{factor}.tif")
            # 实际生产中建议用 ee.batch.Export，如果是小区域可以用 geemap 直接下
            # geemap.ee_export_image(img.clip(roi), filename=out_file, scale=5000)
            downloaded_files.append(out_file)
            
        return json.dumps({"status": "success", "raster_dir": raster_dir, "files": downloaded_files})
    except Exception as e:
        return json.dumps({"error": str(e)})

# 2. R 语言空间预测工具
@tool
def run_spatial_projection(state: dict = None) -> str:
    """调用 R 的 terra 包，将模型应用到下载好的 TIFF 栅格上，生成预测图。"""
    workspace = state.get("model_output_dir", "./workspace")
    raster_dir = os.path.join(workspace, "projection_rasters")
    model_path = os.path.join(workspace, "sdm_model.rds")
    
    script_path = os.path.join(os.path.dirname(__file__), "../scripts/project_map.R")
    
    try:
        process = subprocess.run(["Rscript", script_path, model_path, raster_dir, workspace], capture_output=True, text=True)
        if process.returncode != 0:
            return json.dumps({"error": process.stderr})
            
        return json.dumps({
            "status": "success", 
            "prediction_map": f"{workspace}/habitat_suitability_map.tif"
        })
    except Exception as e:
        return json.dumps({"error": str(e)})