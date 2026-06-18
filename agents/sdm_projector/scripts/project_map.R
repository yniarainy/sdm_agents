args <- commandArgs(trailingOnly = TRUE)
model_path <- args[1]
raster_dir <- args[2]
out_dir <- args[3]

library(terra)
library(randomForest)

# 1. 加载模型
model <- readRDS(model_path)

# 2. 读取 GEE 下载的所有 TIFF 形成一个 Stack
tif_files <- list.files(raster_dir, pattern="\\.tif$", full.names=TRUE)
env_stack <- rast(tif_files)

# 注意：栅格图层的名字必须和训练集 CSV 的列名完全一致！
names(env_stack) <- tools::file_path_sans_ext(basename(tif_files))

# 3. 执行空间预测 (耗时操作)
# type="prob" 输出栖息地适宜度概率 (0~1)
pred_map <- predict(env_stack, model, type="prob")

# 4. 导出最终结果
writeRaster(pred_map, file.path(out_dir, "habitat_suitability_map.tif"), overwrite=TRUE)
cat("SUCCESS")