# 接收 Python 传来的参数: 1. CSV路径 2. 输出文件夹 3. 算法名称(逗号分隔)
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_dir <- args[2]
algos <- unlist(strsplit(args[3], ","))

# 加载必要库 (这里以通用随机森林和 GLM 为例，你可以替换为 biomod2)
suppressPackageStartupMessages(library(randomForest))
suppressPackageStartupMessages(library(jsonlite))

cat(sprintf("R script started. Data: %s, Algos: %s\n", data_path, args[3]))

# 1. 读取 Python 喂过来的数据
# 数据格式预期: lon, lat, is_presence, sst, chl_a, ...
df <- read.csv(data_path)

# 分离响应变量和环境变量
env_cols <- setdiff(names(df), c("lon", "lat", "is_presence"))
formula_str <- paste("as.factor(is_presence) ~", paste(env_cols, collapse = " + "))

# 2. 训练模型 (演示逻辑，实际请接入 biomod2 框架)
metrics <- list()
model_path <- file.path(out_dir, "sdm_model.rds")

tryCatch({
    # 以随机森林为例
    if ("RF" %in% algos) {
        rf_model <- randomForest(as.formula(formula_str), data = df, ntree=500)
        # 保存模型
        saveRDS(rf_model, file = model_path)
        
        # 提取极简指标 (OOB 错误率)
        metrics[["RF"]] <- list(
            oob_error = mean(rf_model$err.rate[,"OOB"]),
            status = "success"
        )
    }
    
    # 将结果写入 JSON 供 Python 读取
    result_json <- file.path(out_dir, "metrics.json")
    write_json(metrics, result_json, auto_unbox = TRUE)
    cat("SUCCESS\n")
    
}, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    quit(status = 1)
})