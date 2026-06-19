#!/usr/bin/env Rscript
# ───────────────────────────────────────────────────────────
# biomod2 ensemble SDM training
# Called by r_bridge.py with:
#   args[1] = CSV path (lon, lat, is_presence, env_factors...)
#   args[2] = output directory
#   args[3] = algorithms (comma-separated: RF,GLM,GBM,GAM,MAXENT,ANN)
# ───────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop("Usage: Rscript run_biomod2.R <csv_path> <out_dir> <algorithms>")
}

data_path  <- args[1]
out_dir    <- args[2]
algos_str  <- args[3]
algos      <- unlist(strsplit(algos_str, ","))

# ── Load libraries ─────────────────────────────────────────
suppressPackageStartupMessages({
  library(biomod2)
  library(jsonlite)
  library(randomForest)  # biomod2 backend
})

cat(sprintf("[biomod2] Data: %s\n[biomod2] Output: %s\n[biomod2] Algos: %s\n",
            data_path, out_dir, algos_str))

# ── Read data ──────────────────────────────────────────────
df <- read.csv(data_path, stringsAsFactors = FALSE)

# Identify columns
env_cols_all <- setdiff(names(df), c("lon", "lat", "is_presence", "species_name",
                                      "year", "month", "date", "source", "occurrence_id"))
if (length(env_cols_all) == 0) {
  stop("No environmental factor columns found in CSV")
}

# Remove rows with NA in env columns
df <- df[complete.cases(df[, env_cols_all]), ]
cat(sprintf("[biomod2] %d rows after removing NAs\n", nrow(df)))

if (nrow(df) < 10) {
  stop("Too few rows after NA removal (< 10)")
}

# ── Prepare biomod2 data ───────────────────────────────────
# biomod2 expects:
#  - resp.var: vector of 0/1 or NA for true absences
#  - expl.var: data.frame of environmental variables
#  - resp.xy: matrix of coordinates (optional)

my_resp   <- df$is_presence
my_expl   <- df[, env_cols_all, drop = FALSE]
my_xy     <- as.matrix(df[, c("lon", "lat")])
my_resp_name <- "species"

# Handle pseudo-absences (0 = pseudo-absence, 1 = presence, NA = true absence)
# biomod2 treats NA as true absence; we use 0 for random pseudo-absences
my_resp[my_resp == 0] <- NA  # Convert to NA for biomod2 PA treatment

cat(sprintf("[biomod2] %d presences, %d pseudo-absences\n",
            sum(my_resp == 1, na.rm = TRUE), sum(is.na(my_resp))))

# ── Format data ────────────────────────────────────────────
biomod_data <- tryCatch({
  BIOMOD_FormatingData(
    resp.var     = my_resp,
    expl.var     = my_expl,
    resp.xy      = my_xy,
    resp.name    = my_resp_name,
    PA.nb.rep    = 3,          # 3 PA sets
    PA.nb.absences = min(1000, sum(is.na(my_resp))),
    PA.strategy  = "random"
  )
}, error = function(e) {
  cat(sprintf("[biomod2] FormatingData failed: %s\n", e$message))
  quit(status = 1)
})

# ── Configure models ───────────────────────────────────────
# Map our algorithm names to biomod2 model names
algo_map <- list(
  rf    = "RF",
  xgb   = "GBM",    # XGBoost ~ GBM in biomod2
  lgbm  = "GBM",
  gbm   = "GBM",
  glm   = "GLM",
  gam   = "GAM",
  maxent = "MAXENT",
  ann   = "ANN"
)

bio_algo <- unique(unlist(algo_map[tolower(algos)]))
bio_algo <- bio_algo[!is.na(bio_algo)]

if (length(bio_algo) == 0) {
  bio_algo <- c("RF", "GLM")  # fallback
}

cat(sprintf("[biomod2] Running algorithms: %s\n", paste(bio_algo, collapse = ", ")))

# ── Modeling options ────────────────────────────────────────
modeling_opts <- BIOMOD_ModelingOptions()

# ── Run models ─────────────────────────────────────────────
biomod_models <- tryCatch({
  BIOMOD_Modeling(
    bm.formated      = biomod_data,
    modeling.id      = "sdm_agents",
    models           = bio_algo,
    models.options   = modeling_opts,
    NbRunEval        = 3,
    DataSplit        = 80,
    VarImport        = 3,
    models.eval.meth = c("TSS", "ROC", "KAPPA"),
    SaveObj          = TRUE,
    rescal.all.models = TRUE,
    do.full.models   = FALSE,
    modeling.id      = paste0("sdm_agents_", format(Sys.time(), "%Y%m%d%H%M%S"))
  )
}, error = function(e) {
  cat(sprintf("[biomod2] Modeling failed: %s\n", e$message))
  # Fall back to simple RF + GLM
  biomod_models <- BIOMOD_Modeling(
    bm.formated      = biomod_data,
    modeling.id      = "sdm_agents_fallback",
    models           = c("RF", "GLM"),
    models.options   = modeling_opts,
    NbRunEval        = 1,
    DataSplit        = 80,
    VarImport        = 0,
    models.eval.meth = c("ROC"),
    SaveObj          = TRUE,
    rescal.all.models = TRUE,
    do.full.models   = FALSE,
    modeling.id      = paste0("sdm_fallback_", format(Sys.time(), "%Y%m%d%H%M%S"))
  )
})

# ── Build ensemble ─────────────────────────────────────────
biomod_ensemble <- tryCatch({
  BIOMOD_EnsembleModeling(
    bm.mod          = biomod_models,
    em.by           = "all",
    em.select       = "TSS",
    em.select.thres = 0.3,
    em.metric       = c("TSS", "ROC"),
    em.metric.select.thres = 0.3,
    em.algo         = c("EMmean", "EMwmean", "EMca"),  # mean, weighted-mean, committee-averaging
    SaveObj         = TRUE
  )
}, error = function(e) {
  cat(sprintf("[biomod2] Ensemble failed: %s\n", e$message))
  NULL
})

# ── Extract metrics ────────────────────────────────────────
eval_scores <- get_evaluations(biomod_models)
eval_df <- as.data.frame(eval_scores)

# Build JSON output
model_metrics <- list()
for (algo_name in names(eval_scores)) {
  algo_eval <- eval_scores[[algo_name]]
  model_metrics[[algo_name]] <- list(
    roc  = if ("ROC" %in% rownames(algo_eval)) mean(algo_eval["ROC", ], na.rm = TRUE) else NA,
    tss  = if ("TSS" %in% rownames(algo_eval)) mean(algo_eval["TSS", ], na.rm = TRUE) else NA,
    kappa = if ("KAPPA" %in% rownames(algo_eval)) mean(algo_eval["KAPPA", ], na.rm = TRUE) else NA
  )
}

# Variable importance
var_imp <- tryCatch({
  vi <- get_variables_importance(biomod_models)
  as.data.frame(vi)
}, error = function(e) NULL)

var_imp_list <- list()
if (!is.null(var_imp)) {
  for (v in rownames(var_imp)) {
    var_imp_list[[v]] <- as.list(colMeans(var_imp[v, , drop = FALSE], na.rm = TRUE))
  }
}

# Ensemble metrics
ensemble_metrics <- list()
if (!is.null(biomod_ensemble)) {
  ens_eval <- tryCatch(get_evaluations(biomod_ensemble), error = function(e) NULL)
  if (!is.null(ens_eval)) {
    for (ens_name in names(ens_eval)) {
      ens_e <- ens_eval[[ens_name]]
      ensemble_metrics[[ens_name]] <- list(
        roc  = if ("ROC" %in% rownames(ens_e)) mean(ens_e["ROC", ], na.rm = TRUE) else NA,
        tss  = if ("TSS" %in% rownames(ens_e)) mean(ens_e["TSS", ], na.rm = TRUE) else NA
      )
    }
  }
}

output <- list(
  status          = "success",
  algorithms_run  = bio_algo,
  individual      = model_metrics,
  variable_importance = var_imp_list,
  ensemble        = ensemble_metrics,
  ensemble_models = if (!is.null(biomod_ensemble)) names(biomod_ensemble) else list()
)

# ── Save results ────────────────────────────────────────────
write_json(output, file.path(out_dir, "biomod2_metrics.json"), auto_unbox = TRUE, pretty = TRUE)
cat(sprintf("[biomod2] SUCCESS — metrics saved to %s/biomod2_metrics.json\n", out_dir))
