#!/usr/bin/env Rscript
# Descriptive REML sensitivity on curated exact assay cells, not a causal partition.
suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})
args <- commandArgs(trailingOnly=TRUE)
scope <- if (length(args)) args[1] else "sample"
curation_only <- "--curation-only" %in% args
source <- file.path("data", "curated", scope, "all_exact_assay_cells.csv")
out <- file.path("results", "reconstruction_2026-10-20", scope, "statistics")
dir.create(out, recursive=TRUE, showWarnings=FALSE)
d <- read.csv(source, stringsAsFactors=FALSE, na.strings="")
required <- c("log2_mic","chemical_id","publication_id","target_exact","medium")
d <- d[complete.cases(d[,required]), ]
for (key in required[-1]) d[[key]] <- factor(d[[key]])
counts <- aggregate(publication_id ~ chemical_id, d, function(x) length(unique(x)))
crossed_ids <- as.character(counts$chemical_id[counts$publication_id > 1])
exact_crossed <- aggregate(publication_id ~ chemical_id + target_exact, d,
                          function(x) length(unique(x)))
pair_key <- function(x) paste(x$chemical_id, x$target_exact, sep="||")
matched_keys <- pair_key(exact_crossed[exact_crossed$publication_id > 1,])
control <- lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=100000))
fit_one <- function(data, interaction=FALSE, fixed_medium=FALSE) {
  data <- droplevels(data)
  terms <- "(1|chemical_id) + (1|publication_id) + (1|target_exact)"
  if (interaction) terms <- paste(terms, "+ (1|chemical_id:target_exact) + (1|chemical_id:medium)")
  fixed <- if (fixed_medium && nlevels(data$medium)>1) "medium" else "1"
  form <- as.formula(paste("log2_mic ~", fixed, "+", terms))
  warning_messages <- character()
  fit <- withCallingHandlers(lmer(form, data=data, REML=TRUE, control=control),
                             warning=function(w) {
                               warning_messages <<- c(warning_messages, conditionMessage(w))
                               invokeRestart("muffleWarning")
                             })
  vc <- as.data.frame(VarCorr(fit))
  vc$fraction <- vc$vcov / sum(vc$vcov)
  list(n_used=nobs(fit), n_requested=nrow(data),
       chemical_entities=length(unique(data$chemical_id)),
       publications=length(unique(data$publication_id)),
       exact_targets=length(unique(data$target_exact)),
       formula=deparse(form), variance_components=vc,
       singular=isSingular(fit), warnings=warning_messages,
       optimizer_messages=fit@optinfo$conv$lme4$messages,
       note="Descriptive exact-label sensitivity; no component is labeled causal laboratory variance.")
}
tasks <- list(
  all_additive=list(data=d, interaction=FALSE, fixed_medium=FALSE),
  all_interactions=list(data=d, interaction=TRUE, fixed_medium=TRUE),
  multi_publication_additive=list(data=d[as.character(d$chemical_id) %in% crossed_ids,],
                                 interaction=FALSE, fixed_medium=FALSE),
  multi_publication_interactions=list(data=d[as.character(d$chemical_id) %in% crossed_ids,],
                                     interaction=TRUE, fixed_medium=TRUE),
  cross_publication_same_exact_target=list(data=d[pair_key(d) %in% matched_keys,],
                                         interaction=FALSE, fixed_medium=TRUE)
)
results <- list()
for (name in if (curation_only) character() else names(tasks)) {
  cat("Fitting", name, "\n")
  results[[name]] <- tryCatch(do.call(fit_one, tasks[[name]]),
                             error=function(e) list(error=conditionMessage(e)))
  write_json(results, file.path(out,"reml_sensitivity.json"), auto_unbox=TRUE,
             pretty=TRUE, null="null", digits=10)
}

# These exploratory restrictions retain the primary results above unchanged.
# They assess dependence on conversion, incomplete context, tails and support.
membership_path <- file.path(out, "curation_sensitivity_membership.csv")
if (file.exists(membership_path)) {
  membership <- read.csv(membership_path, stringsAsFactors=FALSE)
  membership <- membership[match(d$assay_cell_id, membership$assay_cell_id),]
  stopifnot(identical(as.character(membership$assay_cell_id), as.character(d$assay_cell_id)))
  sensitivity <- list()
  for (name in setdiff(names(membership), "assay_cell_id")) {
    selected <- tolower(as.character(membership[[name]])) == "true"
    cat("Fitting curation sensitivity", name, "\n")
    sensitivity[[name]] <- tryCatch(
      fit_one(d[selected,], interaction=FALSE, fixed_medium=FALSE),
      error=function(e) list(n_requested=sum(selected), error=conditionMessage(e)))
    write_json(sensitivity, file.path(out,"reml_curation_sensitivity.json"), auto_unbox=TRUE,
               pretty=TRUE, null="null", digits=10)
  }
}
