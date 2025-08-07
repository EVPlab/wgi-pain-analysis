library("mediation")
library("foreach")
library("doParallel")

set.seed(2014)



#Load Data
df <- read.csv(file = "/PATH/Longitudinal_Nociplastic_cols.csv")
names(df) <- gsub("\\.", "_", names(df))
df[, setdiff(names(df), "eid")] <- lapply(df[, setdiff(names(df), "eid")], function(col) ifelse(col > 0, 1, col))
targets <- as.list(colnames(df))
items_to_remove <- c("eid")
targets <- targets[!targets %in% items_to_remove]
gi <- read.csv(file = "/PATH/new_GI_probabilities_I.csv")
var_m <- 'Meta_Model_Prob'
gi <- gi[, c('eid',var_m,'Sex')]#
gi <- na.omit(gi)
merged_df <- merge(df, gi, by = "eid")

# Set up the parallel backendR
cl <- makeCluster(detectCores() - 1) 
registerDoParallel(cl)

# Parallelize the loop
results <- foreach(item=targets, .combine=rbind, .packages="mediation") %dopar% {
  df_new <- merged_df[!is.na(merged_df[[item]]), ]
  sex_counts <- tapply(df_new[[item]], df_new$Sex, sum)
  if (any(sex_counts == 0)) {
      cat("Warning: Perfect separation detected for", item, "due to Sex stratification\n")
      return(data.frame(Name=character(), proportion_mediated=numeric(), pval=numeric(), total_effect=numeric(), pval_te=numeric()))
  }
  med.fit <- lm(as.formula(paste(var_m, "~ Sex")), data = df_new)
  out.fit <- glm(as.formula(paste(item, "~", var_m, "+ Sex")), data = df_new, family = binomial)
  med.out <- mediate(med.fit, out.fit, treat = "Sex", mediator = var_m, robustSE = TRUE, sims = 1000)
  summed <- summary(med.out)
  prop_med <- summed$n.avg
  p <- summed$n.avg.p
  te_coef <- summed$tau.coef
  te_p <- summed$tau.p
  
  return(data.frame(Name=item, proportion_mediated=prop_med, pval=p, total_effect=te_coef, pval_te=te_p))
}

stopCluster(cl)

write.csv(results, file = paste("/PATH/T2_change_Mediation_",var_m,".csv"), row.names = FALSE)
