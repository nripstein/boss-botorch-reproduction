# Persistent R evaluator for BOSS Simulation 3 (released as simulation2_change_point).

.boss_change_point_state <- new.env(parent = emptyenv())

boss_initialize_change_point <- function(data_path,
                                         bounds = c(0.1, 9.9),
                                         prior_bounds = c(0, 10),
                                         basis_k = 30L,
                                         inner_aghq_k = 3L,
                                         seed = 123L,
                                         release_faithful = FALSE) {
  if (!requireNamespace("BayesGP", quietly = TRUE)) {
    stop("The R package 'BayesGP' is required.")
  }
  if (!requireNamespace("aghq", quietly = TRUE)) {
    stop("The R package 'aghq' is required.")
  }
  if (length(bounds) != 2L || bounds[1] >= bounds[2]) {
    stop("bounds must be an increasing numeric vector of length two")
  }
  if (length(prior_bounds) != 2L || prior_bounds[1] >= prior_bounds[2]) {
    stop("prior_bounds must be an increasing numeric vector of length two")
  }
  if (!file.exists(data_path)) {
    stop("change-point data file does not exist: ", data_path)
  }

  suppressPackageStartupMessages(library(BayesGP))
  if (requireNamespace("data.table", quietly = TRUE)) {
    data.table::setDTthreads(1L)
  }
  loaded <- new.env(parent = emptyenv())
  object_names <- load(data_path, envir = loaded)
  if (!"data" %in% object_names || !is.data.frame(loaded$data)) {
    stop("data_path must be an .rda file containing a data.frame named 'data'")
  }
  if (!all(c("x", "y") %in% names(loaded$data))) {
    stop("change-point data must contain columns 'x' and 'y'")
  }

  state <- .boss_change_point_state
  state$data <- loaded$data
  state$bounds <- as.numeric(bounds)
  state$prior_bounds <- as.numeric(prior_bounds)
  state$basis_k <- as.integer(basis_k)
  state$inner_aghq_k <- as.integer(inner_aghq_k)
  state$seed <- as.integer(seed)
  state$release_faithful <- isTRUE(release_faithful)
  state$evaluation_count <- 0L
  state$initialization_count <- if (exists("initialization_count", state, inherits = FALSE)) {
    state$initialization_count + 1L
  } else {
    1L
  }
  state$dll_path <- getLoadedDLLs()[["BayesGP"]][["path"]]
  set.seed(state$seed)
  invisible(TRUE)
}

.boss_fit_change_point <- function(alpha, sample_count, aghq_k = .boss_change_point_state$inner_aghq_k) {
  state <- .boss_change_point_state
  if (!exists("data", state, inherits = FALSE)) {
    stop("boss_initialize_change_point() must be called before evaluation")
  }
  alpha <- as.numeric(alpha)
  if (length(alpha) != 1L || !is.finite(alpha)) {
    stop("alpha must be one finite numeric value")
  }
  if (alpha < state$bounds[1] || alpha > state$bounds[2]) {
    stop(sprintf("alpha %.17g is outside evaluator bounds [%.17g, %.17g]",
                 alpha, state$bounds[1], state$bounds[2]))
  }

  fit_data <- state$data
  fit_data$x1 <- ifelse(fit_data$x <= alpha, fit_data$x, alpha)
  fit_data$x2 <- ifelse(fit_data$x > alpha, fit_data$x - alpha, 0)
  if (isTRUE(state$release_faithful)) {
    # Deliberately mirrors 04_boss.R: k, M, and envir are omitted.
    return(BayesGP::model_fit(
      formula = y ~
        f(x1, model = "IWP", order = 2,
          sd.prior = list(param = 1, h = 1), initial_location = 0) +
        f(x2, model = "IWP", order = 2,
          sd.prior = list(param = 1, h = 1), initial_location = 0),
      data = fit_data, method = "aghq", family = "Gaussian",
      aghq_k = aghq_k
    ))
  }
  BayesGP::model_fit(
    formula = y ~
      f(x1, model = "IWP", order = 2, k = state$basis_k,
        sd.prior = list(param = 1, h = 1), initial_location = 0) +
      f(x2, model = "IWP", order = 2, k = state$basis_k,
        sd.prior = list(param = 1, h = 1), initial_location = 0),
    data = fit_data,
    method = "aghq",
    family = "Gaussian",
    aghq_k = aghq_k,
    M = as.integer(sample_count),
    envir = environment()
  )
}

boss_evaluate_change_point <- function(alpha, aghq_k = .boss_change_point_state$inner_aghq_k, report_warnings = FALSE) {
  state <- .boss_change_point_state
  start <- proc.time()[["elapsed"]]
  # The live profile must preserve release warnings and default M=3000.
  fit <- if (isTRUE(state$release_faithful)) {
    .boss_fit_change_point(alpha, sample_count = NULL, aghq_k = aghq_k)
  } else if (report_warnings) {
    .boss_fit_change_point(alpha, sample_count = 2L, aghq_k = aghq_k)
  } else suppressWarnings(.boss_fit_change_point(alpha, sample_count = 2L, aghq_k = aghq_k))
  log_marginal <- as.numeric(fit$mod$normalized_posterior$lognormconst)
  alpha <- as.numeric(alpha)
  if (alpha < state$prior_bounds[1] || alpha > state$prior_bounds[2]) {
    log_prior <- -Inf
  } else {
    log_prior <- -log(diff(state$prior_bounds))
  }
  objective <- log_marginal + log_prior
  if (!all(is.finite(c(log_marginal, log_prior, objective)))) {
    stop("nonfinite change-point objective at alpha=", format(alpha, digits = 17))
  }
  state$evaluation_count <- state$evaluation_count + 1L
  elapsed <- proc.time()[["elapsed"]] - start
  result <- structure(
    c(log_marginal = log_marginal, log_prior = log_prior, objective = objective),
    elapsed_seconds = elapsed
  )
  rm(fit)
  invisible(gc(verbose = FALSE))
  result
}

boss_change_point_diagnostics <- function() {
  state <- .boss_change_point_state
  list(
    initialization_count = state$initialization_count,
    evaluation_count = state$evaluation_count,
    dll_path = state$dll_path,
    dll_loaded = "BayesGP" %in% names(getLoadedDLLs()),
    bounds = state$bounds,
    prior_bounds = state$prior_bounds,
    basis_k = state$basis_k,
    inner_aghq_k = state$inner_aghq_k,
    release_faithful = state$release_faithful
  )
}

