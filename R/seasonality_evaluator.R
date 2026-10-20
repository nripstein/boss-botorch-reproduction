# Persistent R evaluator for the released Simulation 1 seasonality example.

.boss_seasonality_state <- new.env(parent = emptyenv())

boss_generate_seasonality_data <- function(seed = 123L) {
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(as.integer(seed))
  n <- 100L
  x <- runif(n, 0, 10)
  log_mu <- 1 + 0.5*cos(2*pi*x/1.5) - 1.3*sin(2*pi*x/1.5) +
    1.1*cos(4*pi*x/1.5) + 0.3*sin(4*pi*x/1.5) + rnorm(n, sd = 2)
  data.frame(y = rpois(n, exp(log_mu)), x = x, indx = seq_len(n), log_mu = log_mu)
}

boss_initialize_seasonality <- function(data_path = NULL, bounds = c(0.5, 4.5), seed = 123L) {
  if (!requireNamespace("BayesGP", quietly = TRUE)) {
    stop("BayesGP is required for the seasonality evaluator")
  }
  # ``model_fit`` parses the formula's unqualified ``f(...)`` term.
  # Attaching is therefore intentional and mirrors the released scripts.
  suppressPackageStartupMessages(library(BayesGP))
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(as.integer(seed))
  if (is.null(data_path)) {
    data <- boss_generate_seasonality_data(seed)
  } else {
  data_path <- normalizePath(data_path, mustWork = FALSE)
  if (!file.exists(data_path)) stop("seasonality data file does not exist: ", data_path)
  loaded <- new.env(parent = emptyenv())
  load(data_path, envir = loaded)
  if (!exists("data", envir = loaded, inherits = FALSE)) {
    stop("seasonality data file must contain object 'data'")
  }
  data <- get("data", envir = loaded)
  }
  if (!is.data.frame(data) || !all(c("y", "x", "indx") %in% names(data))) {
    stop("seasonality data must contain columns 'y', 'x', and 'indx'")
  }
  if (length(bounds) != 2L || !is.finite(bounds[1]) || !is.finite(bounds[2]) || bounds[1] >= bounds[2]) {
    stop("seasonality bounds must be an increasing finite pair")
  }
  state <- .boss_seasonality_state
  state$data <- data
  state$bounds <- as.numeric(bounds)
  state$seed <- as.integer(seed)
  state$initialization_count <- if (is.null(state$initialization_count)) 1L else state$initialization_count + 1L
  state$evaluation_count <- 0L
  state$prior_audit <- NULL
  invisible(NULL)
}

.boss_fit_seasonality <- function(alpha, aghq_k = 4L) {
  state <- .boss_seasonality_state
  if (is.null(state$data)) stop("boss_initialize_seasonality() must be called before evaluation")
  alpha <- as.numeric(alpha)
  if (length(alpha) != 1L || !is.finite(alpha) || alpha < state$bounds[1] || alpha > state$bounds[2]) {
    stop("seasonality alpha must be finite and lie within configured bounds")
  }
  data <- state$data
  frequency <- (2 * pi) / alpha
  data$cosx <- cos(frequency * data$x)
  data$sinx <- sin(frequency * data$x)
  data$cos2x <- cos(2 * frequency * data$x)
  data$sin2x <- sin(2 * frequency * data$x)
  # BayesGP draws conditional samples internally. Resetting before each fit
  # makes evaluation and checkpoint resume independent of earlier RNG use.
  set.seed(state$seed)
  fit <- model_fit(
    formula = y ~ cosx + sinx + cos2x + sin2x + f(x = indx, model = "IID", sd.prior = list(param = 1)),
    data = data, method = "aghq", family = "Poisson", aghq_k = aghq_k
  )
  if (is.null(state$prior_audit)) {
    fixed <- fit$control.fixed
    sd <- fit$instances[[1]]@sd.prior
    stopifnot(length(fixed) == 5L,
              all(vapply(fixed, function(p) p$mean == 0 && p$prec == 0.001, logical(1))),
              sd$prior == "exp", sd$param$u == 1, sd$param$alpha == 0.5)
    state$prior_audit <- list(fixed = fixed, sd = sd,
                             sd_rate = -log(sd$param$alpha)/sd$param$u,
                             BayesGP_version = as.character(packageVersion("BayesGP")),
                             RNGkind = RNGkind())
  }
  log_marginal <- as.numeric(fit$mod$normalized_posterior$lognormconst)
  log_prior <- dnorm(alpha, mean = 3, sd = 0.5, log = TRUE)
  list(log_marginal = log_marginal, log_prior = log_prior, objective = log_marginal + log_prior)
}

boss_evaluate_seasonality <- function(alpha, aghq_k = 4L) {
  result <- .boss_fit_seasonality(alpha, aghq_k = aghq_k)
  state <- .boss_seasonality_state
  state$evaluation_count <- state$evaluation_count + 1L
  if (!is.finite(result$objective)) stop("nonfinite seasonality objective")
  unlist(result, use.names = TRUE)
}

.boss_allocate_seasonality_draws <- function(weights, total_draws) {
  raw <- weights * total_draws
  allocated <- floor(raw)
  remainder <- as.integer(total_draws - sum(allocated))
  if (remainder > 0L) {
    order_index <- order(raw - allocated, decreasing = TRUE, method = "radix")
    allocated[order_index[seq_len(remainder)]] <- allocated[order_index[seq_len(remainder)]] + 1L
  }
  as.integer(allocated)
}

