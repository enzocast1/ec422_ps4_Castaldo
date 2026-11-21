library(tidyverse)
library(fpp3)
library(BVAR)

liquor_raw <- read_csv(
  "data/MRTSSM4453USN.csv",
  na = ".",
  col_types = cols(
    observation_date = col_date(format = "%Y-%m-%d"),
    MRTSSM4453USN = col_double()
  )
)

liquor <- liquor_raw |>
  transmute(Month = yearmonth(observation_date), sales = MRTSSM4453USN) |>
  as_tsibble(index = Month) |>
  filter(Month <= yearmonth("2019 Dec"))

has_gaps(liquor)

liquor |>
  autoplot(sales) +
  labs(
    title = "U.S. Retail Sales at Beer, Wine, and Liquor Stores",
    subtitle = "Monthly, not seasonally adjusted",
    x = NULL, y = "Millions of dollars"
  ) +
  theme_minimal()

liquor |>
  gg_season(log(sales)) +
  labs(
    title = "Seasonal Pattern in Log Liquor Store Sales",
    x = "Month", y = "Log(millions of dollars)"
  ) +
  theme_minimal()

# 2.1b: Liquor store sales trend steadily upward from about $1.4 billion a month
# in 1992 to over $6 billion by 2019. There is strong seasonality: sales spike
# every December for the holidays, are lowest in January and February, have a
# smaller rise in the summer months (May-August), and dip in September. The
# seasonal swings grow as the level of sales grows (the December spike is much
# larger in dollars in the 2010s than in the 1990s), which means the
# seasonality is roughly proportional to the level. This suggests using logs:
# on the log scale, proportional swings become constant-size swings, and the
# log season plot confirms this, with each year's line having roughly the same
# shape, just shifted up over time.

liquor_fit <- liquor |>
  model(arima = ARIMA(log(sales)),
        ets   = ETS(log(sales)))

liquor_fit |> select(arima) |> report()
liquor_fit |> select(ets) |> report()

# 2.2a: fable chose ARIMA(1,0,2)(2,1,1)[12] with drift for the ARIMA model and
# ETS(A,A,A) for the ETS model, both fit to log(sales).
#
# ARIMA differencing: d = 0 means no regular (month-to-month) difference was
# taken, and D = 1 means one seasonal difference was taken at lag 12. So the
# model works with log(sales) this month minus log(sales) in the same month
# last year, which is approximately the year-over-year growth rate. Taking that
# difference removes the seasonal pattern and the trend. The drift term
# (constant = 0.0036) allows for steady growth on top of that.
#
# ETS letters: the first A means additive errors, the second A means an
# additive trend (a steady amount of growth each month that is updated over
# time), and the third A means additive seasonality (a fixed monthly bump added
# on top of the level). Because the model is fit to log(sales), additive in
# logs means the trend and seasonal effects are proportional in dollars. The
# smoothing parameters for the trend (beta = 0.008) and season
# (gamma = 0.0002) are close to zero, so the growth rate and seasonal pattern
# barely change over time.

liquor_fit |> select(arima) |> gg_tsresiduals()

liquor_fit |>
  select(arima) |>
  augment() |>
  features(.innov, ljung_box, lag = 24, dof = 6)

# 2.2b: The residuals do not look like white noise. The Ljung-Box test at lag
# 24 (dof = 6) gives a statistic of 120 with a p-value of about 0, so we reject
# the null that the residuals are white noise at the 5% level. There is still
# some autocorrelation the ARIMA model didn't capture, though far less than
# the trend model in PS3.

liquor_cv <- liquor |>
  stretch_tsibble(.init = 120, .step = 1) |>
  filter(.id < max(.id))

liquor_cv_fit <- liquor_cv |>
  model(snaive = SNAIVE(sales),
        ets    = ETS(log(sales)),
        arima  = ARIMA(log(sales)))

saveRDS(liquor_cv_fit, "results/liquor_cv_fit.rds")



#2.3a: 216 forecast origins. The sample has 336 months (Jan 1992 - Dec 2019).
# The first training window is 120 months, so the first origin is December
# 2001,


liquor_cv_fc <- liquor_cv_fit |>
  forecast(h = 1)

liquor_cv_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE))


# 2.3b: The first forecast is for January 2002 (from the training sample
# ending in December 2001) and the last is for December 2019 (from the
# training sample ending in November 2019). The Arima model forecasts the most accurately and both metrics agree. ARIMA forecasts most accurately, with the lowest RMSE (81.4) and MAE (65.7)

liquor_errs <- liquor_cv_fc |>
  as_tibble() |>
  left_join(as_tibble(liquor) |> select(Month, actual = sales),
            by = "Month") |>
  mutate(e = actual - .mean) |>
  select(Month, .model, e) |>
  pivot_wider(names_from = .model, values_from = e)

forecast::dm.test(liquor_errs$arima, liquor_errs$ets, h = 1, power = 2)

forecast::dm.test(liquor_errs$arima, liquor_errs$snaive, h = 1, power = 2)


# 2.4: ARIMA vs. ETS: DM = -2.906, p-value = 0.004. ARIMA vs. SNAIVE:
# DM = -10.755, p-value < 2.2e-16 (essentially 0). Both DM statistics are
# negative, meaning ARIMA had smaller squared forecast errors, and both
# p-values are well below 0.05, so we reject the null of equal forecast
# accuracy in each case: ARIMA is significantly more accurate than both ETS
# and the seasonal random walk benchmark. If I were the liquor store chain,
# I would use the ARIMA model to forecast next month's sales.



liquor_combo_fc <- liquor_cv_fit |>
  mutate(combo = (arima + ets) / 2) |>
  forecast(h = 1)

liquor_combo_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE))

# 3a: The equal-weight combination of ARIMA and ETS has an RMSE of 85.2 and
# an MAE of 66.8. It is more accurate than ETS (94.4 / 72.3) but slightly less
# accurate than ARIMA (81.4 / 65.7), and much better than the seasonal random
# walk (161 / 137). The combo lands between its two components but much closer
# to ARIMA, especially on MAE (66.8 vs. 65.7), so averaging in the weaker ETS
# forecast cost only a little accuracy.

cor(liquor_errs$arima, liquor_errs$ets)
# 3b: The correlation between the ARIMA and ETS forecast errors is 0.88,
# which is very high: the two models tend to make similar mistakes, so when
# ARIMA misses high, ETS usually misses high too. Because their errors move
# together, averaging the two forecasts cancels out very little error.



qd_raw <- read_csv("data/fredqd_2026-09.csv", show_col_types = FALSE) |>
  slice(-(1:2))

growth <- function(x) 100 * difference(log(x))

qd <- qd_raw |>
  mutate(date = as.Date(sasdate, "%m/%d/%Y")) |>
  transmute(date,
            gdp    = growth(GDPC1),     # real GDP
            cons   = growth(PCECC96),   # real consumption
            inv    = growth(GPDIC1),    # real investment
            emp    = growth(PAYEMS),    # payroll employment
            infl   = growth(GDPCTPI),   # GDP price index
            unrate = difference(UNRATE),     # unemployment rate
            ffr    = difference(FEDFUNDS),   # federal funds rate
            gs10   = difference(GS10)) |>    # 10-year Treasury yield
  filter(date >= as.Date("1960-03-01"), date <= as.Date("2019-12-01"))

y <- qd |> select(-date) |> as.matrix()

nrow(y)
head(y)

N <- nrow(y)
origins <- 120:(N - 1)
actual <- y[121:N, ]

# 4.1c: We produce 120 forecasts for each variable. The first training sample
# is 1960 Q1 - 1989 Q4 (120 quarters), so the first forecast is for 1990 Q1,
# and the last origin is 2019 Q3, so the last forecast is for 2019 Q4.

var_ols_fc <- function(z, p) {
  e <- embed(z, p + 1)            # columns: y_t, then its lags 1 to p
  M <- ncol(z)                    # number of variables
  B <- coef(lm(e[, 1:M] ~ e[, -(1:M)]))
  x <- c(1, t(z[nrow(z):(nrow(z) - p + 1), ]))   # constant, latest p lags
  x %*% B
}

fc_var1 <- NULL
fc_var4 <- NULL
for (T in origins) {
  fc_var1 <- rbind(fc_var1, var_ols_fc(y[1:T, ], 1))
  fc_var4 <- rbind(fc_var4, var_ols_fc(y[1:T, ], 4))
}

rmse_by_var <- function(f) sqrt(colMeans((actual - f)^2))

rmse_by_var(fc_var1)
rmse_by_var(fc_var4)

#the most recent quarter contains most of the predictive information, and the small amount of extra information in older lags is outweighed by the estimation error from the additional coefficients.

bvar_cv <- function(lambda) {
  pr <- bv_priors(hyper = "lambda",
                  mn = bv_mn(lambda = bv_lambda(mode = lambda, sd = 0.4,
                                                min = 0.999 * lambda,
                                                max = 1.001 * lambda),
                             b = 0))
  fc <- NULL
  for (T in origins) {
    fit_T <- bvar(y[1:T, ], lags = 4, priors = pr,
                  n_draw = 2000, n_burn = 1000, verbose = FALSE)
    x <- c(1, t(y[T:(T - 3), ]))         # constant and 4 lags of each variable
    fc <- rbind(fc, x %*% coef(fit_T))   # forecast of y_{T+1}
  }
  fc
}

set.seed(422)
fc_bvar_03 <- bvar_cv(0.3)
saveRDS(fc_bvar_03, "results/fc_bvar_03.rds")


set.seed(422)
fc_bvar_005 <- bvar_cv(0.05)

var_results <- tibble(
  Variable = colnames(y),
  `VAR(1)` = rmse_by_var(fc_var1),
  `VAR(4)` = rmse_by_var(fc_var4),
  `BVAR(4), lambda = 0.3` = rmse_by_var(fc_bvar_03),
  `BVAR(4), lambda = 0.05` = rmse_by_var(fc_bvar_005)
)

saveRDS(var_results, "results/var_results.rds")

var_results |> knitr::kable(digits = 3)


# 4.3: The BVAR(4) with lambda = 0.3 forecasts more accurately than the OLS
# VAR(4) for all eight variables. Since both models have the same variables
# and lags, the difference comes entirely from the Minnesota prior: shrinking
# the coefficients toward zero reduces the estimation error that comes from
# estimating 33 coefficients per equation with limited data. The BVAR(4) also
# beats the VAR(1) for investment, inflation, and the 10-year yield, though
# the VAR(1) is still best for GDP, consumption, employment, and unemployment.
#
# With lambda = 0.05, forecasts get worse for most variables, especially
# inflation (0.197 to 0.330) and employment (0.211 to 0.315). A small lambda
# means very heavy shrinkage, so the prior is trusted too much: the
# coefficients are pushed so close to zero that the model ignores real
# persistence in the data. The fed funds rate is the exception, where heavy
# shrinkage gives the best forecasts of any model. Too little shrinkage
# (OLS VAR(4)) leaves too much estimation error, while too much shrinkage
# (lambda = 0.05) makes the model too rigid; lambda = 0.3 strikes a better
# balance.


# 4.3c: DM tests, OLS VAR(4) vs. BVAR(4) with lambda = 0.3 (squared error loss)

# GDP growth (column 1)
e_var4 <- actual[, "gdp"] - fc_var4[, 1]
e_bvar <- actual[, "gdp"] - fc_bvar_03[, 1]

forecast::dm.test(e_var4, e_bvar, h = 1, power = 2)

# Federal funds rate (column 7)
e_var4_ffr <- actual[, "ffr"] - fc_var4[, 7]
e_bvar_ffr <- actual[, "ffr"] - fc_bvar_03[, 7]

forecast::dm.test(e_var4_ffr, e_bvar_ffr, h = 1, power = 2)

# 4.3c: For GDP growth, DM = 2.137 with p-value = 0.035, and for the federal
# funds rate, DM = 2.000 with p-value = 0.048. Both p-values are below 0.05,
# so we reject the null of equal forecast accuracy in each case. The DM
# statistics are positive (VAR(4) listed first), meaning the OLS VAR(4) had
# larger squared errors, so the BVAR(4) with lambda = 0.3 is significantly
# more accurate for both variables. However, both p-values are only just
# below 0.05, so the evidence is real but not overwhelming. This suggests the
# improvement from shrinkage in 4.3b is genuine but modest, and that the
# smaller RMSE differences in the 4.3b table should not be over-interpreted,
# since many of them would likely not be statistically significant.



md_raw <- read_csv("data/fredmd_2026-09.csv", show_col_types = FALSE)
tcode <- unlist(md_raw[1, -1])
md <- md_raw[-1, ] |> mutate(date = as.Date(sasdate, "%m/%d/%Y"))

transform_series <- function(x, code) {
  d <- function(v) c(NA, diff(v))
  switch(as.character(code),
         "1" = x,                      # level
         "2" = d(x),                   # first difference
         "3" = d(d(x)),                # second difference
         "4" = log(x),                 # log
         "5" = 100 * d(log(x)),        # growth rate
         "6" = 100 * d(d(log(x))),     # change in the growth rate
         "7" = d(x / dplyr::lag(x) - 1))
}


fm <- map2_dfc(md[names(tcode)], tcode, transform_series) |>
  mutate(date = md$date, .before = 1) |>
  filter(date >= as.Date("1960-03-01"), date <= as.Date("2019-12-01"))

fm <- fm[, colSums(is.na(fm)) == 0]

dim(fm)

# 5.1c: Transformation codes for four series
tcode[c("INDPRO", "PAYEMS", "CPIAUCSL", "FEDFUNDS")]


# 5.1c: INDPRO and PAYEMS (code 5) are transformed to growth rates, 100 times
# the first difference of the log. CPIAUCSL (code 6) is transformed to the
# change in its growth rate, i.e., the change in inflation. FEDFUNDS (code 2)
# is first differenced, giving the change in the rate in percentage points.
# FEDFUNDS differs from INDPRO because it is already a rate measured in
# percent: it does not grow exponentially, so a percent growth rate is not
# meaningful (and log growth would be unstable when the rate is near zero, as
# in 2009-2015), while INDPRO is an index level that grows over time, so its
# percent growth rate is the natural measure.


# 5.2a: Estimate the factors by principal components ---------------------
Z <- fm |> select(-date) |> as.matrix()

pc <- prcomp(scale(Z))

summary(pc)$importance[, 1:5]

# 5.2a: The first factor explains 15.0% of the variation in the 121 series,
# the second 7.4%, and the third 7.1%. Together, the first three factors
# explain about 29.5% of the variation, matching the 29% we saw in class.


# 5.2b: Plot the first factor --------------------------------------------
F1 <- pc$x[, 1] * sign(cor(pc$x[, 1], fm$INDPRO))

tibble(date = fm$date, F1 = F1) |>
  ggplot(aes(date, F1)) +
  geom_line() +
  labs(title = "First Factor from FRED-MD", x = NULL, y = NULL) +
  theme_minimal()


# 5.2c: Series most correlated with the first two factors ----------------
sort(abs(cor(pc$x[, 1], Z))[1, ], decreasing = TRUE)[1:8]
sort(abs(cor(pc$x[, 2], Z))[1, ], decreasing = TRUE)[1:8]

# 5.2c: Factor 1 is most correlated with payroll employment (PAYEMS, 0.84),
# industrial production (INDPRO and its manufacturing and final-products
# components), goods-producing and manufacturing employment, and
# manufacturing capacity utilization. Together with the plot, which shows
# sharp drops in every recession, Factor 1 captures real economic activity:
# the business cycle in production and employment.
# Factor 2 is most correlated with consumer and PCE price measures (CPI for
# all items, commodities, transportation, and CPI excluding shelter, food,
# or medical care, plus the PCE price index). Since these series are
# transformed to the change in inflation, Factor 2 is an inflation factor,
# capturing broad changes in inflation across many price measures.

# 5.3: Replicate the industrial production horse race --------------------
ip <- fm$INDPRO
n <- nrow(Z)
init <- which(fm$date == as.Date("1989-12-01"))

e_ar <- NULL
e_fa <- NULL
for (T in init:(n - 1)) {
  F_T <- prcomp(scale(Z[1:T, ]))$x[, 1:3]         # factors using data through T
  d <- data.frame(y_next = ip[2:T], y = ip[1:(T - 1)], F_T[1:(T - 1), ])
  new <- data.frame(y = ip[T], t(F_T[T, ]))
  e_ar <- c(e_ar, ip[T + 1] - predict(lm(y_next ~ y, d), new))
  e_fa <- c(e_fa, ip[T + 1] - predict(lm(y_next ~ ., d), new))
}

c(RMSE_AR = sqrt(mean(e_ar^2)), RMSE_FA = sqrt(mean(e_fa^2)))

# 5.3: RMSE is 0.617 for the AR(1) and 0.589 with three factors, matching
# class (about a 4.6% reduction). Estimating the factors once on the full
# sample would be cheating because it uses data from after each forecast
# origin, information a real forecaster wouldn't have had. This look-ahead
# bias would make the factor model look better than it really is.


# 5.4: Direct 12-month-ahead forecast of CPI inflation -------------------
direct_race <- function(series, h = 12, r = 3) {
  lv <- md |>
    mutate(log_x = log(.data[[series]]),
           g = 100 * (log_x - lag(log_x))) |>
    filter(date >= as.Date("1960-03-01"), date <= as.Date("2019-12-01"))
  yh <- 100 * (lead(lv$log_x, h) - lv$log_x)      # growth over the next h months
  g  <- lv$g                                      # growth over the latest month
  e_ar <- NULL
  e_fa <- NULL
  for (T in init:(n - h)) {
    F_T <- prcomp(scale(Z[1:T, ]))$x[, 1:r, drop = FALSE]
    i <- 1:(T - h)
    d <- data.frame(yh = yh[i], g = g[i], F_T[i, , drop = FALSE])
    new <- data.frame(g = g[T], F_T[T, , drop = FALSE])
    e_ar <- c(e_ar, yh[T] - predict(lm(yh ~ g, d), new))
    e_fa <- c(e_fa, yh[T] - predict(lm(yh ~ ., d), new))
  }
  tibble(series, forecasts = length(e_ar),
         RMSE_AR = sqrt(mean(e_ar^2)), RMSE_FA = sqrt(mean(e_fa^2)),
         MAE_AR = mean(abs(e_ar)), MAE_FA = mean(abs(e_fa)),
         DM_pvalue = forecast::dm.test(e_ar, e_fa, h = h, power = 2,
                                       varestimator = "bartlett")$p.value)
}

cpi_race <- direct_race("CPIAUCSL")
cpi_race


# 5.4: Yes, the factors improve the 12-month CPI inflation forecast. RMSE falls
# from 2.11 (AR) to 1.89 (with factors), about a 10% reduction. MAE also falls,
# from 1.58 to 1.35. The DM p-value is 0.0004, so the difference is statistically
# significant even after adjusting for the overlapping 12-month errors (Bartlett).

