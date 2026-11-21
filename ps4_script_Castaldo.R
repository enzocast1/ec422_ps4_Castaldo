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










