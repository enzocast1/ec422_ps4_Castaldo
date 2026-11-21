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

There’s still some autocorrelation the ARIMA model didn’t capture.


