from typing import List,Any,Union
import polars as pl
import pandas as pd
from datetime import datetime
from openbb import obb
import pandas_market_calendars as mcal
import logging
import polars.selectors as cs
from polars import DataFrame
import numpy as np

log = logging.getLogger(__name__)
_DEFAULT_SOURCE='FMP'
_CALENDAR = mcal.get_calendar('NYSE')

def _clean(data):
    sched = _CALENDAR.schedule(*data.select(pl.col('dt').min(),pl.col('dt').max().alias('dt2')).row(0))
    return data.sort('dt') \
    .filter(pl.all_horizontal(pl.all().exclude('dt')).is_null())\
    .filter(pl.col('dt').is_in(sched['market_open'].dt.date))\
    .filter(~pl.col('dt').is_in(_CALENDAR.early_closes(sched)['market_open'].dt.date))\
    .select(*[s for s in data.columns if data.select(pl.col(s).null_count()).item() != data.height])\
    .select('dt', cs.matches("year*").cast(pl.Float32))


def get_treasury_yields(start_date: Union[datetime, Any] = None, end_date: Union[datetime, Any] = None,
                        source: str = _DEFAULT_SOURCE, clean: bool = True) -> DataFrame:
    """
    Fetches treasury yields for the given date range and source, optionally cleaning the data.

    This function retrieves treasury yield data based on the provided start and end dates, source, and
    a flag to clean the data. If no start and end dates are provided, it defaults to a 10-year window
    ending today. The data is converted to a DataFrame for ease of use.

    Args:
        start_date (Union[datetime, Any], optional): The start date of the timeframe for which
            treasury yields are to be fetched. Defaults to the beginning of the 10th previous year.
        end_date (Union[datetime, Any], optional): The end date of the timeframe for which
            treasury yields are to be fetched. Defaults to today's date.
        source (str): The data source from which treasury yields are retrieved. Defaults to
            a predefined constant `_DEFAULT_SOURCE`.
        clean (bool): A flag indicating whether to clean the data before returning it. Defaults to True.

    Returns:
        DataFrame: A DataFrame containing the treasury yields for the specified date range and source.

    Raises:
        Exception: Logs an error and returns an empty DataFrame in case of a failure during data retrieval.
    """
    try:
        sd = pd.to_datetime(start_date) if start_date is not None else datetime.today()- pd.tseries.offsets.YearBegin(10)
        ed = pd.to_datetime(end_date) if start_date is not None else datetime.today()
        data = pl.DataFrame(obb.fixedincome.government.treasury_rates(f"{sd:%Y-%m-%d}",f"{ed:%Y-%m-%d}",source=source).results).rename({'date':'dt'})
        if clean:
            data = _clean(data)
        return data
    except Exception as e:
        log.error(f'Error fetching treasury yields: {e}')
        return pl.DataFrame()


_TEST_DURATIONS={2:1.94,5:4.53,7:6.11,10:8.24,20:13.55,30:17.05}

def bond_metrics(rate, coupon, term, periods_per_year):
    # Number of periods (n)
    n = term * periods_per_year

    # Coupon payment per period
    coupon_payment = coupon / periods_per_year

    # Periodic market interest rate (discount rate)
    period_rate = rate / periods_per_year

    # Time periods for discounting
    time_periods = np.array([i / periods_per_year for i in range(1, n + 1)])

    # Cash flows: Coupons + Face Value at the end
    cash_flows = np.full(n, coupon_payment)
    cash_flows[-1] += 100  # Assume face value is 100 (par value)

    # Discount factor for each period
    discount_factors = (1 + period_rate) ** -time_periods

    # Clean Price Calculation: Present Value of Cash Flows
    clean_price = np.sum(cash_flows * discount_factors)

    # Accrued Interest (AI) calculation: Based on how many periods the bond has accrued interest
    last_coupon_date = term - (time_periods[-1] % (1 / periods_per_year))  # Adjust for last coupon
    accrued_interest = coupon_payment * (time_periods[-1] - last_coupon_date)

    # Dirty Price is Clean Price + Accrued Interest
    dirty_price = clean_price + accrued_interest

    # Duration Calculation: Weighted Average Time
    weights = (cash_flows * discount_factors) / clean_price
    macaulay_duration = np.sum(weights * time_periods)

    # Modified Duration (Duration adjusted for yield)
    modified_duration = macaulay_duration / (1 + period_rate)

    # Convexity Calculation: Second derivative of price with respect to interest rate
    convexity_numerator = np.sum(cash_flows * discount_factors * time_periods * (time_periods + 1))
    convexity = convexity_numerator / (clean_price * (1 + period_rate) ** 2)

    return pd.Series({
        "clean_price": clean_price,
        "dirty_price": dirty_price,
        "macaulay_durationn": macaulay_duration,
        "modified_duration": modified_duration,
        "convexity": convexity,
        'dv01':dirty_price*modified_duration

    })

def add_bond_metrics_to_obb_data(obb_data:pl.DataFrame,notional_mio=100)-> pl.DataFrame:
    melted_frame = obb_data.unpivot(index='dt', variable_name='term', value_name='par_yield') \
        .with_columns(pl.col('term').str.extract(r'(\d+)').cast(pl.Int64)) \
        .filter(pl.col('term').is_in(_TEST_DURATIONS.keys())) \
        .with_columns(
        pl.col('term').replace_strict(_TEST_DURATIONS, default=None).alias('duration'))\
        .with_columns(pl.col('duration').mul(notional_mio*100).alias('dv01'))


    return melted_frame
