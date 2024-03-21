# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of the License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

from datetime import datetime
from dotenv import load_dotenv
from time import sleep

import gradio as gr
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import spaces
import torch

from alpaca.data import (
    CryptoHistoricalDataClient,
    StockHistoricalDataClient,
    StockLatestQuoteRequest
)
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetAssetsRequest
from chronos import ChronosPipeline
from plotly.subplots import make_subplots

# from indicators.smabb import SmaBB
# from indicators.rsi import RSI

load_dotenv()

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_API_SECRET")

stock_historical_data_client = StockHistoricalDataClient(
    api_key=api_key,
    secret_key=secret_key
)

class Security:
    """
    A class representing a financial security for fetching and plotting historical data.

    Attributes
    ----------
    api_key : str
        The Alpaca API key for authentication.

    secret_key : str
        The Alpaca secret key for authentication.

    symbol : str
        The symbol of the security.

    stock_client : StockHistoricalDataClient
        The Alpaca client for fetching historical stock data.

    dataframe : pl.DataFrame
        DataFrame containing the fetched historical data.
    """
    def __init__(
        self,
        symbol: str
    ) -> None:
        """
        Initialize the Security object with API keys and symbol.

        Parameters
        ----------
        symbol : str
            The symbol of the security.
        """
        self.symbol = symbol
        self.stock_client = stock_historical_data_client
        self.dataframe = None


    def fetch(
        self,
        start: datetime,
        timeframe: TimeFrame=TimeFrame.Day
    ) -> pl.DataFrame:
        """
        Fetches historical data for the security.

        Parameters
        ----------
        start : datetime
            The start date for fetching historical data.

        timeframe : TimeFrame, optional
            The timeframe for fetching historical data. Default is Day.

        Returns
        -------
        pl.DataFrame
            A DataFrame containing the fetched historical data.
        """
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=self.symbol,
                timeframe=timeframe,
                start=start,
            )

            quotes = self.stock_client.get_stock_bars(request_params)
            self.dataframe = pl.from_dicts(quotes[self.symbol])

            return self.dataframe

        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None


def calculate_quantiles(
    forecast_data: np.ndarray,
    quantiles: list = [0.1, 0.5, 0.9],
    axis: int = 0
) -> tuple:
    """
    Calculate quantiles from forecast data.

    Parameters
    ----------
    forecast_data : numpy.ndarray
        Array-like object containing forecast data.

    quantiles : list of float, optional
        List of quantiles to compute. Default is [0.1, 0.5, 0.9].

    axis : int, optional
        Axis along which to compute quantiles. Default is 0.

    Returns
    -------
    tuple
        A tuple containing the calculated quantiles.

    Examples
    --------
    >>> import numpy as np
    >>> forecast_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> low, median, high = calculate_quantiles(forecast_data)
    """
    return np.quantile(forecast_data, quantiles, axis)


def calculate_forecast_index(
    dataframe:pl.DataFrame,
    offset_days:int,
    interval:str="1d"
) -> pl.DataFrame:
    """
    Calculate the forecast index based on the last timestamp in the dataframe and an offset.

    Parameters
    ----------
    dataframe : pl.DataFrame
        The input Polars DataFrame containing the timestamps.

    offset_days : int
        The number of days by which to offset the last timestamp.

    interval : str, optional
        The interval for the date range, e.g., "1d" for daily, "1w" for weekly. Default is "1d".

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the forecast index dates.

    Examples
    --------
    >>> import polars as pl
    >>> dataframe = pl.DataFrame({
    ...     'timestamp': ['2022-01-01', '2022-01-02', '2022-01-03']
    ... })
    >>> forecast_index = calculate_forecast_index(dataframe, offset_days=20)
    >>> print(forecast_index)
    shape: (21, 1)
    ╭──────────────────╮
    │ date             │
    │ ---              │
    │ datetime         │
    ╞══════════════════╡
    │ 2022-01-22 00:00 │
    │ 2022-01-23 00:00 │
    │ 2022-01-24 00:00 │
    │ 2022-01-25 00:00 │
    │ 2022-01-26 00:00 │
    │ 2022-01-27 00:00 │
    │ 2022-01-28 00:00 │
    │ 2022-01-29 00:00 │
    │ 2022-01-30 00:00 │
    │ 2022-01-31 00:00 │
    │ 2022-02-01 00:00 │
    │ 2022-02-02 00:00 │
    │ 2022-02-03 00:00 │
    │ 2022-02-04 00:00 │
    │ 2022-02-05 00:00 │
    │ 2022-02-06 00:00 │
    │ 2022-02-07 00:00 │
    │ 2022-02-08 00:00 │
    │ 2022-02-09 00:00 │
    │ 2022-02-10 00:00 │
    │ 2022-02-11 00:00 │
    │ 2022-02-12 00:00 │
    │ 2022-02-13 00:00 │
    ╰──────────────────╯
    """
    last_timestamp = dataframe.select(
        pl.last(
            "timestamp"
        )
    )

    # Offset the timestamp by the specified number of days
    offset_timestamp = last_timestamp.with_columns(
        offset=pl.col("timestamp").dt.offset_by(f"{offset_days}d")
    )

    # Create a date range from the original timestamp to the offset timestamp
    forecast_index_dates = pl.date_range(
        start=offset_timestamp["timestamp"],
        end=offset_timestamp["offset"],
        interval=interval,
        eager=True
    ).alias("date")

    return forecast_index_dates


def create_forecast_plot(
    dataframe: pl.DataFrame,
    forecast_index: pl.DataFrame,
    median: np.ndarray,
    high: np.ndarray,
    low: np.ndarray
) -> go.Figure:
    """
    Create a Plotly figure for time series forecasting with historical data, median forecast, and prediction interval.

    Parameters
    ----------
    dataframe : pl.DataFrame
        The input Polars DataFrame containing the historical data.

    forecast_index : pl.DataFrame
        Polars DataFrame of datetime objects representing the forecast dates.

    median : array-like
        Array-like object containing the median forecast values.

    high : array-like
        Array-like object containing the upper bound of the prediction interval.

    low : array-like
        Array-like object containing the lower bound of the prediction interval.

    Returns
    -------
    go.Figure
        A Plotly figure object.

    Examples
    --------
    >>> import plotly.graph_objects as go
    >>> import numpy as np
    >>> dataframe = pl.DataFrame({"timestamp": ["2022-01-01", "2022-01-02", "2022-01-03"], "close": [100, 110, 120]})
    >>> date = ["2022-01-04", "2022-01-05", "2022-01-06"]
    >>> median = [105, 115, 125]
    >>> high = [110, 120, 130]
    >>> low = [100, 110, 120]
    >>> fig = create_forecast_plot(dataframe, date, median, high, low)
    >>> fig.show()
    """
    fig = go.Figure()

    # Add historical data to the plot
    fig.add_trace(
        go.Scatter(
            x=dataframe["timestamp"],
            y=dataframe["close"],
            mode="lines",
            name="Historical Data",
            line=dict(color="royalblue")
        )
    )

    # Add median forecast data to the plot
    fig.add_trace(
        go.Scatter(
            x=forecast_index,
            y=median,
            mode="lines",
            name="Median Forecast",
            line=dict(color="tomato")
        )
    )

    # Add prediction interval (fill area between lines) to the plot
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([forecast_index, forecast_index[::-1]]),
            y=np.concatenate([high, low[::-1]]),
            fill="toself",
            fillcolor="tomato",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% Prediction Interval",
            showlegend=True,
            opacity=0.3
        )
    )

    # Update the layout for better visualization
    fig.update_layout(
        title="Time Series Forecasting: 80% Prediction Interval",
        xaxis_title="Time",
        yaxis_title="Value",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        # template="plotly_dark"
    )

    return fig


@spaces.GPU
def make_forecast(
    context: torch.Tensor,
    prediction_length: int,
    pipeline
) -> torch.Tensor:
    """
    Generate a forecast using the specified context and prediction length.

    Parameters
    ----------
    context : torch.Tensor or list of torch.Tensor
        The context data used for generating the forecast.

    prediction_length : int
        The length of the forecast.

    pipeline : torch.Module
        The forecasting pipeline model.

    Returns
    -------
    torch.Tensor
        The forecast tensor.

    Examples
    --------
    >>> import torch
    >>> context = torch.tensor([1, 2, 3, 4, 5])
    >>> prediction_length = 20
    >>> pipeline = YourForecastingPipeline()
    >>> forecast = make_forecast(context, prediction_length, pipeline)
    """
    forecast = pipeline.predict(
        context,
        prediction_length
    )

    return forecast


def fetch_data(
    company:str
):
    """
    Fetch historical data for a given company symbol.

    Parameters
    ----------
    company : str
        The symbol representing the company.

    Returns
    -------
    Tuple[gr.Dataframe, gr.Radio, gr.Button, gr.Row]
        A tuple containing Gradio components representing the fetched data.

    Notes
    -----
    This function fetches historical data for a given company symbol using the `Security` 
    class and returns Gradio components to visualize and interact with the fetched data.

    Examples
    --------
    >>> import datetime
    >>> # Fetch data for company with symbol "AAPL"
    >>> data, radio, button, row = fetch_data("AAPL")
    >>> # Display fetched data and components
    >>> gr.Interface([data, radio, button, row], "Fetch Data").launch()
    """
    security = Security(
        symbol=company
    )

    dataframe = security.fetch(
        start=datetime(2020, 9, 1)
    )

    return gr.Dataframe(
        visible=True,
        value=dataframe,
    ), gr.Radio(
        choices=dataframe.columns, 
        visible=True,
        interactive=True,
        value="close"
    ), gr.Button(
        variant="primary",
        interactive=True
    ), gr.Row(
        visible=True
    )


def plot_forecast(
    dataframe: pl.DataFrame,
    periods: int,
    column: str
) -> go.Figure:
    """
    Generate a forecast plot based on the provided data.

    Parameters
    ----------
    dataframe : pl.DataFrame
        The input DataFrame containing the data.
        
    periods : int
        The number of periods for the forecast.
        
    column : str
        The column name containing the data to forecast.

    Returns
    -------
    plotly.graph_objs.Figure
        A Plotly Figure object representing the forecast plot.

    Notes
    -----
    This function generates a forecast plot based on the provided data using the specified number of periods and the column containing the data to forecast. It internally uses other functions such as `make_forecast`, `calculate_quantiles`, `calculate_forecast_index`, and `create_forecast_plot` to generate the forecast plot.

    Examples
    --------
    >>> import polars as pl
    >>> from plotly import graph_objs as go
    >>> # Create a sample DataFrame
    >>> df = pl.DataFrame({'timestamp': ['2024-01-01', '2024-01-02', '2024-01-03'],
    ...                    'data': [10, 20, 30]})
    >>> # Plot the forecast
    >>> fig = plot_forecast(df, periods=5, column='data')
    >>> fig.show()
    """
    global pipeline

    dataframe = dataframe.with_columns(
        pl.col("timestamp").str.to_datetime()
    )

    forecast = make_forecast(
        context=torch.tensor(
            dataframe[
                column
            ]
        ),
        prediction_length=periods,
        pipeline=pipeline
    )

    low, median, high = calculate_quantiles(
        forecast_data=forecast[0].numpy(),
        quantiles=[0.1, 0.5, 0.9],
        axis=0
    )

    forecast_index = calculate_forecast_index(
        dataframe=dataframe,
        offset_days=periods
    )

    fig = create_forecast_plot(
        dataframe=dataframe,
        forecast_index=forecast_index,
        median=median,
        high=high,
        low=low
    )

    return fig


title = "# MANATEE(lm) : Market Analysis based on language model architectures"
description = '''
This Space focuses on employing LLM to analyze time series data for forecasting purposes, based on the "Chronos: Learning the Language of Time Series" paper from the Amazon Web Services and Amazon Supply Chain Optimization Technologies.

Opened with 🤗 in Paris by Louis Brulé Naudet (@louisbrulenaudet).
'''

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.DuplicateButton()

    with gr.Row():
        with gr.Column():

            with gr.Group():
                company_dropdown = gr.Dropdown(
                    choices=[
                        "AAPL", 
                        "NVDA", 
                        "MSFT"
                    ], 
                    label="Company Selection",
                    allow_custom_value=True,
                )

                fetch_button = gr.Button(
                    "Fetch data"
                )

            periods_slider = gr.Slider(
                minimum=1, 
                maximum=64,
                value=20,
                step=1.0,
                interactive=True,
                label="Forecast periods"
            )

            column_dropdown = gr.Radio(
                label="Column Selection",
                visible=False,
            )

            submit_button = gr.Button(
                "Submit",
                interactive=False
            )


        with gr.Column():
            dataframe_vis = gr.Dataframe(
                type="polars",
                visible=False,
                interactive=False,
                height=400,
            )
    
    with gr.Row(visible=False) as results_row:
        forecast_plot = gr.Plot(
            label="forecast"
        )    

        
    fetch_button.click(
        fn=fetch_data,
        inputs=company_dropdown,
        outputs=[
            dataframe_vis,
            column_dropdown,
            submit_button,
            results_row
        ]
    )

    submit_button.click(
        fn=plot_forecast,
        inputs=[
            dataframe_vis,
            periods_slider,
            column_dropdown
        ],
        outputs=forecast_plot
    )


if __name__ == "__main__":
    demo.launch(
        show_api=False
    )

    # sma, upper_band, lower_band = SmaBB(
    #     dataframe=dataframe,
    #     column="close",
    #     window_size=14
    # ).compute() 

    # rsi = RSI(
    #     dataframe=dataframe,
    #     column="close",
    #     window_size=14
    # ).compute() 