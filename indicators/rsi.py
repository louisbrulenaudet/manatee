# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of the License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import plotly.graph_objects as go
import polars as pl

from typing import Union

from plotly.subplots import make_subplots

from indicators.indicator import Indicator

class RSI(Indicator):
    """
    Class for computing the Relative Strength Index (RSI) for a specified column.

    It is intended to chart the current and historical strength or weakness of 
    a stock or market based on the closing prices of a recent trading period.
    The RSI is classified as a momentum oscillator, measuring the velocity and 
    magnitude of price movements. Momentum is the rate of the rise or fall in price. 
    The relative strength RS is given as the ratio of higher closes to lower closes. 
    Concretely, one computes two averages of absolute values of closing price changes, 
    i.e. two sums involving the sizes of candles in a candle chart. The RSI computes momentum 
    as the ratio of higher closes to overall closes: stocks which have had more or stronger 
    positive changes have a higher RSI than stocks which have had more or stronger negative changes.

    The RSI is most typically used on a 14-day timeframe, measured on a scale 
    from 0 to 100, with high and low levels marked at 70 and 30, respectively. 
    Short or longer timeframes are used for alternately shorter or longer 
    outlooks. High and low levels—80 and 20, or 90 and 10—occur less frequently but 
    indicate stronger momentum.
    
    This class inherits from the Indicator class and provides methods to calculate
    the SMA Bollinger Bands and plot them using Plotly.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Polars DataFrame containing the data.
    
    column : str
        The name of the column for which to calculate RSI.
    
    window_size : int, optional
        Size of the window for computing the RSI, by default 1.

    min_periods : int, optional
        The minimum number of observations required to have a non-null result. Default is 1.

    Attributes
    ----------
    dataframe : pl.DataFrame
        Polars DataFrame containing the data.
    
    column : str
        Name of the column in the DataFrame to compute RSI for.
    
    window_size : int
        Size of the window for computing the RSI.

    min_periods : int, optional
        The minimum number of observations required to have a non-null result. Default is 1.

    Methods
    -------
    compute()
        Calculates the Relative Strength Index (RSI).

    plot()
        Plots the Relative Strength Index (RSI) using Plotly.
    """
    def __init__(
        self, 
        dataframe: pl.DataFrame, 
        column: str, 
        window_size: int = 14,
        min_periods:int = 1
    ) -> None:
        """
        Initializes the RSI object.

        Parameters
        ----------
        dataframe : pl.DataFrame
            Polars DataFrame containing the data.
        
        column : str
            Name of the column in the DataFrame to compute RSI for.
        
        window_size : int
            Size of the window for computing the RSI.

        min_periods : int, optional
            The minimum number of observations required to have a non-null result. Default is 1.
        """
        super().__init__(
            name="RSI"
        )
        self.dataframe = dataframe
        self.column = column
        self.window_size = window_size
        self.min_periods = min_periods


    def get_name(
        self
    ) -> str:
        """
        Get the name of the indicator.

        Returns
        -------
        str
            Name of the indicator.
        """
        return self.name


    def get_series(
        self
    ):
        """
        Get the series of values for the indicator.

        Returns
        -------
        None
            Since the series is not yet computed, returns None.
        """
        return self.series


    def compute(
        self
    ) -> Union[pl.Series, pl.Series, pl.Series]:
        """
        Calculates the Relative Strength Index (RSI).

        Returns
        -------
        pl.Series
            Series containing the Relative Strength Index (RSI).
        """
        try:
            self.dataframe = self.dataframe.with_columns(
                change=pl.col(self.column).diff()
            ).with_columns(
                [
                    pl.when(
                    pl.col("change") > 0).then(
                        pl.col("change")
                    ).otherwise(
                        0
                    ).alias("gain"),
                    pl.when(
                        pl.col("change") < 0
                    ).then(
                        -pl.col("change")
                    ).otherwise(
                        0
                    ).alias("loss")
                ]
            ).with_columns(
                [
                    pl.col("gain").rolling_mean(
                        window_size=self.window_size,
                        min_periods=self.min_periods
                    ).alias("avg_gain"),
                    pl.col("loss").rolling_mean(
                        window_size=self.window_size,
                        min_periods=self.min_periods
                    ).alias("avg_loss")
                ]
            ).with_columns(
                rs=pl.col("avg_gain").truediv(pl.col("avg_loss")),
            ).with_columns(
                rsi=pl.lit(100).sub(
                    pl.lit(100).truediv(
                        pl.lit(1).add(pl.col("avg_loss"))
                    )
                )
            )

            return self.dataframe["rsi"].tail(-1)

        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return None


    def plot(
        self, 
        x:str,
        fig=go.Figure,
    ) -> go.Figure:
        """
        Plots the SMA Bollinger Bands using Plotly.

        Parameters
        ----------
        x : str
            The name of the column to be used as the x-axis in the plot.

        fig : plotly.graph_objects.Figure
            Plotly figure object to which the Bollinger Bands plot will be added.

        Returns
        -------
        go.Figure
            Plotly figure containing the SMA Bollinger Bands plot.
        """
        #fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.dataframe[x], 
                y=rsi, 
                mode="lines", 
                name="RSI",
            ),
            row=2, 
            col=1
        )
        
        return fig