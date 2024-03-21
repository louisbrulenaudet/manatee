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

class SmaBB(Indicator):
    """
    Class for computing and plotting Bollinger Bands based on Simple Moving Average (SMA).
    
    This class inherits from the Indicator class and provides methods to calculate
    the SMA Bollinger Bands and plot them using Plotly.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Polars DataFrame containing the data.
    
    column : str
        Name of the column in the DataFrame to compute Bollinger Bands for.
    
    window_size : int, optional
        Size of the window for computing the SMA, by default 14.

    Attributes
    ----------
    dataframe : pl.DataFrame
        Polars DataFrame containing the data.
    
    column : str
        Name of the column in the DataFrame to compute Bollinger Bands for.
    
    window_size : int
        Size of the window for computing the SMA.

    Methods
    -------
    compute()
        Calculates the SMA Bollinger Bands.

    plot()
        Plots the SMA Bollinger Bands using Plotly.
    """
    def __init__(
        self, 
        dataframe: pl.DataFrame, 
        column: str, 
        window_size: int = 14,
        min_periods:int = None
    ) -> None:
        """
        Initializes the SmiBB object.

        Parameters
        ----------
        dataframe : pl.DataFrame
            Polars DataFrame containing the data.

        column : str
            Name of the column in the DataFrame to compute Bollinger Bands for.
        
		window_size : int, optional
            Size of the window for computing the SMA, by default 14.

        min_periods : int, optional
            The number of values in the window that should be non-null before computing a result. 
            If None, it will be set equal to: 
            - the window size, if window_size is a fixed integer ;
            - 1, if window_size is a dynamic temporal size.
        """
        super().__init__(
            name="SmaBB"
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
        Calculates the SMA Bollinger Bands.

        Returns
        -------
        pl.Series
            Series containing the Simple Moving Average (SMA).

        pl.Series
            Series containing the Upper Bollinger Band.

        pl.Series
            Series containing the Lower Bollinger Band.
        """
        try:
            self.sma = self.dataframe[
                self.column
            ].rolling_mean(
                window_size=self.window_size,
                min_periods=self.min_periods
            )

            std = self.dataframe[
                self.column
            ].rolling_std(
                window_size=self.window_size
            )
            
            self.upper_band = self.sma + (std * 2)
            self.lower_band = self.sma - (std * 2)

            return self.sma, self.upper_band, self.lower_band
        
        except Exception as e:
            print(f"Error calculating Bollinger Bands based on Simple Moving Average (SMA): {e}")
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
                y=self.sma, 
                mode="lines", 
                name="SMA"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.dataframe[x], 
                y=self.upper_band, 
                mode="lines", 
                name="Upper Band"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.dataframe[x], 
                y=self.lower_band, 
                mode="lines", 
                fill="tonexty", 
                name="Lower Band"
            )
        )
        
        fig.update_layout(
            title="SMA Bollinger Bands", 
            xaxis_title="Date", 
            yaxis_title="Value"
        )
        
        return fig