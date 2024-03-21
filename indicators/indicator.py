# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of the License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABCMeta, abstractmethod
from typing import List

class Indicator(
    metaclass=ABCMeta
):
    """
    Abstract base class for financial indicators.

    Subclasses should implement the `compute` method to compute the indicator values.
    """
    def __init__(
        self, 
        name: str
    ) -> None:
        """
        Initialize the Indicator object.

        Parameters
        ----------
        name : str
            Name of the indicator.

        Attributes
        ----------
        name : str
            Name of the indicator.
        serie : None
            Series of values for the indicator.
        """
        self.name = name
        self.serie = None


    @abstractmethod
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


    @abstractmethod
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
