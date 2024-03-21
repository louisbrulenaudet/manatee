---
title: >-
  Manatee(lm): Market Analysis based on language model architectures
emoji: 📈
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.22.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: Market Analysis based on language model architectures
---

# MANATEE(lm) : Market Analysis based on language model architectures
[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg)](https://badge.fury.io/py/tensorflow) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Maintainer](https://img.shields.io/badge/maintainer-@louisbrulenaudet-blue)

This project focuses on employing LLM to analyze time series data for forecasting purposes, based on the "Chronos: Learning the Language of Time Series" paper from the Amazon Web Services and Amazon Supply Chain Optimization Technologies. The MANATEE project is designed to fetch, compute, and plot historical data for financial securities, leveraging APIs from Alpaca and the power of Polars and Plotly for data manipulation and visualization. With features like calculating the rolling mean and Relative Strength Index (RSI), this tool also aids in analyzing the past performance of stocks and crypto assets.

![Plot](https://github.com/louisbrulenaudet/manatee/blob/main/scatter.png?raw=true)

From source :
> In this work, we take a step back and ask: what are the fundamental differences between a language model that predicts the next token, and a time series forecasting model that predicts the next values? Despite the apparent distinction — tokens from a finite dictionary versus values from an unbounded, usually continuous domain — both endeavors fundamentally aim to model the sequential structure of the data to predict future patterns. Shouldn't good language models “just work” on time series? This naive question prompts us to challenge the necessity of time-series-specific modifications, and answering it led us to develop Chronos, a language modeling framework minimally adapted for time series forecasting. Chronos tokenizes time series into discrete bins through simple scaling and quantization of real values. In this way, we can train off-the-shelf language models on this “language of time series,” with no changes to the model architecture. Remarkably, this straightforward approach proves to be effective and efficient, underscoring the potential for language model architectures to address a broad range of time series problems with minimal modifications.
[...]

## Dependencies
### Libraries Used:

1. **`json`**: A built-in Python library for parsing JSON data. No need for installation.

2. **`datetime` & `time`**: Built-in Python libraries for handling date and time. Used here for defining time frames for data fetching. No installation required.

3. **`plotly`** (as `px`): Provides an easy-to-use interface to Plotly, which is used for creating interactive plots. Install via pip:
   ```shell
   pip3 install plotly
   ```
   
4. **`polars`** (as `pl`): A fast DataFrames library ideal for financial time-series data. Install using pip:
   ```shell
   pip3 install polars
   ```
   
5. **`alpaca-py`**: A Python library for Alpaca API. It provides access to historical stock/crypto data and trading operations. Install using pip:
   ```shell
   pip3 install alpaca-trade-api
   ```

### Installation Guide

To install all the dependencies, you can use the following command:

```shell
pip3 install plotly polars alpaca-py transformers gradio spaces
```

Note: Ensure you have Python installed on your system before proceeding with the installation of these libraries.

## Best Practices
- **API Keys Management**: For security reasons, avoid hardcoding your API keys into the script. Consider using environment variables or a secure vault service.

- **Data Privacy**: When handling financial data, it's crucial to comply with data protection regulations (such as GDPR, CCPA). Ensure you have the right to use and share the data fetched through this tool.

- **Error Handling**: The script includes basic error handling, but for production use, consider implementing more comprehensive try-except blocks to handle network errors, API limit exceptions, and data inconsistencies.

- **Plotting Considerations**: This tool uses Plotly for visualization, which is very versatile but can be resource-intensive for large datasets. For analyzing large datasets, consider creating plots with fewer data points or aggregating the data before plotting.

- **Resource Management**: When dealing with large datasets or numerous API requests, monitor your system's and the API's usage to avoid overloading.

- **Version Control**: Regularly update your dependencies. Financial APIs and data handling libraries evolve, and keeping them up to date can improve security, efficiency, and accessibility of new features.

## Citing this project
If you use this code in your research, please use the following BibTeX entry.

```BibTeX
@misc{louisbrulenaudet2023,
	author = {Louis Brulé Naudet},
	title = {MANATEE(lm) : Market Analysis based on language model architectures},
	howpublished = {\url{https://huggingface.co/spaces/louisbrulenaudet/manatee}},
	year = {2024}
}

```
## Feedback
If you have any feedback, please reach out at [louisbrulenaudet@icloud.com](mailto:louisbrulenaudet@icloud.com).
