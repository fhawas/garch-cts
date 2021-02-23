# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 06:34:45 2021

@author: Francisco
"""

import yfinance as yf

def get_data(ticker, start_date, end_date, provider):
    """
    input:
        ticker: identifier of stock
        start_date: start date of time series, format "yyyy-mm-dd"
        end_date: end date of time series, format "yyyy-mm-dd"
        provider:
            1: use yfinance
    """
    
    if provider == 1:
        stock_object = yf.Ticker(ticker)
        data = stock_object.history(start=start_date, end=end_date)
    
    return data
    