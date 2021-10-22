from typing import List, Any
import datetime
# import numpy as np
from sklearn import preprocessing
# import pandas as pd
import statistics as stat
from dateutil import parser
from scipy.stats import pearsonr
# import matplotlib.pyplot as plt
import scipy.fftpack
from stock_analysis.ml_utils import train_XGB
# from darts import TimeSeries
from darts.models import ExponentialSmoothing, ExponentialSmoothing, ARIMA, RNNModel, TCNModel, KalmanFilter
from darts.utils.missing_values import fill_missing_values

# # Darts Example
# import sys
# import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from functools import reduce

from darts import TimeSeries
from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    Prophet,
    ExponentialSmoothing,
    ARIMA,
    AutoARIMA,
    RegressionEnsembleModel,
    RegressionModel,
    Theta,
    FFT
)

from darts.metrics import mape, mase
# from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis
# from darts.datasets import AirPassengersDataset

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
from tqdm import tqdm




def correlate_dict(portfolio, stock, date_only_dict, up_or_down_list):
    hour_dict = {}
    for d in up_or_down_list:
        for hour in date_only_dict[d]:
            hour_parse = parser.parse(hour)
            hour_only = hour_parse.strftime("%H:%M:%S")
            # Relative change - price at specific hour/open price
            if hour_only not in hour_dict:
                hours_price = []
                hours_price.append((portfolio[stock]['stock_data']['1M'][hour].get('close', None) /
                                    portfolio[stock]['stock_data']['6M'][d + ' 16:00:00'].get('open', None)) - 1)
                hour_dict[hour_only] = hours_price
            else:
                hour_dict[hour_only].append((portfolio[stock]['stock_data']['1M'][hour].get('close', None) /
                                             portfolio[stock]['stock_data']['6M'][d + ' 16:00:00'].get('open',
                                                                                                       None)) - 1)
    # Close / Open price
    close_price_list = [(portfolio[stock]['stock_data']['6M'][date + ' 16:00:00'].get('close', None) /
                         portfolio[stock]['stock_data']['6M'][date + ' 16:00:00'].get('open', None)) - 1 for date in
                        up_or_down_list]
    corr_dict = {}
    for h in hour_dict:
        array1 = np.array(hour_dict[h])
        array2 = np.array(close_price_list)
        if len(array1) == len(array2) and len(array1) >= 2:
            correlate = pearsonr(array1, array2)
        else:
            correlate = 'N/A'
        corr_dict[h] = correlate
    return corr_dict


def stocks_stat(dx_masked, dx_normal):
    # Create series of filtered array
    count_zeros, count_down, count_up = [], [], []
    end_of_j, j = False, 0
    init_up_count, init_down_count, init_z_count = False, False, False
    z_counter, up_count, down_count = 0, 0, 0
    while j < len(dx_masked):
        z_counter = 0 if init_z_count else z_counter
        down_count = 0 if init_down_count else down_count
        up_count = 0 if init_up_count else up_count
        if j == len(dx_masked): break
        if not end_of_j:
            if dx_masked[j] == 0.0:
                z_counter += 1
                while (dx_normal[j] == 0.0 and j <= len(dx_normal) - 1):
                    z_counter += 1
                    j += 1
                    if j == len(dx_masked): end_of_j = True; break
                if z_counter > 8:
                    init_z_count = False;
                    init_down_count, init_up_count = True, True
                else:
                    count_zeros.append(z_counter - 1);
                    init_z_count = True
                if z_counter == 1: j += 1; continue
        if not end_of_j:
            if dx_masked[j] < 0:
                down_count += 1
                while (dx_normal[j] < 0 and j <= len(dx_normal) - 1):
                    down_count += 1
                    j += 1
                    if j == len(dx_masked): end_of_j = True; break
                if down_count > 8:
                    init_down_count = False;
                    init_up_count, init_z_count = True, True
                else:
                    count_down.append(down_count - 1);
                    init_down_count = True
                if down_count == 1: j += 1; continue
        if not end_of_j:
            if dx_masked[j] > 0:
                up_count += 1
                while (dx_normal[j] > 0 and j <= len(dx_normal) - 1):
                    up_count += 1
                    j += 1
                    if j == len(dx_masked): end_of_j = True; break
                if up_count > 8:
                    init_up_count = False;
                    init_z_count, init_down_count = True, True
                else:
                    count_up.append(up_count - 1);
                    init_up_count = True
                if up_count == 1: j += 1; continue
    if count_up:
        up_med = stat.median(count_up) if count_up else 0
        up_mean = stat.mean(count_up)
        up_medg = stat.median_grouped(count_up)
        up_var = stat.variance(count_up) if len(count_up) > 2 else 0
    else:
        up_med, up_mean, up_var, up_medg = 0, 0, 0, 0
    if count_zeros:
        zero_med = stat.median(count_zeros) if count_zeros else 0
        zero_mean = stat.mean(count_zeros)
        zero_var = stat.variance(count_zeros) if len(count_zeros) > 2 else 0
        zero_medg = stat.median_grouped(count_zeros)
    else:
        zero_var, zero_mean, zero_medg, zero_med = 0, 0, 0, 0
    if count_down:
        down_med = stat.median(count_down) if count_down else 0
        down_mean = stat.mean(count_down)
        down_var = stat.variance(count_down) if len(count_down) > 2 else 0
        down_medg = stat.median_grouped(count_down)
    else:
        down_var, down_mean, down_medg, down_med = 0, 0, 0, 0
    med_arr = np.column_stack([[up_med, up_medg, up_mean, up_var, len(count_up)],
                               [zero_med, zero_medg, zero_mean, zero_var, len(count_zeros)],
                               [down_med, down_medg, down_mean, down_var, len(count_down)]])
    stats_df = pd.DataFrame(med_arr, columns=['Up', 'Zero', 'Down'],
                            index=['Median', 'Grouped Median', 'Mean Value', 'Variance', 'Num of Series'])
    # count_arr = np.array(count_down)
    # count_df = pd.DataFrame(count_arr)
    # count_df.plot(title = '{}'.format(portfolio[stock]['stock_data']['name']) ,kind = 'kde')
    # plt.plot(count_up,range(len(count_up)),'.')
    # npcountup=np.array(count_up)
    # npx=np.linspace(0,4,16)
    # z = np.polyfit(npx,npcountup, 10)
    # plt.plot(z)
    # def func(x,a,b,c):
    #    return a * np.exp(-b * (x*x)) + c
    # optimizedParameters, pcov = opt.curve_fit(func,npx, npcountup)
    # plt.plot(npx, func(npx, *optimizedParameters), label="fit")
    return stats_df


def create_5d_df(portfolio, stock, ref_timeframe, percentile=90):
    deriv_dates = sorted(list(portfolio[stock]['stock_data']['5D'].keys()))
    price_sorted = [portfolio[stock]['stock_data']['5D'][k]['close'] for k in deriv_dates]

    dx_arr = np.diff(price_sorted)
    price_array = np.array(price_sorted[1:])
    med = np.median(dx_arr)
    per = np.percentile(dx_arr, percentile)
    # Create the Scaler object
    # scaler = preprocessing.StandardScaler()
    scaler_MinMax = preprocessing.MinMaxScaler()
    # Create Masking according to the requested Percentiles
    mask_plus = (dx_arr > per)
    mask_minus = (dx_arr < -per)
    mask = np.logical_or(mask_minus, mask_plus)
    ones = np.ones_like(dx_arr)
    mask = np.logical_xor(mask, ones)
    masked_array = np.ma.array(dx_arr, mask=mask)
    dx_masked = np.ma.filled(masked_array, [0])

    # Scaling Prices and Derivatives
    scaled_price = scaler_MinMax.fit_transform(price_array.reshape(-1, 1))
    scaled_dx = scaler_MinMax.fit_transform(dx_masked.reshape(-1, 1))

    deriv_dates = deriv_dates[1:]
    cols = np.column_stack((deriv_dates[-ref_timeframe:], dx_masked[-ref_timeframe:], scaled_dx[-ref_timeframe:],
                            price_array[-ref_timeframe:], scaled_price[-ref_timeframe:], dx_arr[-ref_timeframe:]))
    df = pd.DataFrame(cols, columns=['DateTime', 'Derivatives', 'Deriv_Norm', 'Price', 'Price_Norm', 'Unmasked'])

    # Casting objects
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
    df['Deriv_Norm'] = df['Deriv_Norm'].astype('float64')
    df['Derivatives'] = df['Derivatives'].astype('float64')
    df['Price_Norm'] = df['Price_Norm'].astype('float64')
    df['Price'] = df['Price'].astype('float64')
    df['Unmasked'] = df['Unmasked'].astype('float64')
    return df


def create_dataframe(portfolio, stock, ref_timeframe, period='1M', percentile=90, to_plot=False, is_vol=False):
    today = datetime.datetime.today()
    deriv_dates = sorted(list(portfolio[stock]['stock_data']['6M_Marketbeat'].keys()))
    deriv_1M_dates = sorted(list(portfolio[stock]['stock_data']['{}'.format(period)].keys()))
    price_sorted: List[Any] = []
    price_1M_sorted: List[Any] = []
    for k in deriv_dates:
        check = portfolio[stock]['stock_data']['6M_Marketbeat'][k].get(stock, None)
        if check:
            price = portfolio[stock]['stock_data']['6M_Marketbeat'][k][stock]['ClosingPrice']
            price_sorted.append(price)
        else:
            price_sorted.append(price_sorted[-1])
    if portfolio[stock]['stock_data']['real_time']:
        deriv_dates.append(today.strftime("%Y-%m-%d"))
        price_sorted.append(list(portfolio[stock]['stock_data']['real_time'].values())[0])
    for d in deriv_1M_dates:
        check2 = portfolio[stock]['stock_data']['{}'.format(period)][d].get('close', None) if not is_vol else \
            portfolio[stock]['stock_data']['{}'.format(period)][d].get('volume', None)
        if check2:
            price1M = portfolio[stock]['stock_data']['{}'.format(period)][d]['close'] if not is_vol else \
                portfolio[stock]['stock_data']['{}'.format(period)][d]['volume']
            price_1M_sorted.append(price1M)
        else:
            price_1M_sorted.append(price_sorted[-1])
    if not is_vol:
        if portfolio[stock]['stock_data']['real_time']:
            deriv_1M_dates.append(today.strftime("%Y-%m-%d"))
            price_1M_sorted.append(list(portfolio[stock]['stock_data']['real_time'].values())[0])
    dx_arr = np.diff(price_sorted)
    dx_1M_arr = np.diff(price_1M_sorted)
    price_array = np.array(price_sorted[1:])
    price_1M_array = np.array(price_1M_sorted[1:])
    med = np.median(dx_arr)
    per = np.percentile(dx_arr, percentile)
    per1M = np.percentile(dx_1M_arr, percentile)
    # Create the Scaler object
    # scaler = preprocessing.StandardScaler()
    scaler_MinMax = preprocessing.MinMaxScaler()
    # Create Masking according to the requested Percentiles
    mask_plus = (dx_arr > per)
    mask_minus = (dx_arr < -per)
    mask = np.logical_or(mask_minus, mask_plus)
    ones = np.ones_like(dx_arr)
    mask = np.logical_xor(mask, ones)
    masked_array = np.ma.array(dx_arr, mask=mask)
    dx_masked = np.ma.filled(masked_array, [0])
    # Masking for 1M
    mask_1M_plus = (dx_1M_arr > per1M)
    mask_1M_minus = (dx_1M_arr < -per1M)
    mask1M = np.logical_or(mask_1M_minus, mask_1M_plus)
    ones1M = np.ones_like(dx_1M_arr)
    mask1M = np.logical_xor(mask1M, ones1M)
    masked_1M_array = np.ma.array(dx_1M_arr, mask=mask1M)
    dx_1M_masked = np.ma.filled(masked_1M_array, [0])
    # Scaling Prices and Derivatives
    scaled_price = scaler_MinMax.fit_transform(price_array.reshape(-1, 1))
    scaled_dx = scaler_MinMax.fit_transform(dx_masked.reshape(-1, 1))
    scaled_1M_price = scaler_MinMax.fit_transform(price_1M_array.reshape(-1, 1))
    scaled_1M_dx = scaler_MinMax.fit_transform(dx_1M_masked.reshape(-1, 1))
    deriv_dates = deriv_dates[1:]
    deriv_1M_dates = deriv_1M_dates[1:]
    cols = np.column_stack((deriv_dates[-ref_timeframe:], dx_masked[-ref_timeframe:], scaled_dx[-ref_timeframe:],
                            price_array[-ref_timeframe:], scaled_price[-ref_timeframe:], dx_arr[-ref_timeframe:]))
    df = pd.DataFrame(cols, columns=['DateTime', 'Derivatives', 'Deriv_Norm', 'Price', 'Price_Norm', 'Unmasked'])
    if is_vol:
        cols1M = np.column_stack((deriv_1M_dates[-ref_timeframe:], dx_1M_masked[-ref_timeframe:],
                                  scaled_1M_dx[-ref_timeframe:], price_1M_array[-ref_timeframe:],
                                  scaled_1M_price[-ref_timeframe:], dx_arr[-ref_timeframe:]))
    else:
        cols1M = np.column_stack((deriv_1M_dates[-ref_timeframe:], dx_1M_masked[-ref_timeframe:],
                                  scaled_1M_dx[-ref_timeframe:], price_1M_array[-ref_timeframe:],
                                  scaled_1M_price[-ref_timeframe:], dx_arr[-ref_timeframe:]))
    df1M = pd.DataFrame(cols1M, columns=['DateTime', 'Derivatives', 'Deriv_Norm', 'Price', 'Price_Norm',
                                         'Unmasked']) if not is_vol else pd.DataFrame(cols1M, columns=['DateTime',
                                                                                                       'Derivatives',
                                                                                                       'Deriv_Norm',
                                                                                                       'Volume',
                                                                                                       'Volume_Norm',
                                                                                                       'Unmasked'])
    # Casting objects
    df['DateTime'] = df['DateTime'].astype('datetime64[h]')
    df['Deriv_Norm'] = df['Deriv_Norm'].astype('float64')
    df['Derivatives'] = df['Derivatives'].astype('float64')
    df['Price_Norm'] = df['Price_Norm'].astype('float64')
    df['Price'] = df['Price'].astype('float64')
    df['Unmasked'] = df['Unmasked'].astype('float64')
    # 1M
    df1M['DateTime'] = df1M['DateTime'].astype('datetime64[h]')
    df1M['Deriv_Norm'] = df1M['Deriv_Norm'].astype('float64')
    df1M['Derivatives'] = df1M['Derivatives'].astype('float64')
    df1M['Unmasked'] = df1M['Unmasked'].astype('float64')
    if not is_vol:
        df1M['Price_Norm'] = df1M['Price_Norm'].astype('float64')
    else:
        df1M['Volume_Norm'] = df1M['Volume_Norm'].astype('float64')
    if not is_vol:
        df1M['Price'] = df1M['Price'].astype('float64')
    else:
        df1M['Volume'] = df1M['Volume'].astype('float64')
    # df.info()
    if is_vol:
        return df1M
    else:
        return df


# zero padding
def NextPowerOfTwo(number):
    # Returns next power of two following 'number'
    return np.ceil(np.log2(number))

def PadRight_with_initialization(arr, power):
    nextPower = NextPowerOfTwo(len(arr))
    nextPower = power
    deficit = int(np.power(2, nextPower) - len(arr))
    zero_pad = np.pad(arr, (0, deficit), 'constant', constant_values=(0, 0))
    return zero_pad

def plot_fft(df):
    medium_res = 8
    high_res = 15
    zero_pad_med = PadRight_with_initialization(df.Price, medium_res)
    zero_pad_high = PadRight_with_initialization(df.Price, high_res)
    price_fft = np.fft.rfft(zero_pad_high)
    n_samples = len(df.Price)
    time_delta = 3600*24
    sample_rate = n_samples / time_delta
    freq = np.fft.rfftfreq(n_samples, d=sample_rate)

    #plot code snippet
    # Number of samplepoints
    N = 600
    N = n_samples
    # sample spacing
    T = 1.0 / 800.0
    T = sample_rate

    x = np.linspace(0.0, N * T, N)
    y_med = zero_pad_med
    y_high = zero_pad_high
    y2 = df.Price

    xf_med = np.linspace(0.0, 1.0 / (2.0 * T), len(y_med) // 2)
    yf_med = scipy.fftpack.fft(y_med)
    xf_high = np.linspace(0.0, 1.0 / (2.0 * T), len(y_high) // 2)
    yf_high = scipy.fftpack.fft(y_high)
    yf_no_padding = np.fft.rfft(y2)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    medium_size=np.power(2, medium_res)
    high_size=np.power(2, high_res)
    fig, ax = plt.subplots(nrows=3)
    plt.suptitle("FFT Resolution Comparison")
    ax[0].plot(xf, 2.0 / N * np.abs(yf_no_padding[:N // 2]), 'b', label="No Zero-Padding")
    ax[0].legend(loc="upper right")
    ax[1].plot(xf_med, 2.0 / N * np.abs(yf_med[:len(yf_med) // 2]),'r', label=f"{medium_size} Zero-Padding")
    ax[1].legend(loc="upper right")
    ax[2].plot(xf_high, 2.0 / N * np.abs(yf_high[:len(yf_high) // 2]), 'g',label=f"{high_size} Zero-Padding")
    ax[2].legend(loc="upper right")
    plt.show()

def search_theta(train, val):
    # Search for the best theta parameter, by trying 50 different values
    thetas = 2 - np.linspace(-10, 10, 50)

    best_mape = float('inf')
    best_theta = 0

    for theta in thetas:
        model = Theta(theta)
        model.fit(train)
        pred_theta = model.predict(len(val))
        res = mape(val, pred_theta)

        if res < best_mape:
            best_mape = res
            best_theta = theta
    return best_theta

def train_theta(train, val, best_theta):
    best_theta_model = Theta(best_theta)
    best_theta_model.fit(train)
    pred_best_theta = best_theta_model.predict(len(val))

    print('The MAPE is: {:.2f}, with theta = {}.'.format(mape(val, pred_best_theta), best_theta))
    return pred_best_theta


def eval_model(model, train, val, is_filtered=None):
    model.fit(train)
    forecast = model.predict(len(val))
    if not is_filtered:
        print('model {} obtains MAPE: {:.2f}%'.format(model, mape(val, forecast)))
        forecast.plot(label=f'{model} forecast ', low_quantile=0.05, high_quantile=0.95)
    else:
        print('model {} , filtered, obtains MAPE: {:.2f}%'.format(model, mape(val, forecast)))
        forecast.plot(label=f'{model} , filtered, forecast ', low_quantile=0.05, high_quantile=0.95)
    return mape(val, forecast)


def search_best_forecast_period(model, data):
    best_day = 0
    raw_errors = []
    best_average_error = float('inf')
    for day in tqdm(range(3, 14), desc= 'Backtesting', colour='green'):
        average_error = model.backtest(data, start=0.8, forecast_horizon=day, verbose=False)
        if average_error < best_average_error:
            best_average_error = average_error
            best_day = day
        raw_errors += [(day, average_error)]

    return raw_errors, best_day


def plot_hist_with_counts(data):
    """
    histogram with counts - counts how many times a certain data occured in a bin (range)
    e.g. 13 times in the range 0-0.5

    * Need to add title manually outside the function
    :param data:
    :return:
    """
    if isinstance(data[0],tuple):
        series = [x[1] for x in data]
        bins_data = np.arange(0, max(series), 1)
    else:
        series = data
        bins_data = np.arange(0, max(data), 1)
    density, bins, _ = plt.hist(series, density=True, bins=bins_data)
    count, _ = np.histogram(series, bins)
    for x, y, num in zip(bins, density, count):
        if num != 0:
            plt.text(x, y + 0.005, num, fontsize=10, rotation=-90)  # x,y,str
    plt.show()


def search_best_model(train, val, train_f, val_f, method,  best_mape, best_model, best_method):
    for model_eval in [ExponentialSmoothing(), Prophet(), AutoARIMA(), Theta()]:
        res = eval_model(model_eval, train, val)
        res_f = eval_model(model_eval, train_f, val_f, is_filtered=True)
        if res < best_mape:
            best_mape = res
            best_model = model_eval
            best_method = method
        if res_f < best_mape:
            best_mape = res
            best_model = model_eval
            best_method = method
    return best_mape, best_model, best_method


def create_stock_dataframe(portfolio, timeframe, is_plot=False):
    print("Searching for Best Model and forecast")
    exceptions_list = []
    for stock in tqdm(portfolio, position=0, desc= 'Stocks Forecasting'):
        try:
            print(stock)
            # Create DataFrames
            df_price_frame = create_dataframe(portfolio, stock, timeframe)
            timeframe = 120
            df_price_month = create_dataframe(portfolio, stock, timeframe, period="1M")
            df_volume = create_dataframe(portfolio, stock, timeframe, period="6M", is_vol=True)

            # Calc Stats
            price_stats = stocks_stat(df_price_frame['Derivatives'], df_price_frame['Unmasked'])
            volume_stats = stocks_stat(df_volume['Derivatives'], df_volume['Unmasked'])
            #Add Stats to Portfolio dict
            portfolio[stock]['stock_stats'] = {'volume_stats': volume_stats, 'price_stats': price_stats}
            df_5d_price = create_5d_df(portfolio, stock, timeframe)



            # TODO : add backtesting - check for 1 week preidiction at a time
            # TODO: add particle filter
            # Todo: add catch22 implementation for Time series comparison and analysis

            # Darts Timeseries Playground

            # Kalman Filter noise reduction



            models = [ExponentialSmoothing()] #ARIMA,,TCNModel,  RNNModel
            for model in models:
                # 10 min period intra-day - 5 Days data
                plt.close()
                period = 24*60/10 # 24H*(60Min/10Min)
                # if model == ExponentialSmoothing :
                #     curr_model = model(seasonal_periods=period)
                # else:
                #     curr_model = model()
                series = TimeSeries.from_dataframe(df_5d_price, 'DateTime', 'Price', freq='10min', fill_missing_dates=True)
                methods = ['linear','quadratic','cubic','slinear','piecewise']
                # for method in methods:
                #     series = fill_missing_values(series,'auto',interpolate_kwargs={'method':method})
                #     len_20_percent = int(len(series)*0.2)
                #     train, val = series[:-len_20_percent], series[-len_20_percent:]
                #
                #
                #     filtered_price_5d = KalmanFilter(P=1000., R=50, Q=1).filter(series).values()
                #     filtered_price_5d_ts = TimeSeries.from_times_and_values(series.time_index, filtered_price_5d)
                #     train_f, val_f = filtered_price_5d_ts[:-len_20_percent], filtered_price_5d_ts[-len_20_percent:]
                #
                #     plt.close()
                #     series.plot(color='red', label='Noisy')
                #     filtered_price_5d_ts.plot(color='blue', label='filtered')
                #
                #     curr_model.fit(train)
                #     prediction = curr_model.predict(len(val), num_samples=1000)
                #     prediction.plot(label=f'{model.__str__(model)} forecast;  Interpolation :{method}', low_quantile=0.05, high_quantile=0.95)
                #
                #     curr_model.fit(train_f)
                #     prediction = curr_model.predict(len(val_f), num_samples=1000)
                #     prediction.plot(label=f'{model.__str__(model)} Filtered forecast;  Interpolation :{method}', low_quantile=0.05, high_quantile=0.95,  color = 'green')

                # 1 Day period  120 business Days data
                plt.close()
                period = 30
                # for period in range(2, 31, 3):
                print(f"Period {period}")
                # period = 7 # 7 days
                # if model == ExponentialSmoothing:
                #     curr_model = model(seasonal_periods=period)
                # else:
                #     model = model()
                series = TimeSeries.from_dataframe(df_price_frame, 'DateTime', 'Price', freq='D', fill_missing_dates=True)
                best_mape = float('inf')
                best_method = None
                best_model = None

                methods = ['linear','quadratic','cubic','slinear','piecewise']
                for method in methods:
                    series = fill_missing_values(series,'auto',interpolate_kwargs={'method':method})
                    len_20_percent = int(len(series)*0.2)
                    train, val = series[:-len_20_percent], series[-len_20_percent:]


                    filtered_price_4m = KalmanFilter(P=1000., R=50, Q=1).filter(series).values()
                    filtered_price_4m_ts = TimeSeries.from_times_and_values(series.time_index, filtered_price_4m)
                    train_f, val_f = filtered_price_4m_ts[:-len_20_percent], filtered_price_4m_ts[-len_20_percent:]

                    # best_mape, best_model, best_method = search_best_model(train,val,train_f,val_f,method,best_mape,best_model,best_method)


                    plt.close()
                    series.plot(color='red', label='Noisy')
                    filtered_price_4m_ts.plot(color='blue', label='filtered')


                    model.fit(train)
                    prediction = model.predict(len(val), num_samples=100)
                    prediction.plot(label=f'{model} forecast;  Interpolation :{method}', low_quantile=0.05, high_quantile=0.95)
                    print('model {} , interpolation {}, obtains MAPE: {:.2f}%'.format(model, method, mape(val, prediction)))


                    if mape(val, prediction) < best_mape:
                        best_mape = mape(val, prediction)
                        best_model = model
                        best_method = method


                    model.fit(train_f)
                    prediction_f = model.predict(len(val_f), num_samples=100)
                    prediction_f.plot(label=f'{model} Filtered forecast;  Interpolation :{method}', low_quantile=0.05, high_quantile=0.95, color = 'green')
                    print('model {} , interpolation {}, Filtered, obtains MAPE: {:.2f}%'.format(model, method, mape(val, prediction_f)))

                    if mape(val_f, prediction_f) < best_mape:
                        best_mape = mape(val_f, prediction_f)
                        best_model = model
                        best_method = method

                    # # Theta Model
                    # plt.close()
                    # series.plot(color='red', label='Noisy')
                    # filtered_price_4m_ts.plot(color='blue', label='filtered')
                    #
                    # theta = search_theta(train, val)
                    # pred_theta = train_theta(train, val, theta)
                    # pred_theta.plot(label=f'{model()} Filtered forecast;  Interpolation :{method}', low_quantile=0.05, high_quantile=0.95)
                    # print('model {} , interpolation {}, Filtered, obtains MAPE: {:.2f}%'.format(model(), method, mape(val, pred_theta)))
                    #
                    # theta_f = search_theta(train_f, val_f)
                    # pred_theta_f = train_theta(train_f, val_f, theta_f)
                    # pred_theta_f.plot(label=f'{model()} Filtered forecast;  Interpolation :{method}', low_quantile=0.05, high_quantile=0.95, color = 'green')
                    # print('model {} , interpolation {}, Filtered, obtains MAPE: {:.2f}%'.format(model(), method, mape(val, pred_theta_f)))

                    plt.close() # TODO: Remove


                # backtesting
                periods_errors, best_period = search_best_forecast_period(best_model, series)
                print(f"Best Period for Forecast: {best_period}")
                # plot_hist_with_counts(periods_errors) # TODO:  Uncomment
                plt.title("Periods error scores (histogram)")

                plt.close() # TODO: Remove

                average_error = best_model.backtest(series, start=0.8, forecast_horizon=best_period,
                                                          verbose=False)
                median_error = best_model.backtest(series, start=0.8, forecast_horizon=best_period,
                                                         reduction=np.median, verbose=False)
                print("Average error (MAPE) over all historical forecasts: {}".format(average_error))
                print("Median error (MAPE) over all historical forecasts: {}".format(median_error))

                raw_errors = best_model.backtest(series, start=0.8, forecast_horizon=best_period,
                                                       reduction=None, verbose=False)

                print(f"1 Week Period for Forecast: ")
                week_error = best_model.backtest(series, start=0.8, forecast_horizon=7,
                                                          verbose=False)
                print("1 Week Average error (MAPE) over all historical forecasts: {}".format(week_error))



                # plot_hist_with_counts(raw_errors) # TODO:  Uncomment
                plt.title("Individual error scores (histogram) for best period")

                plt.close()  # TODO: Remove

                series.plot(label='data')
                plt.legend()
                historical_fcast = best_model.historical_forecasts(series, start=0.8,
                                                                               forecast_horizon=best_period, verbose=False)
                # historical_fcast.plot(label='backtest 3-months ahead forecast (Theta)') # TODO:  Uncomment
                plt.title('MAPE = {:.2f}%'.format(mape(historical_fcast, series)))

                plt.close() # TODO: Remove

                # plot_residuals_analysis(best_model.residuals(series)) # TODO:  Uncomment


            # model = AutoARIMA()
            # model.fit(train)
            # prediction = model.predict(len(val))
            # series.plot()
            # prediction.plot(label='forecast-ARIMA', low_quantile=0.05, high_quantile=0.95)
            # plt.legend()



    #######################################################################        #####Continue HERE~!
            # plot_fft(df_price_month) # TODO: Uncomment
            # plot_fft(df_5d_price)

            if is_plot:
                # if not ((price_stats.iloc[:3]['Down']==1).all() or (price_stats.iloc[:3]['Up']==1).all() or (price_stats.iloc[:3]['Down']==1).all()):
                # x = np.linspace(0, 4, 10)
                # price_stats.iloc[:3].plot.kde(title = '{}'.format(portfolio[stock]['stock_data']['name']) ,xticks=x)
                # if (price_stats.iloc[3]>2).any() :
                # stock_frame = create_dataframe(portfolio,stock,int(timeframe/2),70)
                # price_stats = stocks_stat(stock_frame['Derivatives'])
                # Plot Option for manual analysis
                temp_df = pd.DataFrame.copy(df_price_frame)
                temp_df_vol = pd.DataFrame.copy(df_volume)
                corr = temp_df.corrwith(temp_df_vol)
                temp_df.set_index(['DateTime'], inplace=True)
                ax = temp_df[['Price_Norm']].plot(
                    title='{}\n Price vs. Volume\n'.format(portfolio[stock]['stock_data']['name'], temp_df[['Price_Norm']]))
                temp_df_vol.set_index(['DateTime'], inplace=True)
                temp_df_vol[['Volume_Norm']].plot(ax=ax)
                temp_df['Volume_Norm'] = temp_df_vol['Volume_Norm']
                print('Stock {}\n '.format(stock), corr)
            days_list = sorted(list(portfolio[stock]['stock_data']['6M'].keys()))
            hour_list = sorted(list(portfolio[stock]['stock_data']['1M'].keys()))
            date_only_dict = {}

            # Creates Dict - each day with a 30 min delta (except 16:00, closing price)
            for hour in hour_list:
                hour_date = parser.parse(hour)
                date_only = hour_date.strftime("%Y-%m-%d")
                if date_only not in date_only_dict:
                    hours = []
                    hours.append(hour)
                    date_only_dict[date_only] = hours
                else:
                    hours.append(hour)
                    date_only_dict[date_only] = hours
            up_list, down_list, hour_list = [], [], []
            # Creates Down and Up Lists
            for day in date_only_dict:
                if portfolio[stock]['stock_data']['6M'].get(day + ' 16:00:00', None):
                    if portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('close', None) and \
                            portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('open', None):
                        if portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('close', None) > \
                                portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('open', None):
                            up_list.append(day)
                        elif portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('close', None) < \
                                portfolio[stock]['stock_data']['6M'][day + ' 16:00:00'].get('open', None):
                            down_list.append(day)

            # Creates Correlation Dictionary
            corr_dict_up = correlate_dict(portfolio, stock, date_only_dict, up_list)
            corr_dict_down = correlate_dict(portfolio, stock, date_only_dict, down_list)

            # Adds dicts to Portfolio file
            portfolio[stock]['stock_data']['hourly_correlation'] = {'up': corr_dict_up}
            portfolio[stock]['stock_data']['hourly_correlation']['down'] = corr_dict_down

            # Merge the DataFrames
            # Sets DateTime as the columns
            # df_price_frame = df_price_frame.T
            # df_price_frame.columns = df_price_frame.iloc[0]
            # df_price_frame.drop('DateTime', axis=0, inplace=True)
            # df_volume = df_volume.T
            # df_volume.columns = df_volume.iloc[0]
            # df_volume.drop('DateTime', axis=0, inplace=True)
            #df_merged = pd.concat([df_price_frame,df_volume], ignore_index=True)

            # Adds DataFrames to Portfolio file
            portfolio[stock]['stock_data']['stock_DataFrame'] = {'3 Month Price DataFrame': df_price_frame, '1 Month Price DataFrame':df_price_month,'3 Months Volume DataFrame':df_volume}


            def test_xgb_model(df):
                y = df['Price'].to_numpy().reshape(-1, 1)
                y = y[:-1, :]
                X = df.iloc[:-1,:].drop(columns=['Price','DateTime','Price_Norm'])
                X_cols = X.columns.tolist()
                model = train_XGB(X,y, X_cols)


            test_xgb_model(df_price_frame)
            test_xgb_model(df_5d_price)
        except Exception as e:
            exceptions_list += [f"{stock} Forecasting Exception" + str(e)]
            continue
    for exception in exceptions_list:
        print(exception)
