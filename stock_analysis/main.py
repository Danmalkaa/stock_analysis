import sys
from collections import defaultdict
import json
import re
import datetime
import os

from stock_analysis import create_stock_portfolio, create_stock_dataframe, filter_tipranks_accurate

def generate_my_portfolio(portfolio, stocks):
    # Gets input from user -  Test Purpose or Not?
    input_str = input("Test: Y/N ")
    if not re.match("^[ynYN]*$", input_str):
        print("Error! Only letters y/n allowed!")
        sys.exit()
    elif len(input_str) > 1:
        print("Error! Only 1 characters allowed!")
        sys.exit()
    is_dev_mode = True if input_str == ('y' or 'Y') else False
    if is_dev_mode:
        # Loads portfolio from file - Dev/ Test purpose
        f = open("22.4.json", "r")
        portfolio = json.load(f)
    else:
        # Save Portfolio to file
        portfolio = create_stock_portfolio.create_portfolio(stocks)
        json_file = json.dumps(portfolio)
        newpath = os.path.join(os.getcwd(),"stock_data")
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        os.chdir(newpath)
        str_today = str(datetime.datetime.today().date()).replace('-','_')
        f = open(str_today + ".json", "w")
        f.write(json_file)
        f.close()
    return portfolio, is_dev_mode


def main():
    alternative, portfolio = defaultdict(dict), defaultdict(dict)
    good_accuracy = {}
    # Write here your portfolio stocks - format: [stock symbol, stock amount] - Example : [['GOOGL', 0], ['SPWR', 0]]
    stocks = [['GOOGL', 5], ['SPWR', 1], ['XLE', 55], ['XLC', 39], ['XLF', 153], ['RYAAY', 54], ['UAL',163]]
    portfolio, is_dev_mode = generate_my_portfolio(portfolio, stocks)
    chg, better_dict, alt_stock_data = create_stock_portfolio.calc_performance(portfolio, is_dev_mode, 7)
    new_portfolio = create_stock_portfolio.create_alternative_portfolio(better_dict, portfolio, alt_stock_data)
    # new_chg = create_stock_portfolio.calc_performance(new_portfolio, is_dev_mode, 7, with_rival=False) # TODO: deprecate
    # Timeframe for Relative Calculation
    timeframe = 120
    is_plot = False
    create_stock_dataframe.create_stock_dataframe(portfolio, timeframe, is_plot)
    create_stock_dataframe.create_stock_dataframe(alt_stock_data, timeframe, is_plot)
    # Creates a dict of good Tiprank accuracy stocks sorted by score
    good_accuracy = filter_tipranks_accurate.create_alternative_portfolio(portfolio, alt_stock_data)
    return good_accuracy, portfolio, alt_stock_data


if __name__ == "__main__":
    main()
