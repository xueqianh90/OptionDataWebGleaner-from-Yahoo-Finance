

import pandas as pd
import numpy as np
import time
from selenium import webdriver
from bs4 import BeautifulSoup
import os
from datetime import date
from scipy import stats
from math import log, sqrt, exp
import datetime
from scipy.optimize import brentq
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import random
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import matplotlib.pylab as pylab


########## data crawler ##########
def data_crawler(stock):

    chromedriver = "/Users/xueqianhuang/Desktop/5. Financial Programming/project3/chromedriver"

    os.environ["webdriver.chrome.driver"] = chromedriver

    driver = webdriver.Chrome(chromedriver)

    maturity_dates = [date(2018,12,14),date(2018,12,21),date(2018,12,28),date(2019,1,4),date(2019,1,11),
    date(2019,1,18),date(2019,1,25),date(2019,2,15),date(2019,3,15),date(2019,4,18),
    date(2019,6,21),date(2019,7,19),date(2020,1,17),date(2020,6,19),date(2021,1,15)]

    dates = [1544745600,1545350400,1545955200,1546560000,1547164800,1547769600,1548374400,1550188800,
    1552608000,1555545600,1561075200,1563494400,1579219200,1592524800,1610668800]

    df_dict = {}

    try: 
            
        for d, maturity in zip(dates, maturity_dates):
            url = 'http://finance.yahoo.com/quote/' + stock + '/options?date=' + str(d)
            ## Crawl data
            driver.get(url)
            html_source = driver.page_source
            # print(html_source)
            ## Beautifulsoup
            soup = BeautifulSoup(html_source, 'html.parser')

            # tables = soup.select('table')


            if soup.find('table', 'calls') is not None:
                
                print()
                print('Be patient...')
                print('The call data of ' + stock + ' for maturity date ' + str(maturity) +' is being crawling...')

                stock_price = [float(i.text) for i in soup.findAll('span', 'Fz(36px)')]
                title = [i.text for i in soup.find('table', 'calls').find_all('th')]
                # print(title)
                rows = [row for row in soup.find('table', 'calls').find_all("tr")]
                # print(rows)
                text = [i.text for i in soup.find('table', 'calls').find_all('td')]
                # print(text)     

                l_table = len(rows) - 1
                
                dictionary = {}
                # dictionary['OptionType'] = ["call"] * l_table
                dictionary['maturity_date'] = [maturity] * l_table
                dictionary['date'] = [today] * l_table
                dictionary['stock_price'] = stock_price * l_table

                for j in range(11):
                    key = title[j]
                    dictionary[key] = []
                    for i in range(l_table):
                        dictionary[key].append(text[11 * i + j])

                df_call = pd.DataFrame(dictionary)
                # print(df_call.head(5))
                df_call.insert(0, 'OptionType', 'call')

                stock_refined = ''.join(ch for ch in stock if (ch != '.') and (ch != '-'))

            if soup.find('table', 'puts') is not None:

                print()
                print('Be patient...')
                print('The put data of ' + stock + ' for maturity date ' + str(maturity) +' is being crawling...')

                stock_price = [float(i.text) for i in soup.findAll('span', 'Fz(36px)')]
                title = [i.text for i in soup.find('table', 'puts').find_all('th')]
                # print(title)
                rows = [row for row in soup.find('table', 'puts').find_all("tr")]
                # print(rows)
                text = [i.text for i in soup.find('table', 'puts').find_all('td')]
                # print(text)     

                l_table = len(rows) - 1
                
                dictionary = {}
                # dictionary['OptionType'] = ["call"] * l_table
                dictionary['maturity_date'] = [maturity] * l_table
                dictionary['date'] = [today] * l_table
                dictionary['stock_price'] = stock_price * l_table

                for j in range(11):
                    key = title[j]
                    dictionary[key] = []
                    for i in range(l_table):
                        dictionary[key].append(text[11 * i + j])

                df_put = pd.DataFrame(dictionary)
                # print(df_call.head(5))
                df_put.insert(0, 'OptionType', 'put')

                stock_refined = ''.join(ch for ch in stock if (ch != '.') and (ch != '-'))

                df_concat = pd.concat([df_call, df_put])

                if stock_refined not in df_dict.keys():
                    df_dict[stock_refined] = df_concat
                else:
                    df_dict[stock_refined] = pd.concat([df_dict[stock_refined], df_concat], ignore_index=True)
                

        os.chdir("/Users/xueqianhuang/Desktop/5. Financial Programming/project3")

        df_dict[stock_refined].to_csv(stock + '.csv', index = False)
        
        print()
        print('The crawling data file is saved into current working directory.')
        print('--> The original datas are colloected into '+ stock + '.csv')

    except:
        print(stock," is currently not in market")


########## data cleaner ##########
def clean_data(dataframe, stock):
    
    atts = ['Strike', '% Change', 'Volume', 'Open Interest', 'Implied Volatility']

    for att in atts:   

        for i in range(len(dataframe[att])):

            dataframe[att][i] = str(dataframe[att][i]).replace("%", "")
            dataframe[att][i] = str(dataframe[att][i]).replace(",", "")
            dataframe[att][i] = str(dataframe[att][i]).replace("-", "0")
            dataframe[att][i] = float(dataframe[att][i])

    dataframe['Implied Volatility'] = dataframe['Implied Volatility']/100
    dataframe = dataframe.loc[dataframe['Implied Volatility'] < 2]
    
    os.chdir("/Users/xueqianhuang/Desktop/5. Financial Programming/project3")
    dataframe.to_csv(stock + '_clean.csv', index = False)
    print('The data files are saved into current working directory.')
    print('--> The cleaned dataset are colloected into '+ stock + '_clean.csv')
    
    return dataframe


######## data visualization ########
def visualization(year_data, stock):
    
    for i in range(year_data.shape[0]):
    
        maturity = year_data['maturity_date'][i]
        m_date = maturity.split('-')
        year_data['maturity_date'][i] = m_date[0]+m_date[1]+m_date[2]

    year_data['B-A'] = abs(year_data['Bid'] - year_data['Ask'])

    year_data.drop(['date', 'Contract Name', 'Last Trade Date'], axis=1, inplace=True)
    
    for att in year_data:
        print(year_data[att].describe())
    
    call = year_data.loc[year_data['OptionType']=='call']
    put = year_data.loc[year_data['OptionType']=='put']

    options = [call, put]
    option_names = ['call', 'put']

    for i in range(len(options)):

        fig=plt.figure(figsize=(10,5))
        ax=Axes3D(fig)
        ax.scatter(options[i]['Strike'],options[i]['Volume'],options[i]['Implied Volatility'],alpha=0.3)

        ax.set_xlabel('Strike')
        ax.set_ylabel('Volume')
        ax.set_zlabel('Implied Volatility')

        ax.set_title('The Implied Volatility of '+stock+': '+option_names[i])
        plt.show()

        grouped = options[i].groupby('maturity_date')
        cols = ['Strike', 'Last Price', 'Volume', 'Implied Volatility']
        for col in cols:
            df = grouped[col].agg([np.mean, np.std])
            tit = col+' of ' + stock + ': ' +  option_names[i]
            df.plot(title=tit)


########## data separator ##########
def import_file(stock):

    df = pd.read_csv(str(stock)+'_clean.csv')

    return df


def bsm_pricing(S, K, T, r, sigma, option_type):

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type.lower() == 'call':
        n_d1 = stats.norm.cdf(d1, 0.0, 1.0)
        n_d2 = stats.norm.cdf(d2, 0.0, 1.0)
        call_price = (S * n_d1 - K * exp(-r * T) * n_d2)

        return call_price

    elif option_type.lower() == 'put':
        n_d1 = stats.norm.cdf(-d1, 0.0, 1.0)
        n_d2 = stats.norm.cdf(-d2, 0.0, 1.0)
        put_price = K * exp(-r * T) * n_d2 - S * n_d1

        return put_price


def seperate_data(stock, cutoff):

    df = import_file(stock)

    eo_bre_list=[]
    est_imp_vol_list = []
    
    for i in range(df.shape[0]):

        S = df['stock_price'][i]
        K = df['Strike'][i]

        date = df['date'][i]
        today = date.split('-')
        # print(today)
        dt_now = datetime.date(int(today[0]), int(today[1]), int(today[2]))

        maturity = df['maturity_date'][i]
        m_date = maturity.split('-')
        dt_m = datetime.date(int(m_date[0]), int(m_date[1]), int(m_date[2]))

        T = (dt_m - dt_now).days / 365

        # T = df["Expiration time"][i]
        r = 0.03

        #sigma = 
        optionType = df['OptionType'][i]
        market_price = df['Last Price'][i]

        f = lambda calc_sigma: bsm_pricing(S, K, T, r, calc_sigma, optionType) - market_price
    
        a = -df['Implied Volatility'].max()
        b = df['Implied Volatility'].max()
        
        try:
            brentq(f, a, b, full_output = True)[0]
        except:
            df.loc[i,'European_option'] = 0
        else:
            est_imp_vol = brentq(f, a, b, full_output = True)[0]
            est_imp_vol_list.append(est_imp_vol)
            # print(est_imp_vol)
            imp_vol_diff = est_imp_vol - df['Implied Volatility'][i]
        
            if abs(imp_vol_diff) < cutoff:
                df.loc[i,'European_option'] = 1
                eo_bre_list.append(i)
            else:
                df.loc[i,'European_option'] = 0

    df_eur = df[df[ 'European_option'] == 1]
    df_no_eur = df[df[ 'European_option'] == 0]

    print('These are the indexes of European Options:\n', eo_bre_list)

    df_eur.to_csv(stock+'_clean_euro_opt.csv', index = False)
    df_no_eur.to_csv(stock+'_clean_non_euro_opt.csv', index = False)

    print('The data files are saved into current working directory.')
    print('--> European options are colloected into '+ stock+'_clean_euro_opt.csv')
    print('--> Non-European options are colloected into '+ stock+'_clean_non_euro_opt.csv')


############ Graphic Interface ############
def GraphicInterface(category, num):

    s_date = '10/1/2008'
    e_date = '11/15/2018'

    # Retrieve Auto industry: 'GM' , 'F' , 'TM' , 'TSLA’ ,'HMC'
    gm = web.DataReader('GM', data_source='yahoo', start=s_date, end=e_date)
    f = web.DataReader('F', data_source='yahoo', start=s_date, end=e_date)
    tm = web.DataReader('TM', data_source='yahoo', start=s_date, end=e_date)
    tsla = web.DataReader('TSLA', data_source='yahoo', start=s_date, end=e_date)
    hmc = web.DataReader('HMC', data_source='yahoo', start=s_date, end=e_date)

    # Retrieve Bank industry: 'JPM' , 'BAC' , 'HSBC' , 'C’ (Citi group) 'GS’ (Goldman Sach)
    jpm = web.DataReader('JPM', data_source='yahoo', start=s_date, end=e_date)
    bac = web.DataReader('BAC', data_source='yahoo', start=s_date, end=e_date)
    hsbc = web.DataReader('HSBC', data_source='yahoo', start=s_date, end=e_date)
    c = web.DataReader('C', data_source='yahoo', start=s_date, end=e_date)
    gs = web.DataReader('GS', data_source='yahoo', start=s_date, end=e_date)

    # Retrieve Retail industry: ‘WMT’, ‘TGT’, ‘JCP’, ‘HD’,'COST'
    wmt = web.DataReader('WMT', data_source='yahoo', start=s_date, end=e_date)
    tgt = web.DataReader('TGT', data_source='yahoo', start=s_date, end=e_date)
    jcp = web.DataReader('JCP', data_source='yahoo', start=s_date, end=e_date)
    hd = web.DataReader('HD', data_source='yahoo', start=s_date, end=e_date)
    cost = web.DataReader('COST', data_source='yahoo', start=s_date, end=e_date)

    # Retrieve IT industry: 'AAPL','MSFT','AMZN','GOOG','FB','intc'
    aapl = web.DataReader('AAPL', data_source='yahoo', start=s_date, end=e_date)
    msft = web.DataReader('MSFT', data_source='yahoo', start=s_date, end=e_date)
    amzn = web.DataReader('AMZN', data_source='yahoo', start=s_date, end=e_date)
    goog = web.DataReader('GOOG', data_source='yahoo', start=s_date, end=e_date)
    fb = web.DataReader('FB', data_source='yahoo', start=s_date, end=e_date)
    intc = web.DataReader('INTC', data_source='yahoo', start=s_date, end=e_date)

    # Fashion FASHION industry: 'tpr', 'hmb', 'ges', 'mc', 'tif'
    tpr = web.DataReader('TPR', data_source='yahoo', start=s_date, end=e_date)
    hmb = web.DataReader('HM-B.ST', data_source='yahoo', start=s_date, end=e_date)
    ges = web.DataReader('GES', data_source='yahoo', start=s_date, end=e_date)
    mc = web.DataReader('MC', data_source='yahoo', start=s_date, end=e_date)
    tif = web.DataReader('TIF', data_source='yahoo', start=s_date, end=e_date)

    AUTO_name = ['GM', 'F', 'TM', 'TSLA', 'HMC']
    BANK_name = ['JPM', 'BAC', 'HSBC', 'C', 'GS']
    RETAIL_name = ['WMT', 'TGT', 'JCP', 'HD', 'COST']
    IT_name = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'intc']
    FASHION_name = ['tpr', 'hmb', 'ges', 'mc', 'tif']

    AUTO = [gm, f, tm, tsla, hmc]
    BANK = [jpm, bac, hsbc, c, gs]
    RETAIL = [wmt, tgt, jcp, hd, cost]
    IT = [aapl, msft, amzn, goog, intc]
    FASHION = [tpr, hmb, ges, mc, tif]
    
    stocks = []
    stocks_name = []
    
    for i in category:
        for d in range(num):
            if str(i) == 'A':
                random.shuffle(AUTO)
                stocks.append(AUTO[d])
                stocks_name.append((AUTO_name[d]))
            if str(i) == 'B':
                random.shuffle(BANK)
                stocks.append(BANK[d])
                stocks_name.append((BANK_name[d]))
            if str(i) == 'I':
                random.shuffle(IT)
                stocks.append(IT[d])
                stocks_name.append((IT_name[d]))
            if str(i) == 'R':
                random.shuffle(RETAIL)
                stocks.append(RETAIL[d])
                stocks_name.append((IT_name[d]))
            if str(i) == 'F':
                random.shuffle(FASHION)
                stocks.append(FASHION[d])
                stocks_name.append((IT_name[d]))
    
    # colors = ['y-', 'b-', 'g-', 'k-','r-','c-','m-']
    # fig = pylab.figure(figsize = (10,8))
    for i in range(len(stocks)):
        pylab.plot(stocks[i]['Adj Close'], linewidth=1.5)

    pylab.legend(stocks_name, loc='upper right', shadow=True)
    pylab.ylabel('Adjusted Close Price')
    pylab.title('Adjusted Close Price from 2008 to 2018')
    pylab.grid('on')
    pylab.show()


if __name__ == "__main__":
    
    ticker = [line.strip() for line in open('Yahoo_ticker_List.csv')]

    total_ticker = len(ticker)
    print('\nThere are', total_ticker, 'options in Yahoo_ticker_List.csv.')

    ########## data crawler ##########
    user_option2 = input('Which ticker do you want to choose? (e.g. AAPL) --> ')
    stock = str(user_option2)
    
    today = date.today()

    crawl_time_beg = time.time()
    data_crawler(stock)
    crawl_time_end = time.time()
    print('Used time for data crawling:', crawl_time_end - crawl_time_beg)

    ########## data cleaner ##########
    print('Data cleaning may take a while. Please wait...')
    df = pd.read_csv(stock + '.csv')
    clean_time_beg = time.time()
    clean_df = clean_data(df, stock)
    clean_time_end = time.time()
    print('Used time for data cleaning:', clean_time_end - clean_time_beg)

    ######## data visualization ########

    cleanedData = import_file(stock)
    visualization(cleanedData, stock)

    ########## data separator ##########
    now = datetime.datetime.now()

    user_option5 = input('How much cutoff do you want? (e.g. 0.01) --> ')
    cutoff = float(user_option5)

    sep_time_beg = time.time()
    seperate_data(stock, cutoff)
    sep_time_end = time.time()
    print("Used time for options separating:", sep_time_end - sep_time_beg)
    
    ############ Graphic Interface ############
    category = list(input("Please select one category from A(Auto), B(Bank),I(IT), R(Retail), F(Fashion):\n"))
    num = int(input("How many stocks you want to download in that category(s)? <= 5:\n"))
    GraphicInterface(category, num)