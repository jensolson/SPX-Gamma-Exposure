# -*- coding: utf-8 -*-
"""
Jens Olson
jens.olson@gmail.com
"""
import pandas as pd
import numpy as np
import holidays
import datetime
import requests
from pyVolLib import blackIV, blackDelta, blackGamma,\
                     blackVega, blackTheta, blackScholesIV,\
                     blackScholesDelta, blackScholesGamma
                     
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def TRTH_GEX(raw):
    """
    Inputs:
        'raw': raw pandas dataframe output from TRTH with 'RIC', 'Trade Date', 'Open Interest',
               and 'Implied Volatility' fields
               
    Returns:
        pandas dataframe of estimated dealer gamma exposure by day
    """
    letterToMonth = {
        **dict.fromkeys(['a', 'm'], 1),
        **dict.fromkeys(['b', 'n'], 2),
        **dict.fromkeys(['c', 'o'], 3),
        **dict.fromkeys(['d', 'p'], 4),
        **dict.fromkeys(['e', 'q'], 5),
        **dict.fromkeys(['f', 'r'], 6),
        **dict.fromkeys(['g', 's'], 7),
        **dict.fromkeys(['h', 't'], 8),
        **dict.fromkeys(['i', 'u'], 9),
        **dict.fromkeys(['j', 'v'], 10),
        **dict.fromkeys(['k', 'w'], 11),
        **dict.fromkeys(['l', 'x'], 12),
    }

    letterToFlag = {
        **dict.fromkeys(list('abcdefghijkl'), 'c'),
        **dict.fromkeys(list('mnopqrstuvwx'), 'p'),
    }

    df = raw.copy(deep=True)
    df.set_index('Trade Date', drop=True, inplace=True)
    df.index = pd.to_datetime(df.index, infer_datetime_format=True)

    underlying = sorted(set(df['RIC']))[0]
    divisor = 10 if underlying in ['.SPX', 'SPXW'] else 100

    df['F'] = df[df['RIC'] == underlying]['Last']

    df = df[df['RIC'] != underlying]
    
    # Remove options with minimal open interest or with no bids
    df = df[(df['Open Interest'] > 10) &\
            (df['Bid'] > .05) 
           ].copy(deep=True)
    
    df['Mid'] = np.mean(df[['Bid', 'Ask']], axis=1)

    df['TRTH Tag'] = df['RIC'].str[-12]
    df = df[df['TRTH Tag'].notnull()]
    df['TRTH Tag'] = df['TRTH Tag'].str.lower()
    df = df[df['TRTH Tag'].isin(list('abcdefghijklmnopqrstuvwx'))]
    df['Month'] = df['TRTH Tag'].apply(lambda x: letterToMonth[x])

    # Retrieve day and year from TRTH RIC tag
    df['Day'] = pd.to_numeric(df['RIC'].str[-11:-9], downcast='signed')
    df['Year'] = pd.to_numeric(df['RIC'].str[-9:-7], downcast='signed')+2000

    df['Expiry'] = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day),
                                  infer_datetime_format=True)
    
    us_holidays = holidays.UnitedStates(years=list(range(2000, 2030)))
    us_hlist = list(us_holidays.keys())

    A = [d.date() for d in df['Expiry']]
    B = [d.date() for d in df.index]
    df['BDTE'] = np.busday_count(B, A, weekmask='1111100', holidays=us_hlist)
    
    df = df[df['BDTE'] >= 1].copy(deep=True)
    df['Flag'] = df['TRTH Tag'].apply(lambda x: letterToFlag[x])

    # Retrieve strike price from TRTH RIC tag
    df['Strike'] = pd.to_numeric(df['RIC'].str[-7:-2])/divisor

    if underlying in ['.SPX', 'SPXW']:
        df['IV'] = df.apply(blackIV, axis=1)       
        df['Delta'] = df.apply(blackDelta, axis=1)
        df['Gamma'] = df.apply(blackGamma, axis=1)
        
    else:
        df['IV'] = df.apply(blackScholesIV, axis=1)
        df['Delta'] = df.apply(blackScholesDelta, axis=1)
        df['Gamma'] = df.apply(blackScholesGamma, axis=1)
        
    df = df[(df['IV'] > .01) & (df['IV'] < 2.) &\
            (np.abs(df['Delta']) < .95)
           ].copy(deep=True)
        
    df['GEX'] = 10**-6*(-100*(df['Flag'] == 'p')*df['Open Interest']*df['Gamma']*df['F']\
                        +100*(df['Flag'] == 'c')*df['Open Interest']*df['Gamma']*df['F'])
    
    if underlying in ['SPY', 'GLD', 'TLT']: df['GEX']/=10

    df1 = df.pivot_table(values='GEX', index=df.index, aggfunc=np.sum)
    del df1.index.name
    return df1

def CBOE_GEX(filename, sens=True, plot=False, occ=False):
    """
    Calculates dealer gamma exposure from latest CBOE option open interest data at
    http://www.cboe.com/delayedquote/quote-table-download
        
    Parameters:
        filename: string referencing path to local drive. Should be something like 'quotedata.dat'
        sens: boolean; returns sensitivity if true, spot value if false
        plot: boolean; returns plot if True, pandas series if False
        occ: boolean; use Options Clearing Corporation open interest data if True, pull from CBOE file if False
    """
    # Extract top rows of dataframe for latest spot price and date
    raw = pd.read_table(filename)
    spotF = float(raw.columns[0].split(',')[-2]) 
    ticker = raw.columns[0].split(',')[0][1:4] 
    rf = .02
    pltDate = raw.loc[0][0].split(',')[0][:11] 
    pltTime = raw.loc[0][0].split(',')[0][-8:] 
    dtDate = datetime.datetime.strptime(pltDate, '%b %d %Y').date()

    # Extract dataframe for analysis
    raw = pd.read_table(filename, sep=',', header=2)
    c = raw.loc[:, :'Strike'].copy(deep=True)
    c.columns = c.columns.str.replace('Calls', 'ID')
    p = (raw.loc[:, 'Strike':].join(raw.loc[:, 'Expiration Date']))\
                              .copy(deep=True)
                              
    p.columns = p.columns.str.replace('Puts', 'ID')
    p.columns = p.columns.str.replace('.1', '')
    p = p[c.columns]

    c['Flag'] = 'c'
    p['Flag'] = 'p'
    
    c['Expiry'] = pd.to_datetime(c['Expiration Date'],
                                 infer_datetime_format=True)
    p['Expiry'] = pd.to_datetime(p['Expiration Date'],
                                 infer_datetime_format=True)
    
    if occ:
        def getOCC(symbol):
            url = 'https://www.theocc.com/webapps/series-search'
            s = requests.Session()
            r = s.post(url, data={'symbolType': 'U', 'symbolId': symbol})
            df = pd.read_html(r.content)[0]
            df.columns = df.columns.droplevel()
            df1 = df[df['Product Symbol'].isin(['SPX', 'SPXW'])]\
                                         .copy(deep=True)
            df1.reset_index(drop=True, inplace=True)
            df1.rename(columns={'Integer': 'Strike',
                                'Product Symbol': 'Symbol'}, inplace=True)
            
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthNums = list(range(1, 13))
            monthToNum = dict(zip(months, monthNums))
            
            df1['Month'] = df1['Month'].apply(lambda x: monthToNum[x])
            df1['Expiry'] = pd.to_datetime(dict(year=df1.Year,
                                                month=df1.Month,
                                                day=df1.Day),
                                           infer_datetime_format=True)
            return df1
        
        print('Getting open interest data from OCC...')
        df1 = getOCC('SPX')
    
        def idToSymbol(strng):
            if strng[3] == 'W': return strng[:4]
            else: return strng[:3]
            
        c['Symbol'] = c['ID'].apply(idToSymbol)
        p['Symbol'] = p['ID'].apply(idToSymbol)
        
        c1 = pd.merge(c,
                      df1.loc[:, ['Symbol', 'Expiry', 'Strike', 'Call']],
                      how='left',
                      on=['Symbol', 'Expiry', 'Strike'])
        
        p1 = pd.merge(p,
                      df1.loc[:, ['Symbol', 'Expiry', 'Strike', 'Put']],
                      how='left',
                      on=['Symbol', 'Expiry', 'Strike'])
        
        c1.drop(['Open Int'], axis=1, inplace=True)
        p1.drop(['Open Int'], axis=1, inplace=True)
        c1.rename(columns={'Call': 'Open Int'}, inplace=True)
        p1.rename(columns={'Put': 'Open Int'}, inplace=True)
        
        df = c1.append(p1, ignore_index=True)
        
    else:
        df = c.append(p, ignore_index=True)
    
    df = df[(df['ID'].str[-3] != '-') &\
            (df['ID'].str[-4] != '-')
            ].copy(deep=True)

    for item in ['Bid', 'Ask', 'Last Sale', 'IV', 'Delta',
                 'Gamma', 'Open Int', 'Strike']:
        df[item] = pd.to_numeric(df[item], errors='coerce')

    us_holidays = holidays.UnitedStates(years=list(range(2000, 2030)))
    us_hlist = list(us_holidays.keys())

    A = [d.date() for d in df['Expiry']]
    df['BDTE'] = np.busday_count(dtDate, A, weekmask='1111100',
                                 holidays=us_hlist)

    df = df.loc[(df['Open Int'] > 10) &\
                (df['Bid'] > .05) &\
                (df['BDTE'] >= 1) #&\
               ].copy(deep=True)

    df['Mid'] = np.mean(df[['Bid', 'Ask']], axis=1)
    
    print('Calculating Greeks...')
    df['IV'] = df.apply(lambda x: blackIV(x, F=spotF, rf=rf), axis=1)
    df = df[(df['IV'] > .01) & (df['IV'] < 2.)].copy(deep=True)
    
    df['Delta'] = df.apply(lambda x: blackDelta(x, F=spotF, rf=rf), axis=1)
    df = df[np.abs(df['Delta'])<.95].copy(deep=True)

    if sens:
        increment = 10 if ticker in ['SPX', 'NDX'] else 1
        nPoints = 20
        Fs = list((np.linspace(start=spotF, 
                               stop=spotF+increment*nPoints,
                               num=nPoints,
                               endpoint=False)-increment*nPoints//2).astype(int))

        for F in Fs:
            df[str(F)+'_g'] = df.apply(lambda x: blackGamma(x, F=F, rf=rf),
                                       axis=1)

        for F in Fs:
            df[str(F)+'_GEX'] = 10**-6*(100*F*(df['Flag']=='c')*df[str(F)+'_g']*df['Open Int']\
                                       -100*F*(df['Flag']=='p')*df[str(F)+'_g']*df['Open Int'])

        GEXs = [(0.1 if ticker not in ['SPX', 'NDX'] else 1)*np.sum(df[str(F)+'_GEX'], axis=0) for F in Fs]
        s = pd.Series(dict(zip(Fs, GEXs))).astype(int)
    
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.plot(s, color='xkcd:red')
            fig.suptitle(ticker+' Dealer Gamma Exposure per Index Point\n (=0.1 ETF pt) as of '+pltDate+' '+pltTime,
                         fontsize=12, weight='bold')
            ax.set_ylabel('Dealer Gamma in $mm')
            ax.yaxis.set_major_formatter(plt.FuncFormatter('{:,.0f}'.format))
            ax.axvline(x=spotF, color='xkcd:deep blue', linestyle=':')
            zeroGEX = int(np.interp(x=0, xp=s.values, fp=s.index))
            ax.axvline(x=zeroGEX, color='xkcd:black', linestyle='--')
            ax.legend(labels=['SPX', 'Last', 'Zero GEX: '+str(zeroGEX)])
            plt.xticks(rotation=30)
            plt.show()

        else: return s
        
    else:
        df['Gamma'] = df.apply(lambda x: blackGamma(x, F=spotF, rf=rf), axis=1)
        gams = 10**-6*(100*spotF*(df['Flag']=='c')*df['Gamma']*df['Open Int']\
                      -100*spotF*(df['Flag']=='p')*df['Gamma']*df['Open Int'])
        gam = (1 if ticker in ['SPX', 'NDX'] else 0.1)*np.sum(gams, axis=0)
        return gam
    
def CBOE_Greeks(filename, low, high, incr, expiry, field):
    """
    Parameters:
        filename: string referencing path to local drive. Should be something like 'quotedata.dat'
        low, high, incr: integers describing low and high end of plot range, with increment
        expiry: target option expiry in YYYY-MM-DD format
        field: 'IV', 'Delta', 'Gamma', 'Vega', 'Gamma/Theta', 'Vega/Theta', or 'Theta/Mid'
    
    Returns:
        Plot of option Greeks by strike
    """
    fields = ['IV', 'Delta', 'Gamma', 'Vega', 'Theta', 'Gamma/Theta', 'Vega/Theta', 'Theta/Mid']
    fltDigs = ['{:,.'+str(x)+'f}' for x in [3, 2, 4, 2, 2, 4, 2, 4]]
    fltDigDict = dict(zip(fields, fltDigs))

    raw = pd.read_table(filename)
    spotF = float(raw.columns[0].split(',')[1])
    rf = .02
    pltDate = raw.loc[0][0][:11]
    pltTime = raw.loc[0][0][14:22]
    dtDate = datetime.datetime.strptime(pltDate, '%b %d %Y').date()

    # Extract dataframe for analysis
    raw = pd.read_table(filename, sep=',', header=2)
    c = raw.loc[:, :'Strike'].copy(deep=True)
    c.columns = c.columns.str.replace('Calls', 'ID')
    p = raw.loc[:, ['Expiration Date', 'Strike', 'Puts', 'Last Sale.1',
                    'Net.1', 'Bid.1', 'Ask.1', 'Vol.1', 'IV.1',
                    'Delta.1', 'Gamma.1', 'Open Int.1']].copy(deep=True)

    p.columns = p.columns.str.replace('Puts', 'ID')
    p.columns = p.columns.str.replace('.1', '')

    c['Flag'] = 'c'
    p['Flag'] = 'p'

    df = c.append(p, ignore_index=True, sort=True)

    df = df[(df['ID'].str[-3] != '-') &\
            (df['ID'].str[-4] != '-')
            ].copy(deep=True)

    for item in ['Bid', 'Ask', 'Last Sale']:
        df[item] = pd.to_numeric(df[item], errors='coerce')
        
    df['Expiry'] = pd.to_datetime(df['Expiration Date'], infer_datetime_format=True)

    us_holidays = holidays.UnitedStates(years=list(range(2000, 2030)))
    us_hlist = list(us_holidays.keys())

    A = [d.date() for d in df['Expiry']]
    df['BDTE'] = np.busday_count(dtDate, A, weekmask='1111100', holidays=us_hlist)
    
    df = df.loc[(df['Open Int'] > 10) &\
                (df['Bid'] > .05) &\
                (df['BDTE'] >= 1) 
                ].copy(deep=True)

    df['Mid'] = np.mean(df[['Bid', 'Ask']], axis=1)    
    
    df['IV'] = df.apply(lambda x: blackIV(x, F=spotF, rf=rf), axis=1)
    df['Delta'] = df.apply(lambda x: blackDelta(x, F=spotF, rf=rf), axis=1)
    df = df[np.abs(df['Delta'])<.9].copy(deep=True)
    
    if field in ['Gamma', 'Gamma/Theta']:
        df['Gamma'] = df.apply(lambda x: blackGamma(x, F=spotF, rf=rf), axis=1)

    if field in ['Vega', 'Vega/Theta']:
        df['Vega'] = df.apply(lambda x: blackVega(x, F=spotF, rf=rf), axis=1)

    if field in ['Theta', 'Gamma/Theta', 'Vega/Theta', 'Theta/Mid']:
        df['Theta'] = df.apply(lambda x: blackTheta(x, F=spotF, rf=rf), axis=1)

    if field == 'Gamma/Theta':
        df['Gamma/Theta'] = -df['Gamma']/df['Theta']

    if field == 'Vega/Theta':
        df['Vega/Theta'] = -df['Vega']/df['Theta']

    if field == 'Theta/Mid':
        df['Theta/Mid'] = df['Theta']/df['Mid']
        
    pGreeks = df[(df['Expiry']==expiry) &\
                 (df['Strike'].isin(range(low, high, incr))) &\
                 (df['Flag'] == 'p')
                ].groupby('Strike', axis=0).mean()[field]

    cGreeks = df[(df['Expiry']==expiry) &\
                 (df['Strike'].isin(range(low, high, incr))) &\
                 (df['Flag'] == 'c')
                ].groupby('Strike', axis=0).mean()[field]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(pGreeks, color='xkcd:red')
    ax.plot(cGreeks, color='xkcd:dark green')
    fig.suptitle('SPX '+expiry+' Expiry\n'+field+' by Strike as of '+pltDate+' '+pltTime,
                 fontsize=12, weight='bold')
    ax.set_ylabel(field)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fltDigDict[field].format))
    ax.axvline(x=spotF, color='xkcd:deep blue', linestyle=':')
    ax.legend(labels=['Puts', 'Calls', 'Last'])
    plt.xticks(rotation=30)
    plt.show()
