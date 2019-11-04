"""
Jens Olson
jens.olson@gmail.com
"""
import numpy as np
from py_vollib.black.implied_volatility import implied_volatility
from py_vollib.black.greeks.numerical import delta, gamma, theta, vega
from py_vollib.black_scholes.implied_volatility import implied_volatility as bs_IV
from py_vollib.black_scholes.greeks.numerical import delta as bs_delta
from py_vollib.black_scholes_merton.greeks.numerical import gamma as bsm_gamma

# Black calcs for futures
def blackIV(df, F=None, rf=None):
    if F is None: F = df['F']
    if rf is None: rf = .02
    try:
        iv = implied_volatility(discounted_option_price=df['Mid'], # change to 'Mid'
                                F=F,
                                K=df['Strike'],
                                r=rf,
                                t=df['BDTE']/252,
                                flag=df['Flag'])
    except:
        iv = np.nan
    return iv

def blackDelta(df, F=None, rf=None):
    if F is None: F = df['F']
    if rf is None: rf = .02
    try:
        delt = delta(flag=df['Flag'],
                     F=F,
                     K=df['Strike'],
                     t=df['BDTE']/252,
                     r=rf,
                     sigma=df['IV'])
    except:
        delt = np.nan
    return delt

def blackGamma(df, F=None, rf=None):
    if F is None: F = df['F']
    if rf is None: rf = .02
    try:
        gam = gamma(flag=df['Flag'],
                    F=F,
                    K=df['Strike'],
                    t=df['BDTE']/252,
                    r=rf,
                    sigma=df['IV'])
    except:
        gam = np.nan
    return gam

def blackVega(df, F=None, rf=None):
    if F is None: F = df['F']
    if rf is None: rf = .02
    try:
        veg = vega(flag=df['Flag'],
                   F=F,
                   K=df['Strike'],
                   t=df['BDTE']/252,
                   r=rf,
                   sigma=df['IV'])
    except:
        veg = np.nan
    return veg

def blackTheta(df, F=None, rf=None):
    if F is None: F = df['F']
    if rf is None: rf = .02
    try:
        thet = theta(flag=df['Flag'],
                     F=F,
                     K=df['Strike'],
                     t=df['BDTE']/252,
                     r=rf,
                     sigma=df['IV'])
    except:
        thet = np.nan
    return thet

# Black Scholes calcs for equities
def blackScholesIV(df, F=None, rf=None):
    if F is None: F = df['F']
    if rf is None: rf = .02
    try:
        iv = bs_IV(price=df['Mid'], # change to 'Mid'
                   S=F,
                   K=df['Strike'],
                   r=rf,
                   t=df['BDTE']/252,
                   flag=df['Flag'])
    except:
        iv = np.nan
    return iv

def blackScholesDelta(df, F=None, rf=None):
    if F is None: F = df['F']
    if rf is None: rf = .02
    try:
        delt = bs_delta(flag=df['Flag'],
                        S=F,
                        K=df['Strike'],
                        t=df['BDTE']/252,
                        r=rf,
                        sigma=df['IV'])
    except:
        delt = np.nan
    return delt

def blackScholesGamma(df, F=None, rf=None):
    if F is None: F = df['F']
    if rf is None: rf = .02
    try:
        gam = bsm_gamma(flag=df['Flag'],
                        S=F,
                        K=df['Strike'],
                        t=df['BDTE']/252,
                        r=rf,
                        sigma=df['IV'],
                        q=.015)
    except:
        gam = np.nan
    return gam
