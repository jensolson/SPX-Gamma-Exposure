# SPX Gamma Exposure
Calculates estimate of market maker gamma exposure derived from S&amp;P 500 index options

Dependencies: pandas, numpy, holidays, datetime, requests, pyVolLib, matplotlib

* "TRTH_GEX" requires a pandas dataframe of output from Thomson Reuters's Tick History, listing end of day SPX option quotes. Pandas dataframe must contain "RIC", "Trade Date", "Open Interest", and "Implied Volatility" fields to calculate time series of estimated daily market maker gamma exposure

* "CBOE_GEX" is the simplest to use. Go to http://www.cboe.com/delayedquote/quote-table-download and enter "SPX" in the ticker box. You will download a .dat file. Set the variable "filename" equal to the file path to that download on your local drive. The function then outputs estimated spot market maker SPX gamma exposure with an optional sensitivity table

[SPX Gamma Exposure](gex.png)

* "CBOE_Greeks" returns a plot of Black-Scholes option Greeks by option strike. Uses the same "filename" variable as above, to come from the CBOE website


