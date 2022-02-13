The data in this folder was pulled from the open source G-Research Crypto Comptetition. 

It contains information for the following assets; 
- Bitcoin Cash
- Binance Coin
- Bitcoin
- EOS.IO
- Ethereum Classic
- Ethereum
- Litecoin
- Monero
- TRON
- Stellar
- Cardano
- IOTA
- Maker
- Dogecoin 


The Data in the training sets follow this schema:
- timestamp: A timestamp for the minute covered by the row.
- Asset_ID: An ID code for the cryptoasset.
- Count: The number of trades that took place this minute.
- Open: The USD price at the beginning of the minute.
- High: The highest USD price during the minute.
- Low: The lowest USD price during the minute.
- Close: The USD price at the end of the minute.
- Volume: The number of cryptoasset units traded during the minute.
- VWAP: The volume weighted average price for the minute.
- Target: 15 minute residualized returns. See the 'Prediction and Evaluation' section of this notebook for details of how the target is calculated.



