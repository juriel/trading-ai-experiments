import argparse
from datetime import datetime, timedelta
import mplfinance as mpf
import pandas as pd
import yfinance as yf
import talib
from keras.layers import Dense, Dropout,LSTM
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

WINDOW = 20
WIN_OBJECTIVE      =  0.05
LOSS_OBJECTIVE     =  0.05
SPLIT_TRAIN        =  0.80
SPLIT_TEST         =  0.1
EPOCHS             =  1000
STOCK = None

# Function to validate date format
def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a valid date: '{s}'. Use YYYY-MM-DD format.")

# Function to fetch stock data
def fetch_stock_data(stock, start_date, end_date):
    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for stock: {stock} between {start_date} and {end_date}")
    return data

# Function to plot the stock data
def plot_stock_data(stock, data):
    ap = [mpf.make_addplot(data['RSI-14'], panel=2, color='blue', ylabel='RSI-14')]
    #mpf.plot(data, type='candle', volume=True, title=f"{stock} Stock Price", style='yahoo')
    mpf.plot(data, type='candle', volume=True, title=f"{stock} Stock Price", style='yahoo', addplot=ap, panel_ratios=(3,1))



def add_indicators_to_data(data):

    columns_to_delete = []
    rsi_periods = [4,8,12,14]
    for period in rsi_periods:
        data["RSI-"+str(period)] = talib.RSI(data['Adj Close'], timeperiod=period) /100.0
    ma_periods = [4,6,8,10,12,14]
    for period in ma_periods:
        data['EMA_'+str(period)] = talib.EMA(data['Adj Close'], timeperiod=period)
        data['EMA_'+str(period)+"-Close/Close"] = (data['EMA_'+str(period)] - data["Adj Close"])/data["Adj Close"] * 100.0
        columns_to_delete.append('EMA_'+str(period))

        data['SMA_'+str(period)] = talib.SMA(data['Adj Close'], timeperiod=period) 
        data['SMA_'+str(period)+"-Close/Close"] = (data['SMA_'+str(period)] - data["Adj Close"])/data["Adj Close"] * 100.0
        columns_to_delete.append('SMA_'+str(period))

        data['WMA_'+str(period)] = talib.WMA(data['Adj Close'], timeperiod=period) 
        data['WMA_'+str(period)+"-Close/Close"] = (data['WMA_'+str(period)] - data["Adj Close"])/data["Adj Close"] * 100.0
        columns_to_delete.append('WMA_'+str(period))

        for i in range(1,5):
            data["SMA_"+str(period)+"_diff_"+str(i)] = data['SMA_'+str(period)].pct_change(i) * 100.0
            data["EMA_"+str(period)+"_diff_"+str(i)] = data['EMA_'+str(period)].pct_change(i) * 100.0
            data["WMA_"+str(period)+"_diff_"+str(i)] = data['WMA_'+str(period)].pct_change(i) * 100.0

    

    vol_periods = [5,10,15]
    for period in vol_periods:
        data['VolChg-'+str(period)] = (data['Volume'] / data['Volume'].shift(1).rolling(window=period).mean() ) - 1.0 
       

    data['LINEARREG'] = (talib.LINEARREG(data['Adj Close'])- data['Adj Close']) /  data['Adj Close']
    data['MOM'] = talib.MOM(data['Adj Close'])
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(data['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    return columns_to_delete


def buy_sell(df, index, window):
    global LOSS_OBJETIVE
    global WIN_OBJETIVE
    num_rows = df.shape[0]
    ii = min(num_rows-1, index+1)
    buy_price = df["Open"].iloc[ii]
    num_rows = df.shape[0]
    i = index+1
    num_rows2 = index+ window
    num_rows = min(num_rows,num_rows2)
    while i < num_rows:
       min_price = df["High"].iloc[i]
       close_price =df["Low"].iloc[i]
       loss = (min_price - buy_price) / buy_price
       win = (close_price - buy_price) / buy_price
       #print("loss",loss,"win",win)
       if loss < LOSS_OBJECTIVE * (-1.0):
          return "SELL"
       if win > WIN_OBJECTIVE:
          return "BUY"
       
       i = i +1
    return   "NOTHING"


def add_buy_sell(data):
    data["BUY_SELL"] = ""
    data["BUY"] = 0
    data["SELL"] = 0
    data["NOTHING"] = 0 
    i =  0
    num_rows = data.shape[0]
    data_copy = data.copy()

    while i < num_rows:
        bs = buy_sell(data, i,WINDOW)
        data.loc[data.index[i], 'BUY_SELL'] = bs 
        if bs == "BUY": 
            data.loc[data.index[i], 'BUY'] = 1
        elif bs == "SELL":
            data.loc[data.index[i], 'SELL'] = 1
        else:
            data.loc[data.index[i], 'NOTHING'] = 1
            
        i = i + 1
    return ["BUY_SELL"]

def add_olhc_deltas(data):
    columns_to_delete = []
    data['d_adj_close'] = data["Adj Close"].pct_change() * 100.0
    data["PrevClose"] = data["Adj Close"].shift(1)
    


    columns_to_delete.append("PrevClose")
    
    data["L-O/O"]  =  (data["Low"]-data["Open"])/data["Open"] * 100.0
    data["C-O/O"]  =  (data["Adj Close"]-data["Open"])/data["Open"]* 100.0
    data["H-O/O"]  =  (data["High"]-data["Open"])/data["Open"]* 100.0
    

    data["O-PC/pC"]  =  (data["Open"]-data["PrevClose"])/data["PrevClose"]* 100.0
    data["L-PC/pC"]  =  (data["Low"]-data["PrevClose"])/data["PrevClose"]* 100.0
    data["C-PC/pC"]  =  (data["Adj Close"]-data["PrevClose"])/data["PrevClose"]* 100.0
    data["H-PC/pC"]  =  (data["High"]-data["PrevClose"])/data["PrevClose"]* 100.0


    for i in range(2,10):
        data["C-" +str(i)] = data["Adj Close"].shift(i)
        data["delta_C-" +str(i)] = (data["C-" +str(i)] -data["Adj Close"] )/ data["Adj Close"] 
        columns_to_delete.append("C-" +str(i))

        data["H-" +str(i)] = data["High"].shift(i)
        data["delta_H-" +str(i)] = (data["H-" +str(i)] -data["High"] )/ data["High"] 
        columns_to_delete.append("H-" +str(i))

        data["L-" +str(i)] = data["Low"].shift(i)
        data["delta_L-" +str(i)] = (data["L-" +str(i)] -data["Low"] )/ data["Low"] 
        columns_to_delete.append("L-" +str(i))


    


    return columns_to_delete


def create_X_y(data):
    
    X = data.drop(['BUY', 'SELL', 'NOTHING'], axis=1)
    y = data[['BUY', 'SELL', 'NOTHING']]
    return X,y

def create_model(X,y):
    global STOCK
    num_rows = len(X)
    train_rows = int(float(num_rows) * SPLIT_TRAIN)
    test_rows =  int(float(num_rows) * (SPLIT_TRAIN+SPLIT_TEST))
    X_train   = X.iloc[:train_rows]
    y_train   = y.iloc[:train_rows]

    X_test = X.iloc[train_rows+1:test_rows]
    y_test = y.iloc[train_rows+1:test_rows]


    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)



    model = Sequential()
    shape_size = input_shape=X_train.shape[1]

    model.add(Dense(shape_size , activation='tanh', input_shape=(shape_size,)))
    model.add(Dropout(0.2))

    model.add(Dense(shape_size , activation='sigmoid'))
    model.add(Dropout(0.2))

    model.add(Dense(shape_size , activation='sigmoid'))
    model.add(Dropout(0.2))

    model.add(Dense(shape_size , activation='sigmoid'))
    model.add(Dropout(0.2))

   # model.add(Dense(32, activation='relu'))
   # model.add(Dropout(0.2))

    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(3 , activation='sigmoid'))

    model.add(Dense(3 , activation='softmax'))

    model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
    model.fit(X_train, y_train,epochs=EPOCHS, batch_size=1, verbose=1)
    
    
    last_X = X.tail(1)

    last_X = scaler.transform(last_X)


    last_row_pred_prob = model.predict(last_X)
    print("ARREGLO DE PREDICCION")
    print(last_row_pred_prob)
    last_row_pred = np.argmax(last_row_pred_prob, axis=1)

    # Convert the prediction to the original label
    last_row_pred_label = pd.Series(last_row_pred).map({0: 'BUY', 1: 'SELL', 2: 'NOTHING'})
    print("-------------------------------------")
    buy = round(last_row_pred_prob[0,0]*100.0,2)
    sell = round(last_row_pred_prob[0,1]*100.0,2)
    nothing = round(last_row_pred_prob[0,2]*100.0,2)
    print("-------------------------------------")
    print("Buy       : ",buy,"%")
    print("Sell      : ",sell,"%")
    print("Do Nothing: ",nothing,"%")
    print("-------------------------------------")
    print("-------------------------------------")
    print("-------------------------------------")
    print("-------------------------------------")
    print("-------------------------------------")
    print(f'PREDICCION PARA {STOCK} : {last_row_pred_label.iloc[0]}')
    print("-------------------------------------")
    print("-------------------------------------")
    print("-------------------------------------")
    print("-------------------------------------")
    print("-------------------------------------")

    last_X = X.tail(20)
    last_X = scaler.transform(last_X)

    last_row_pred_prob = model.predict(last_X)
    print("ARREGLO DE PREDICCION")
    print(last_row_pred_prob)
    last_row_pred = np.argmax(last_row_pred_prob, axis=1)

    # Convert the prediction to the original label
    last_row_pred_label = pd.Series(last_row_pred).map({0: 'BUY', 1: 'SELL', 2: 'NOTHING'})
    print(last_row_pred)
    


#-------------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------------
def main():
    global LOSS_OBJECTIVE
    global WIN_OBJECTIVE
    global WINDOW
    global STOCK
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Stock data fetcher and plotter")
    parser.add_argument("--stock", required=True, type=str, help="Stock symbol (e.g. AAPL, TSLA)")
    parser.add_argument("--start_date", type=valid_date, help="Start date (format: YYYY-MM-DD)")
    parser.add_argument("--end_date", type=valid_date, help="End date (format: YYYY-MM-DD)")
    parser.add_argument("--desired_win", type=float, help="")
    parser.add_argument("--window", type=int, help="")


    # Parse the arguments
    args = parser.parse_args()
    STOCK = args.stock
    if args.desired_win:
        WIN_OBJECTIVE = float(args.desired_win) /100.0
        LOSS_OBJECTIVE = float(args.desired_win) /100.0
    if args.window:
        WINDOW = int(args.window)
    # If end_date is not provided, use today's date
    end_date = None
    if args.end_date:
        end_date = args.end_date 

    # If start_date is not provided, use one year ago from end_date
    start_date = args.start_date if args.start_date else end_date - timedelta(days=365)

    # Fetch and plot stock data
    try:
        end_date_str = None
        if end_date:
            end_date_str = end_date.strftime('%Y-%m-%d')
        data = fetch_stock_data(args.stock, start_date.strftime('%Y-%m-%d'),end_date_str)
        columns_to_delete = [ "Open","High"	,"Low",	"Close",	"Adj Close",	"Volume"]

        more_columns_to_delete = add_indicators_to_data(data)
        columns_to_delete.extend(more_columns_to_delete)        

        more_columns_to_delete = add_olhc_deltas(data)
        columns_to_delete.extend(more_columns_to_delete)        

        more_columns_to_delete = add_buy_sell(data)
        columns_to_delete.extend(more_columns_to_delete)        

        print("_-----------------------------------------------")
        print(data)
        #data.reset_index()
        data.info()
        data.to_excel(f"data/{args.stock}-data.xlsx")
        plot_stock_data(args.stock, data)
        print("_-----------------------------------------------")
        data = data.reset_index()
        columns_to_delete.append("Date")
        data = data.drop(columns_to_delete,axis=1)
        data = data.dropna()
        data.to_excel(f"data/{args.stock}-data-train.xlsx")    
        
        X,y = create_X_y(data)
        print(X)
        print(y)

        model = create_model(X,y)

    except ValueError as e:
        print(e)



# Entry point
if __name__ == "__main__":
    main()
