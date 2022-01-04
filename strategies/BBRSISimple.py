# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce

# This class is a sample. Feel free to customize it.
class BBRSISimple(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.05,
        "30": 0.045,
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.03

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03  # Disabled / not configured

    # Hyperoptable parameters
    rsi_value = IntParameter(low=5, high=50, default=30, space='buy', optimize=True, load=True)
    #rsi_value = float(rsi_v)
    #rsi_enabled = BooleanParameter(default=True, space='buy', optimize=True)
    #trigger = CategoricalParameter(['bb_lower', 'macd_cross_signal', 'sar_reversal'], default='bb_lower', optimize=True, load=True)
    sell_rsi_value = IntParameter(low=51, high=100, default=69, space='sell', optimize=True, load=True)
    sell_rsi_enabled = BooleanParameter(default=True, space='sell', optimize=True)
    sell_trigger = CategoricalParameter(['sell-bb_lower1', 'sell-bb_middle1', 'sell-bb_upper1'], space='sell', default=False, optimize=True, load=True)
    rsi_enabled = BooleanParameter(default=True, space='buy', optimize=True)
    trigger = CategoricalParameter(['bb_lower1', 'bb_lower2', 'bb_lower3', 'bb_lower4', 'macd_cross_signal', 'sar_reversal'], default='bb_lower1', space='buy', optimize=True, load=True)

    #ProtectionGuard
    @property
    def protections(self):
            return [
                {
                    "method": "MaxDrawdown",
                    "lookback_period_candles": 14,
                    "trade_limit": 4,
                    "stop_duration_candles": 2,
                    "max_allowed_drawdown": 0.7
                },
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": 20,
                    "trade_limit": 3,
                    "stop_duration_candles": 4,
                    "only_per_pair": True
                },
                {
                    "method": "LowProfitPairs",
                    "lookback_period_candles": 100,
                    "trade_limit": 4,
                    "stop_duration_candles": 4,
                    "required_profit": 0.01
                }
                
            ]
            
    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
       """
       
        return [("ETH/USDT", "1h"),
                ("ETH/USDT", "15m"),
                ("BTC/USDT", "15m"),
                ("RVN/USDT", "15m")
               ]
       
        #return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
  
        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # Momentum Indicators
        # ------------------------------------

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

       
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
        # dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

       
        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

       
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

       
        # Bollinger Bands
        bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['bb_lowerband1'] = bollinger1['lower']
        dataframe['bb_middleband1'] = bollinger1['mid']
        dataframe['bb_upperband1'] = bollinger1['upper']
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']


        # # EMA - Exponential Moving Average
        dataframe['ema7'] = ta.EMA(dataframe, timeperiod=7)
        dataframe['ema30'] = ta.EMA(dataframe, timeperiod=30)
        dataframe['ema60'] = ta.EMA(dataframe, timeperiod=60)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        
        # Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe)

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']

        
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        
        Conditions=[]
        if self.rsi_enabled:
           Conditions.append(self.rsi_value.value > dataframe['rsi'])
        else:     
              
               if self.trigger == 'bb_lower1':
                    Conditions.append(dataframe['close'] < dataframe['bb_lowerband1'])
               if self.trigger == 'bb_lower2':
                    Conditions.append(dataframe['close'] < dataframe['bb_lowerband2'])
               if self.trigger == 'bb_lower3':
                    Conditions.append(dataframe['close'] < dataframe['bb_lowerband3'])
               if self.trigger == 'bb_lower4':
                    Conditions.append(dataframe['close'] < dataframe['bb_lowerband4'])
               if self.trigger == 'macd_cross_signal':
                    Conditions.append(qtpylib.crossed_above(
                        dataframe['macd'], dataframe['macdsignal']))
               if self.trigger == 'sar_reversal':
                    Conditions.append(qtpylib.crossed_above(
                        dataframe['close'], dataframe['sar']))
        dataframe.loc[
               reduce(lambda x, y: x & y, Conditions),
            'buy']=1
            
        return dataframe
            # GUARDS AND TRENDS
           
            # TRIGGERS
        
                
              

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            # print(params)
            # sell_rsi_enabled = BooleanParameter(default=True, space='sell', optimize=True)
            # sell_trigger = CategoricalParameter(['sell-bb_lower1', 'sell-bb_middle1', 'sell-bb_upper1'], space='sell', default=False, optimize=True, load=True)
            conditions =[]
            if self.sell_rsi_enabled:
                conditions.append(dataframe['rsi'] > self.sell_rsi_value.value)
            else:
            # TRIGGERS
              if self.sell_trigger:
                if self.sell_trigger == 'sell-bb_lower1':
                    conditions.append(dataframe['close'] > dataframe['bb_lowerband1'])
                if self.sell_trigger == 'sell-bb_middle1':
                    conditions.append(dataframe['close'] > dataframe['bb_middleband1'])
                if self.sell_trigger == 'sell-bb_upper1':
                    conditions.append(dataframe['close'] > dataframe['bb_upperband1'])
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

            return dataframe
