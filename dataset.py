import pandas as pd
import pickle as pkl
import os

from sklearn.linear_model import LinearRegression

class DataProcessor():
    def __init__(self, folder, file, asset):
        self.folder = folder
        file = folder + file
        self.pkl = f'{file}.pkl'
        self.raw = f'{file}.csv'
        self.columnHandled = f'{file}_columnHandled.csv'
        self.concated = f'{file}_concated.csv'
        self.rowFiltered = f'{file}_rowFiltered.csv'

        self.asset = asset

    def __call__(self):    
        df = self.pkl_to_csv()
        df = self.columnHandling(df)
        #df = self.concat(df)

    def pkl_to_csv(self):
        print('Converting pickle file to csv...')
        if os.path.exists(self.raw):
            print('\tconverted csv file already exist.')
            df = pd.read_csv(self.raw, dtype='object')
            return df
    
        print('\tNo csv file, converting pickle...')
        with open(self.pkl, "rb") as f:
            object = pkl.load(f)
        
            df = pd.DataFrame(object)
            print(df.head())
            df.to_csv(self.raw)
        return df
    
    def columnHandling(self, raw_dataset):
        print('Handling columns...')
        if os.path.exists(self.columnHandled):
            print('\tcolumn-filtered dataset already exist')
            filtered_dataset = pd.read_csv(self.columnHandled, dtype='object')
            return filtered_dataset

        print('\tNo column-filtered dataset exist')
        print('\tfiltering columns...')
        raw_dataset = raw_dataset[['交易日期','履約價','買賣權','成交量','結算價','到期日期','time to maturity']]
        raw_dataset.columns = ['date', 'strike_price', 'put/call', 'volume', 'option_price', 'exdate', 'tau']
        
        raw_dataset['put/call'] = raw_dataset['put/call'].replace({'買權': 'call', '賣權': 'put'})
        raw_dataset['tau'] = raw_dataset['tau'].astype(float)/252

        print('\tconcating underlying asset...')
        asset = pd.read_csv(self.asset, index_col=None)
        print(asset.head())
        raw_dataset = pd.merge(raw_dataset, asset[['date', 'Adj Close']], on='date', how='left')
        raw_dataset = raw_dataset.rename(columns={'Adj Close': 'underlying'})

        print(raw_dataset.head())
        raw_dataset.to_csv(self.columnHandled)
        return raw_dataset
    
    def concat(self, filtered):
        print('Concating dividend and interest rates...')
        if os.path.exists(self.concated):
            print('\tconcated dataset already exist')
            concated_dataset = pd.read_csv(self.concated, dtype='object')
            return concated_dataset

        print('\tChoosing data between put and call...')
        call = filtered.iloc[::2].reset_index(drop=True)
        put = filtered.iloc[1::2].reset_index(drop=True)
        
        near_atm_call = call[call['']]
        '''
        put_call_filter = call['volume'].astype(int) > put['volume'].astype(int)
        filtered_dataset = call.where(put_call_filter, put)
        print(filtered_dataset.head())
        filtered_dataset.to_csv(self.filtered)'''

def main():
    dataprocessor = DataProcessor('dataset/','2009_2023','dataset/0050TW.csv')
    dataprocessor()

if __name__ == '__main__':
    main()