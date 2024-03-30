import pandas as pd
import pickle as pkl
import os

class DataProcessor():
    def __init__(self, file):
        self.pkl = f'{file}.pkl'
        self.raw = f'{file}.csv'
        self.filtered = f'{file}_filtered.csv'

    def __call__(self):    
        df = self.pkl_to_csv()
        df = self.filter(df)
        df = self.concat(df)

    def pkl_to_csv(self):
        print('Converting pickle file to csv...')
        if os.path.exists(self.raw):
            print('\tconverted csv file already exist.')
            df = pd.read_csv(self.raw, dtype='object')
            #print(df.head())
            return df
    
        print('\tNo csv file, converting pickle...')
        with open(self.pkl, "rb") as f:
            object = pkl.load(f)
        
            df = pd.DataFrame(object)
            print(df.head())
            df.to_csv(self.raw)
        return df

    def filter(self, raw_dataset):
        print('Filtering...')
        if os.path.exists(self.filtered):
            print('\tfiltered dataset already exist')
            filtered_dataset = pd.read_csv(self.filtered, dtype='object')
            return filtered_dataset

        print('\tFiltering columns...')
        raw_dataset = raw_dataset[['交易日期','履約價','買賣權','成交量','結算價','到期日期','time to maturity']]
        rename_col = ['date', 'strike_price', 'put/call', 'volume', 'option_price', 'exdate', 'tau']
        raw_dataset.columns = rename_col
        #print(raw_dataset.head())

        print('\tChoosing data between put and call...')
        call = raw_dataset.iloc[::2].reset_index(drop=True)
        put = raw_dataset.iloc[1::2].reset_index(drop=True)
        #print(call.head())
        #print(put.head())
        put_call_filter = call['volume'].astype(int) > put['volume'].astype(int)
        filtered_dataset = call.where(put_call_filter, put)
        print(filtered_dataset.head())
        filtered_dataset.to_csv(self.filtered)

    def concat(self, filtered):
        print('Concating datas...')

def main():
    dataprocessor = DataProcessor('dataset/2009_2023')
    dataprocessor()

if __name__ == '__main__':
    main()