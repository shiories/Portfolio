import os
import requests
import asyncio
import aiohttp
import pandas as pd
import yfinance as yf
import numpy as np
from io import StringIO
from datetime import datetime

class IndustryData:
    '''
    included 為投資組合名稱,
    start_date %Y%m%d日期字串,
    start_date %Y%m%d日期字串,
    
    get_free_rate(free_risk) = None,"1m","3m","6m","9m","1y","2y","3y"
    '''
    def __init__(self, included=str("投資組合名稱"), start_date=str("2019-04-30"), end_date=str("2024-04-30")):
        self.start_date = start_date
        self.end_date = end_date
        self.included = included
        if not os.path.exists(f'{self.included}'):
            os.makedirs(f'{self.included}')
        self.stock_df = None
        self.data = None
        self.rate_df = None


    async def get_stock(self, cutoff_year=None, min_count=None, excluded=None, included=None):
        included = [included]

        url = "https://isin.twse.com.tw/isin/class_main.jsp?market=1&issuetype=1"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html_text = await response.text()
                html_file = StringIO(html_text)
                stock_df = pd.read_html(html_file)[0]
        
        stock_df.columns = stock_df.iloc[0]
        stock_df = stock_df.iloc[1:]
        stock_df = stock_df.dropna(thresh=3, axis=0).dropna(thresh=3, axis=1)
        selected_columns = ['有價證券代號', '有價證券名稱', '市場別', '有價證券別', '產業別', '公開發行/上市(櫃)/發行日']
        stock_df = stock_df.loc[:, selected_columns]
        stock_df = stock_df.drop(columns=['有價證券別'])
        stock_df = stock_df.rename(columns={'有價證券代號': '證券代號','有價證券名稱': '公司名稱', '公開發行/上市(櫃)/發行日': '公開發行日'})
        stock_df = stock_df.reset_index(drop=True)

        if cutoff_year:
            stock_df['公開發行日'] = pd.to_datetime(stock_df['公開發行日'])
            stock_df = stock_df[stock_df['公開發行日'].dt.year <= cutoff_year]

        if excluded:
            stock_df = stock_df[~stock_df['產業別'].isin(excluded)]

        if included:
            stock_df = stock_df[stock_df['產業別'].isin(included)]

        if min_count:
            industry_counts = stock_df['產業別'].value_counts()
            industries_to_keep = industry_counts[industry_counts >= min_count].index.tolist()
            stock_df = stock_df[stock_df['產業別'].isin(industries_to_keep)]

        stock_list = stock_df['證券代號'].tolist()
        stock_list = [code + '.TW' for code in stock_list]

        industry_df = stock_df.groupby('產業別').size().reset_index(name='公司數')
        industry_df = industry_df.sort_values(by='公司數', ascending=False).reset_index(drop=True)

        return stock_df, stock_list, industry_df


    async def get_index(self, start_date, end_date):

        index_mapping = {'綠能環保': 'IX0185', '數位雲端': 'IX0186', '運動休閒': 'IX0187', '居家生活': 'IX0188', 
                        '水泥工業': 't01', '食品工業': 't02', '塑膠工業': 't03', '紡織纖維': 't04', '電機機械': 't05', '電器電纜': 't06',
                        '玻璃陶瓷': 't08', '造紙工業': 't09', '鋼鐵工業': 't10', '橡膠工業': 't11', '汽車工業': 't12', '建材營造業': 't14',
                        ' 航運業': 't15', '觀光餐旅': 't16', '金融保險業': 't17', '貿易百貨業': 't18', '其他業': 't20', '化學工業': 't21',
                        '生技醫療業': 't22', '油電燃氣業': 't23', '半導體業': 't24', '電腦及週邊設備業': 't25', '光電業': 't26',
                        '通信網路業': 't27', '其他電子業': 't28', '電子通路業': 't29', '資訊服務業': 't30', '電子零組件業': 't31'}
        
        index = index_mapping.get(self.included)
        url = f"https://backend.taiwanindex.com.tw/api/indexes/{index}/records"
        params = {
            "start": start_date,
            "end": end_date
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            # Assuming the response contains JSON data
            datasets = response.json()

            # 提取'data'鍵的值
            data_value = datasets['data']
            # 提取資料
            extracted_data = {'Date': data_value['labels'], self.included: data_value['datasets'][0]['data']}
            df = pd.DataFrame(extracted_data)
            df.set_index('Date', inplace=True)
            df.set_index(pd.to_datetime(df.index), inplace=True)

        else:
            print("Failed to fetch data. Status code:", response.status_code)

        return df


    async def get_close(self, start_date, end_date):

        data = yf.download(self.stock_list, start=start_date, end=end_date, progress=False )
        data = data['Adj Close']

        return data


    async def convert_index_to_ad(self, index: str) -> str:
        # 將民國年轉換為西元年
        year = int(index[:3]) + 1911
        month_day = index[3:]
        return f"{year}-{month_day}"
    

    async def get_free_rate(self, free_risk=str("1y")):

        # 讀取 Excel 檔案
        file_url = "https://www.cbc.gov.tw/tw/public/data/a13rate.xls"
        xl = pd.read_excel(file_url, sheet_name=None, engine='openpyxl')

        # 讀取每個分頁中的表格並存為字典
        sheet_to_df = {}
        for sheet_name, df in xl.items():
            # 將第一列作為索引
            df.set_index(df.columns[0], inplace=True)
            df.drop(df.index[:4], inplace=True)
            # 只保留指定的列
            new_df = df.iloc[:, [4, 6, 8, 10, 12, 14, 16]]  # 選擇第 6、8、10、12、14、16、18 列
            new_df.columns = ["1m","3m","6m","9m","1y","2y","3y"]
            sheet_to_df[sheet_name] = new_df

        # 合併所有 DataFrame，並按索引分組求平均
        merged_df = pd.concat(sheet_to_df.values()).groupby(level=0).mean()

        # 轉換索引為西元年
        merged_df.index = await asyncio.gather(*map(self.convert_index_to_ad, merged_df.index))

        rate_df = merged_df[free_risk]
        rate_df = rate_df / 365
    
        print(f"無風險利率選定為 {free_risk} : \n{rate_df}")
        return rate_df


    def get_data(self, free_risk=None , to_excel=True):
        cutoff_year = datetime.strptime(self.start_date, "%Y-%m-%d").year
        self.stock_df, self.stock_list, self.industry_df = asyncio.run(self.get_stock(cutoff_year=cutoff_year , included=self.included))
        print(f'{self.stock_df}\n\n{self.industry_df}')
        
        # 獲取股票數據和指數數據
        data = asyncio.run(self.get_close(self.start_date, self.end_date))
        index_data = asyncio.run(self.get_index(self.start_date, self.end_date))
        data = index_data.join(data, how='inner')
        data = data.apply(pd.to_numeric, errors='coerce')

        # 將數據轉換為對數收益率並去除NaN值
        data = np.log(data / data.shift(1))
        self.data = data.dropna()

        if free_risk is not None:
            self.rate_df = asyncio.run(self.get_free_rate(free_risk))
            if to_excel:
                self.save_data()
            return self.data, self.rate_df
        else:
            if to_excel:
                self.save_data()
            return self.data


    def save_data(self):
        file_path = f'{self.included}/基本資料.xlsx'
        mode = 'a' if os.path.exists(file_path) else 'w'

        with pd.ExcelWriter(file_path, mode=mode, engine='openpyxl') as writer:
            if self.stock_df is not None:
                if '基本資料' in writer.book.sheetnames:
                    writer.book.remove(writer.book['基本資料'])
                self.stock_df.to_excel(writer, sheet_name='基本資料', index=False)
            if self.data is not None:
                if 'ln報酬率' in writer.book.sheetnames:
                    writer.book.remove(writer.book['ln報酬率'])
                self.data.to_excel(writer, sheet_name='ln報酬率')
            if self.rate_df is not None:
                if '平均歷史利率表' in writer.book.sheetnames:
                    writer.book.remove(writer.book['平均歷史利率表'])
                self.rate_df.to_excel(writer, sheet_name='平均歷史利率表')




if __name__ == "__main__" :

    start_date = "2019-04-30"
    end_date = "2024-04-30"
    free_risk = "1y" #1m, 3m, 6m, 9m, 1y, 2y, 3y
    industry_data = IndustryData("金融保險業", start_date, end_date)
    data = industry_data.get_data()
    print(f'data:\n{data}')
    data, free_rate_data = industry_data.get_data(free_risk)
    print(f'free_rate_data:\n{free_rate_data}')
    
    #綠能環保 數位雲端 運動休閒 居家生活 水泥工業 食品工業 塑膠工業 紡織纖維 電機機械 電器電纜 玻璃陶瓷
    #造紙工業 鋼鐵工業 橡膠工業 汽車工業 建材營造業 航運業 觀光餐旅 金融保險業 貿易百貨業 其他業 化學工業
    #生技醫療業 油電燃氣業 半導體業 電腦及週邊設備業 光電業 通信網路業 其他電子業 電子通路業 資訊服務業 電子零組件業

