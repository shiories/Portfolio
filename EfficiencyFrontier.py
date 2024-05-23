import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import asyncio


class EfficiencyFrontier:
    '''
    included 為投資組合名稱,
    data 需要轉成年化報酬率資料
    to_be_annualized  是否需要年化處理
    interval 提示資料週期 "1d", "1wk", "1mo"
    '''
    def __init__(self, included=str("投資組合名稱"), data=pd.DataFrame, free_risk=float(0.01), interval="1d"):
        interval_mapping = { "1d": 250, "1wk": 50, "1mo": 12 }
        interval = interval_mapping.get(interval, interval)
        self.included = included
        self.cov_df = data.cov() * interval
        self.cov = np.array(self.cov_df)
        self.data_columns = data.columns.tolist()
        self.data_describe=data.describe().T
        self.data_describe["mean"] = self.data_describe["mean"] * interval
        self.data_describe["std"] = self.data_describe["std"] * (interval ** 0.5)
        self.u = self.data_describe["mean"]
        
        if not os.path.exists(f'{self.included}'):
            os.makedirs(f'{self.included}')
        self.EF_df = None
        self.limiteEF_df = None
        self.CAL_df = None
        self.data =data
        
    
    async def run_limit(self, num_iterations):
        tasks = []
        self.result_df = pd.DataFrame(columns=['u_p', 'std_p'] + self.data_columns)
        for _ in range(num_iterations):
            tasks.append(self.random_combination())
        results = await asyncio.gather(*tasks)
        self.result_df = pd.concat(results, ignore_index=True)

        self.result_df = self.result_df.sort_values(by='u_p')
        self.result_df = self.result_df.reset_index(drop=True)
            
        # 構建效率前緣表
        self.limiteEF_df = await self.run_limit_parallel()
        
        return self.result_df, self.limiteEF_df


    async def random_combination(self):
        w = np.random.random(len(self.u))
        w /= np.sum(w)  # 使权重总和为1
        std_p = np.sqrt(np.matmul(np.matmul(w.T, self.cov), w))
        u_p = np.matmul(w, self.u)
        row_data = pd.DataFrame([u_p, std_p] + w.tolist(), ['u_p', 'std_p'] + self.data_columns).T
        
        return row_data


    async def run_limit_parallel(self):
        self.limiteEF_df = pd.DataFrame(columns=['u_p', 'std_p'] + self.data_columns)
        u_p_values = np.linspace(self.result_df['u_p'].min(), self.result_df['u_p'].max(), 30)
        unit = (self.result_df['u_p'].max()*1.1 - self.result_df['u_p'].min()*1.1) / 15
        tasks = []
        for u_p in u_p_values:
            tasks.append(self.calculate_min_std_parallel(u_p, unit))
        results = await asyncio.gather(*tasks)
        self.limiteEF_df = pd.concat(results, ignore_index=True)
        return self.limiteEF_df


    async def calculate_min_std_parallel(self, u_p, unit):
        min_std_row = self.result_df[self.result_df['u_p'].sub(u_p).abs().lt(unit)].nsmallest(1, 'std_p')
        if not min_std_row.empty:
            min_std_row = min_std_row.iloc[0]
            return min_std_row.to_frame().T
        else:
            print(f"No valid row found for u_p = {u_p}")
            return None


    def limit_solution(self, num_iterations=3000):
        print("限制式EF規劃求解中. . .")
        self.result_df, self.limiteEF_df = asyncio.run(self.run_limit(num_iterations))
        self.limiteEF_plot()
        self.save_data()
        return self.limiteEF_df


    def limiteEF_plot(self):
        # 繪製成表格
        plt.rcParams['font.family'] = ['Microsoft YaHei', 'sans-serif']
        plt.figure(figsize=(8, 4.5))
        plt.scatter(self.result_df['std_p'], self.result_df['u_p'], marker='o', s=5, label='配置結果', color='dodgerblue')
        plt.plot(self.limiteEF_df['std_p'], self.limiteEF_df['u_p'], label='限制式EF', color='sandybrown', linewidth=2,linestyle='-')
        plt.xlabel('標準差', size=10)
        plt.ylabel('收益率', size=10)
        plt.title('限制式效率前緣', size=12)
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.savefig(f'{self.included}/限制式效率前緣.png', dpi=300)
        plt.close()
        
        if not self.EF_df.empty:
            plt.rcParams['font.family'] = ['Microsoft YaHei', 'sans-serif']
            plt.figure(figsize=(8, 4.5))
            plt.plot(self.limiteEF_df['std_p'], self.limiteEF_df['u_p'], label='限制式EF', color='sandybrown', linewidth=2,linestyle='-')
            plt.plot(self.EF_df['std_p'], self.EF_df['u_p'], label='EF', color='dodgerblue')
            plt.xlabel('標準差', size=10)
            plt.ylabel('收益率', size=10)
            plt.title('限制式效率前緣比較', size=12)
            plt.legend(fontsize=8)
            plt.grid(True)
            plt.savefig(f'{self.included}/限制式效率前緣比較.png', dpi=300)
            plt.close()


    async def run_unlimit(self, A, l):
        # 創建一個空的 DataFrame 來存儲結果
        self.EF_df = pd.DataFrame(columns=['u_p', 'var_p', 'std_p'] + self.data_columns)
        tasks = []
        for u_p in np.arange(-0.3, 0.7, 0.01).tolist():
            tasks.append(self.run_u_p(u_p, A, l))
        
        results = await asyncio.gather(*tasks)
        
        self.EF_df = pd.concat(results, ignore_index=True)

        return self.EF_df


    def unlimit_solution(self):
        l = np.array([1] * len(self.cov))
        A = np.matmul(np.matmul(np.vstack([self.u, l]), np.linalg.inv(self.cov)), np.vstack([self.u, l]).T)
        #u_p = 0.01
        #var_p = np.matmul(np.matmul(np.array([u_p, 1]),np.linalg.inv(A)),np.array([u_p, 1]).T)
        #std_p = np.sqrt(var_p)
        self.EF_df = asyncio.run(self.run_unlimit(A, l))
        self.EF_plot(self.EF_df)
        
        return self.EF_df 
        

    async def run_u_p(self, u_p, A, l):
        u_l = np.array([u_p, 1])
        var_p = np.matmul(np.matmul(u_l, np.linalg.inv(A)), u_l.T)
        std_p = np.sqrt(var_p)
        w = np.matmul(np.matmul(u_l, np.linalg.inv(A)), np.matmul(np.vstack([self.u, l]), np.linalg.inv(self.cov)))
        row_data = pd.DataFrame([u_p, std_p] + w.tolist(), ['u_p', 'std_p'] + self.data_columns).T
        return row_data
    

    def EF_plot(self, EF_df):
        plt.rcParams['font.family'] = ['Microsoft YaHei', 'sans-serif']  
        # 繪製 u_p 對 std_p 的曲線
        plt.figure(figsize=(8, 4.5))
        plt.plot(EF_df['std_p'], EF_df['u_p'], label='EF', color='sandybrown')
        plt.scatter(EF_df['std_p'], EF_df['u_p'], label='EF點', color='dodgerblue', marker='o')
        plt.xlabel('標準差', size=10)
        plt.ylabel('收益率', size=10)
        plt.title('效率前緣', size=12)
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.savefig(f'{self.included}/效率前緣.png', dpi=300)
        plt.close()

        plt.figure(figsize=(8, 4.5))
        plt.plot(EF_df['std_p'], EF_df['u_p'], label='EF', color='sandybrown')
        plt.scatter(self.data_describe['std'], self.data_describe['mean'], label='股票',color="dodgerblue")
        plt.xlabel('標準差', size=10)
        plt.ylabel('收益率', size=10)
        plt.title('效率前緣', size=12)
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.savefig(f'{self.included}/效率前緣對上股票.png', dpi=300)
        plt.close()


    def capital_allocation(self):
        l = np.array([1] * len(self.cov))
        u_f = 0.01 #假定無風險利率
        self.CAL_df = pd.DataFrame(columns=['u_', 'u_p', 'var_p', 'std_p'] + self.data_columns) 
        u_bar = np.array([i - u_f  for i in self.u])
        
        a = np.matmul(np.matmul(u_bar.T, np.linalg.inv(self.cov)), u_bar)
        b = np.matmul(u_bar.T, np.linalg.inv(self.cov))
        
        orp_w =  b / np.matmul(b, l)
        orp_std = np.sqrt(a / np.square(np.matmul(b, l)))
        orp_u = a / np.matmul(b, l) + u_f
        
        print(f'orp_w: {orp_w} , orp_var: {orp_std} ,orp_u: {orp_u}')
        for i, u_ in enumerate(np.arange(0, 0.7, 0.01).tolist()):
            u_p = u_ + u_f
            var_p = np.square(u_) / np.matmul(b, u_bar)
            std_p = np.sqrt(var_p)            
            w = u_p / a * b
            
            self.CAL_df.loc[i] = [u_, u_p, var_p, std_p] + w.tolist()

        # 繪製 u_p 對 std_p 的曲線
        plt.figure(figsize=(8, 4.5))
        if not self.EF_df.empty:
            plt.plot(self.EF_df['std_p'], self.EF_df['u_p'], label='EF', color='sandybrown')
        plt.plot(self.CAL_df['std_p'], self.CAL_df['u_p'], label='CAL', color='dodgerblue')
        plt.scatter(orp_std, orp_u, color='red', marker='o', label='orp')
        plt.xlabel('標準差', size=10)
        plt.ylabel('收益率', size=10)
        plt.title('效率前緣與資產配置線', size=12)
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.savefig(f'{self.included}/效率前緣與資產配置線.png', dpi=300)
        plt.close()
        self.save_data()
        return self.CAL_df


    def save_data(self):
        file_path = f'{self.included}/效率前緣.xlsx'
        mode = 'a' if os.path.exists(file_path) else 'w'
        with pd.ExcelWriter(file_path, mode=mode, engine='openpyxl') as writer:
            if self.data_describe is not None:
                if '年化基礎統計' in writer.book.sheetnames:
                    writer.book.remove(writer.book['年化基礎統計'])
                self.data_describe.to_excel(writer, sheet_name='年化基礎統計')
            if self.cov_df is not None:
                if '變異數-共變異數矩陣' in writer.book.sheetnames:
                    writer.book.remove(writer.book['變異數-共變異數矩陣'])
                self.cov_df.to_excel(writer, sheet_name='變異數-共變異數矩陣')
            if self.limiteEF_df is not None:
                if '限制式效率前緣' in writer.book.sheetnames:
                    writer.book.remove(writer.book['限制式效率前緣'])
                self.limiteEF_df.to_excel(writer, sheet_name='限制式效率前緣', index=False)
            if self.result_df is not None:
                if '限制式隨機權重' in writer.book.sheetnames:
                    writer.book.remove(writer.book['限制式隨機權重'])
                self.result_df.to_excel(writer, sheet_name='限制式隨機權重', index=False)
            if self.CAL_df is not None:
                if '資產配置線' in writer.book.sheetnames:
                    writer.book.remove(writer.book['資產配置線'])
                self.CAL_df.to_excel(writer, sheet_name='資產配置線', index=False)



if __name__ == "__main__" :
    import yfinance as yf
    
    included = "課本範例"   
    start_date = "2015-01-01"
    end_date = "2019-12-31"
    interval = "1mo"
    stock = ["2454.TW", "2330.TW", "3008.TW", "2395.TW", "2882.TW", "1101.TW", "2002.TW", "1301.TW"]
    data = yf.download(tickers=stock,start=start_date, end=end_date, progress=False, interval=interval)
    data = data["Adj Close"]
    data = data.apply(pd.to_numeric, errors='coerce')
    # 將數據轉換為對數收益率並去除NaN值
    data = np.log(data / data.shift(1))
    data = data.dropna()

    free_risk = 0.01
    num_iterations = 10000
    efficiency_frontier = EfficiencyFrontier(included, data, free_risk, interval )
    EF_df = efficiency_frontier.unlimit_solution()
    limiteEF_df = efficiency_frontier.limit_solution(num_iterations)
    CAL_df = efficiency_frontier.capital_allocation()
    
    
    
    
