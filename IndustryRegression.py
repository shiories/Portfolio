import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import asyncio
import statsmodels.api as sm


class IndustryRegression:
    '''
    included 為投資組合名稱,
    data 需要轉成年化報酬率資料
    rate_df 為歷史利率表
    '''
    def __init__(self, included=str("投資組合名稱"), data=pd.DataFrame, rate_df=pd.DataFrame):
        self.included = included
        self.data = data
        self.u = np.array(data.mean().tolist())
        self.data_cov = self.data.cov()
        self.data_corr = self.data.corr()
        self.cov = np.array(self.data_cov)
        self.data_describe=self.data.describe().T
        self.rate_df = rate_df
        if not os.path.exists(f'{self.included}'):
            os.makedirs(f'{self.included}')
        self.free_rate_data = self.data.copy()


    def get_statistics(sself, data_df):
        X_avg = np.array(data_df["Avg"])
        X_avg = sm.add_constant(X_avg)
        model_avg = sm.OLS(data_df["Beta"], X_avg).fit()
        avg_beta = model_avg.params.iloc[1]
        avg_alpha = model_avg.params.iloc[0]
        avg_r_squared = model_avg.rsquared
        avg_beta_p_value = model_avg.pvalues.iloc[1]
        avg_alpha_p_value = model_avg.pvalues.iloc[0]
        stats_df = pd.DataFrame({
            'Parameter': ['Beta', 'Alpha', 'R_squared', 'β: P_value', 'α: P_value'],
            'Value': [avg_beta, avg_alpha, avg_r_squared, avg_beta_p_value, avg_alpha_p_value]
        })

        return stats_df


    async def regression_plot(self, X, y, model, stock_name):
        plt.rcParams['font.family'] = ['Microsoft YaHei', 'sans-serif']
            
        max_size = 20  
        min_size = 5   
        sizes = max(min(15- len(y) /200, max_size), min_size)        
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(X, y, label=f'與{self.included}相關性', color='dodgerblue', s=sizes)
        ax.plot(X, model.predict(sm.add_constant(X)), color='sandybrown', label='證券特徵線SCL') 
        ax.set_title(f'{stock_name}對{self.included}的散點圖與證券特徵線', fontsize=12)  
        ax.set_xlabel(f'{self.included}的ln收益率', fontsize=10)  
        ax.set_ylabel(f'{stock_name}的ln收益率', fontsize=10)  
        ax.legend(fontsize=8)
        
        # 調整坐標軸刻度的大小
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        
        # 設置圖像的 DPI (每英寸點數) 為 300，並保存圖像
        plt.savefig(f'{self.included}/{stock_name.replace(".TW", "")}.png', dpi=300)
        plt.close()


    async def calculate_regression(self, X, y, column_name):
        avg_return = np.mean(y)

        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()

        beta, alpha = model.params[1], model.params[0]
        r_squared = model.rsquared
        p_value = model.pvalues[1]

        # 保存散點圖
        await self.regression_plot(X, y, model, column_name)

        return {
            'Stock': column_name,
            'Avg': avg_return,
            'Beta': beta,
            'Alpha': alpha,
            'R_squared': r_squared,
            'P_value': round(p_value, 6)
        }


    async def get_regression(self):

        X = self.free_rate_data.iloc[:, 0].values
        y = self.free_rate_data.iloc[:, 1:].values
        
        tasks = [self.calculate_regression(X, y[:, i], self.free_rate_data.columns[i+1]) for i in range(y.shape[1])]
        regression_results = await asyncio.gather(*tasks)
        
        regression_result = pd.DataFrame(regression_results)
        regression_result.set_index('Stock', inplace=True)

        return regression_result


    def industry_plot(self, std, mean):
        # 生成散點圖
        plt.rcParams['font.family'] = ['Microsoft YaHei', 'sans-serif']  
        plt.subplots(figsize=(8, 4.5))
        plt.scatter(self.data_describe['std'], self.data_describe['mean'], label='股票',color="dodgerblue")
        plt.scatter(std, mean, color='sandybrown', label='等權重')
        plt.xlabel('標準差', size = 10)
        plt.ylabel('平均值', size = 10)
        plt.title('年化等權重投資組合', size=12)
        plt.legend(fontsize=8)
        plt.savefig(f'{self.included}/年化等權重投資組合.png', dpi=300)
        plt.close()


    def run_regression(self):
        # 將每月的利率從每個月的資料中減去
        for month in self.free_rate_data.index.month.unique():
            self.free_rate_data.loc[self.free_rate_data.index.month == month] -= self.rate_df.iloc[month]
        print(f'self.free_rate_data: \n{self.free_rate_data}')
        # 進行迴歸分析並獲取結果
        self.regression_result = asyncio.run(self.get_regression())
        self.stats_df = self.get_statistics(self.regression_result)

        w = np.array([1 / len(self.cov)] * len(self.cov))
        mean = np.sum(self.u * w)
        var = np.matmul(np.matmul(w, self.cov), w.T)
        std = np.sqrt(var)
        print(f'portfolio mean: {mean}')
        print(f'portfolio var: {var}')
        print(f'portfolio std: {mean}')
        self.data_describe.loc['等權重', 'mean'] = mean
        self.data_describe.loc['等權重', 'std'] = std
        
        self.industry_plot(std, mean)
        self.save_data()
        
        return self.data_describe, self.regression_result, self.stats_df


    def save_data(self):
        #file_path = f'{self.included}/產業回歸.xlsx'
        file_path = f'{self.included}/{self.included}資料.xlsx'
        mode = 'a' if os.path.exists(file_path) else 'w'
        with pd.ExcelWriter(file_path, mode=mode, engine='openpyxl') as writer:
            if self.free_rate_data is not None:
                if '無風險ln報酬率' in writer.book.sheetnames:
                    writer.book.remove(writer.book['無風險ln報酬率'])
                self.free_rate_data.to_excel(writer, sheet_name='無風險ln報酬率')
            if self.regression_result is not None:
                if '回歸結果' in writer.book.sheetnames:
                    writer.book.remove(writer.book['回歸結果'])
                self.regression_result.to_excel(writer, sheet_name='回歸結果')
            if self.stats_df is not None:
                if '全產業回歸' in writer.book.sheetnames:
                    writer.book.remove(writer.book['全產業回歸'])
                self.stats_df.to_excel(writer, sheet_name='全產業回歸')
            if self.data_describe is not None:
                if '年化基礎統計' in writer.book.sheetnames:
                    writer.book.remove(writer.book['年化基礎統計'])
                self.data_describe.to_excel(writer, sheet_name='年化基礎統計')
            if self.data_cov is not None:
                if '變異數-共變異數矩陣' in writer.book.sheetnames:
                    writer.book.remove(writer.book['變異數-共變異數矩陣'])
                self.data_cov.to_excel(writer, sheet_name='變異數-共變異數矩陣')
            if self.data_corr is not None:
                if '相關係數矩陣' in writer.book.sheetnames:
                    writer.book.remove(writer.book['相關係數矩陣'])
                self.data_corr.to_excel(writer, sheet_name='相關係數矩陣')


    
