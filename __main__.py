from IndustryData import * #下載產業資料
from IndustryRegression import * #計算產業指數回歸
from EfficiencyFrontier import * #計算效率前緣與資產配置線


if __name__ == "__main__" :
    #綠能環保 數位雲端 運動休閒 居家生活 水泥工業 食品工業 塑膠工業 紡織纖維 電機機械 電器電纜 玻璃陶瓷
    #造紙工業 鋼鐵工業 橡膠工業 汽車工業 建材營造業 航運業 觀光餐旅 金融保險業 貿易百貨業 其他業 化學工業
    #生技醫療業 油電燃氣業 半導體業 電腦及週邊設備業 光電業 通信網路業 其他電子業 電子通路業 資訊服務業 電子零組件業
    
    start_date = "2019-01-01"
    end_date = "2023-12-31"
    free_risk = "1y" #1m, 3m, 6m, 9m, 1y, 2y, 3y
    included = "電腦及週邊設備業"
    
    
    #----------------------下載產業資料----------------------
    industry_data = IndustryData(included, start_date, end_date)
    data, rate_df = industry_data.get_data(free_risk)
    
    #----------------------計算產業指數回歸----------------------
    industry_regression = IndustryRegression(included, data, rate_df)
    data_describe, regression_result, stats_df = industry_regression.run_regression()  

    free_risk = 0.01
    num_iterations = 10000
    interval = "1d"  # "1d", "1wk", "1mo"
    data = data.iloc[:,1:] # 拿掉產業指數
    
    #----------------------計算效率前緣與資產配置線----------------------
    efficiency_frontier = EfficiencyFrontier(included, data, free_risk, interval)
    EF_df = efficiency_frontier.unlimit_solution() # 計算產業效率前緣
    limiteEF_df = efficiency_frontier.limit_solution(num_iterations) # 計算限制式效率前緣
    CAL_df = efficiency_frontier.capital_allocation() # 計算資產配置線

