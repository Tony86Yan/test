"""
时间序列趋势识别方法集合
支持线性和非线性趋势识别，包括趋势方向（上升/下降/平稳）和程度量化
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesTrendAnalyzer:
    """时间序列趋势分析器"""
    
    def __init__(self, dates, amounts):
        """
        初始化分析器
        
        Args:
            dates: 日期列表或数组
            amounts: 对应的数值列表或数组
        """
        self.df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'amount': amounts
        })
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.df['time_index'] = range(len(self.df))
        
    def method1_linear_regression(self):
        """
        方法1: 线性回归分析
        - 最适合线性趋势
        - 返回斜率、R²、趋势方向和显著性
        """
        X = self.df['time_index'].values.reshape(-1, 1)
        y = self.df['amount'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(y, model.predict(X))
        
        # 计算标准误差和p值
        y_pred = model.predict(X)
        residuals = y - y_pred
        std_error = np.sqrt(np.sum(residuals**2) / (len(y) - 2))
        
        # 斜率的标准误差
        x_mean = np.mean(X)
        slope_se = std_error / np.sqrt(np.sum((X - x_mean)**2))
        t_stat = slope / slope_se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 2))
        
        # 判断趋势
        if p_value > 0.05:
            trend = "平稳"
            degree = "无显著趋势"
        else:
            if slope > 0:
                trend = "上升"
            else:
                trend = "下降"
            
            # 根据斜率和R²判断程度
            normalized_slope = abs(slope) / (np.std(y) + 1e-10)
            if r2 > 0.7 and normalized_slope > 0.1:
                degree = "强"
            elif r2 > 0.4 or normalized_slope > 0.05:
                degree = "中等"
            else:
                degree = "弱"
        
        return {
            'method': '线性回归',
            'trend': trend,
            'degree': degree,
            'slope': slope,
            'r2': r2,
            'p_value': p_value,
            'normalized_slope': abs(slope) / (np.std(y) + 1e-10)
        }
    
    def method2_mann_kendall(self):
        """
        方法2: Mann-Kendall检验
        - 非参数方法，不假设数据分布
        - 适合有异常值或非线性趋势的情况
        """
        data = self.df['amount'].values
        n = len(data)
        
        # 计算S统计量
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(data[j] - data[i])
        
        # 计算方差
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # 计算Z统计量
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # 计算p值
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Kendall's Tau (趋势强度)
        tau = s / (0.5 * n * (n - 1))
        
        # 判断趋势
        if p_value > 0.05:
            trend = "平稳"
            degree = "无显著趋势"
        else:
            if tau > 0:
                trend = "上升"
            else:
                trend = "下降"
            
            # 根据tau值判断程度
            abs_tau = abs(tau)
            if abs_tau > 0.5:
                degree = "强"
            elif abs_tau > 0.3:
                degree = "中等"
            else:
                degree = "弱"
        
        return {
            'method': 'Mann-Kendall检验',
            'trend': trend,
            'degree': degree,
            'tau': tau,
            'z_score': z,
            'p_value': p_value
        }
    
    def method3_polynomial_regression(self, degree=2):
        """
        方法3: 多项式回归
        - 适合非线性趋势识别
        - 可以识别曲线趋势（如二次、三次）
        """
        X = self.df['time_index'].values.reshape(-1, 1)
        y = self.df['amount'].values
        
        # 拟合多项式
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        
        # 计算一阶导数来判断趋势
        # 对于二次多项式: y = a + bx + cx²，导数 = b + 2cx
        if degree == 2:
            b = model.coef_[1]
            c = model.coef_[2]
            
            # 在中点评估导数
            mid_point = len(X) / 2
            derivative = b + 2 * c * mid_point
            
            # 判断是否加速或减速
            if abs(c) > 0.001:
                if c > 0:
                    curvature = "加速上升" if derivative > 0 else "先降后升"
                else:
                    curvature = "减速上升" if derivative > 0 else "加速下降"
            else:
                curvature = "近似线性"
        else:
            derivative = np.mean(np.gradient(y_pred))
            curvature = "复杂非线性"
        
        # 判断整体趋势
        start_val = y_pred[0]
        end_val = y_pred[-1]
        change_rate = (end_val - start_val) / (abs(start_val) + 1e-10)
        
        if abs(change_rate) < 0.05:
            trend = "平稳"
            degree_str = "波动较小"
        else:
            if end_val > start_val:
                trend = "上升"
            else:
                trend = "下降"
            
            if r2 > 0.7 and abs(change_rate) > 0.2:
                degree_str = "强"
            elif r2 > 0.4 or abs(change_rate) > 0.1:
                degree_str = "中等"
            else:
                degree_str = "弱"
        
        return {
            'method': f'{degree}次多项式回归',
            'trend': trend,
            'degree': degree_str,
            'curvature': curvature,
            'r2': r2,
            'change_rate': change_rate
        }
    
    def method4_moving_average_slope(self, window=5):
        """
        方法4: 移动平均斜率分析
        - 基于移动平均线的斜率变化
        - 适合识别短期和长期趋势
        """
        data = self.df['amount'].values
        
        # 计算移动平均
        ma = pd.Series(data).rolling(window=window, center=True).mean()
        
        # 计算移动平均的斜率
        slopes = np.gradient(ma.dropna())
        
        avg_slope = np.nanmean(slopes)
        slope_std = np.nanstd(slopes)
        
        # 趋势一致性
        positive_slopes = np.sum(slopes > 0)
        negative_slopes = np.sum(slopes < 0)
        total_slopes = len(slopes)
        
        consistency = max(positive_slopes, negative_slopes) / total_slopes
        
        # 判断趋势
        if consistency < 0.6:
            trend = "平稳"
            degree = "波动无明显方向"
        else:
            if avg_slope > 0:
                trend = "上升"
            else:
                trend = "下降"
            
            # 根据一致性和斜率标准差判断程度
            if consistency > 0.8 and slope_std < np.abs(avg_slope):
                degree = "强"
            elif consistency > 0.7:
                degree = "中等"
            else:
                degree = "弱"
        
        return {
            'method': f'移动平均斜率(窗口={window})',
            'trend': trend,
            'degree': degree,
            'avg_slope': avg_slope,
            'consistency': consistency,
            'slope_std': slope_std
        }
    
    def method5_spearman_correlation(self):
        """
        方法5: Spearman秩相关系数
        - 非参数方法，基于排名
        - 对异常值鲁棒，适合单调趋势
        """
        time_index = self.df['time_index'].values
        amounts = self.df['amount'].values
        
        rho, p_value = stats.spearmanr(time_index, amounts)
        
        # 判断趋势
        if p_value > 0.05:
            trend = "平稳"
            degree = "无显著趋势"
        else:
            if rho > 0:
                trend = "上升"
            else:
                trend = "下降"
            
            abs_rho = abs(rho)
            if abs_rho > 0.7:
                degree = "强"
            elif abs_rho > 0.4:
                degree = "中等"
            else:
                degree = "弱"
        
        return {
            'method': 'Spearman秩相关',
            'trend': trend,
            'degree': degree,
            'rho': rho,
            'p_value': p_value
        }
    
    def method6_detrended_fluctuation(self):
        """
        方法6: 去趋势波动分析 (简化版)
        - 分析数据的波动特性
        - 可识别长期趋势的强度
        """
        data = self.df['amount'].values
        
        # 计算累积离差
        mean_val = np.mean(data)
        cumsum = np.cumsum(data - mean_val)
        
        # 线性拟合累积离差
        X = np.arange(len(cumsum)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, cumsum)
        
        trend_strength = abs(model.coef_[0]) / (np.std(data) + 1e-10)
        
        # 计算残差的波动
        residuals = cumsum - model.predict(X)
        fluctuation = np.std(residuals)
        
        # 判断趋势
        if trend_strength < 0.1:
            trend = "平稳"
            degree = "随机波动"
        else:
            if model.coef_[0] > 0:
                trend = "上升"
            else:
                trend = "下降"
            
            if trend_strength > 0.5:
                degree = "强"
            elif trend_strength > 0.2:
                degree = "中等"
            else:
                degree = "弱"
        
        return {
            'method': '去趋势波动分析',
            'trend': trend,
            'degree': degree,
            'trend_strength': trend_strength,
            'fluctuation': fluctuation
        }
    
    def method7_change_point_detection(self):
        """
        方法7: 变点检测
        - 识别趋势改变的时间点
        - 分段分析趋势
        """
        data = self.df['amount'].values
        n = len(data)
        
        if n < 10:
            return {
                'method': '变点检测',
                'trend': '数据点太少',
                'degree': 'N/A'
            }
        
        # 简单的变点检测：找到最大方差变化点
        best_split = n // 2
        max_variance_diff = 0
        
        for i in range(n // 4, 3 * n // 4):
            var1 = np.var(data[:i])
            var2 = np.var(data[i:])
            var_diff = abs(var1 - var2)
            
            if var_diff > max_variance_diff:
                max_variance_diff = var_diff
                best_split = i
        
        # 分析两段的趋势
        seg1_slope = np.polyfit(range(best_split), data[:best_split], 1)[0]
        seg2_slope = np.polyfit(range(len(data[best_split:])), data[best_split:], 1)[0]
        
        # 整体趋势
        overall_slope = (seg1_slope + seg2_slope) / 2
        
        if abs(overall_slope) < 0.01 * np.std(data):
            trend = "平稳"
            degree = "整体稳定"
        else:
            if overall_slope > 0:
                trend = "上升"
            else:
                trend = "下降"
            
            # 检查趋势是否一致
            if np.sign(seg1_slope) == np.sign(seg2_slope):
                degree = "一致趋势"
            else:
                degree = "趋势转折"
        
        return {
            'method': '变点检测',
            'trend': trend,
            'degree': degree,
            'change_point': self.df['date'].iloc[best_split],
            'seg1_slope': seg1_slope,
            'seg2_slope': seg2_slope
        }
    
    def comprehensive_analysis(self):
        """
        综合分析：使用所有方法并给出综合结论
        """
        results = []
        
        # 执行所有分析方法
        results.append(self.method1_linear_regression())
        results.append(self.method2_mann_kendall())
        results.append(self.method3_polynomial_regression(degree=2))
        results.append(self.method3_polynomial_regression(degree=3))
        results.append(self.method4_moving_average_slope(window=5))
        results.append(self.method5_spearman_correlation())
        results.append(self.method6_detrended_fluctuation())
        results.append(self.method7_change_point_detection())
        
        # 统计趋势投票
        trends = [r['trend'] for r in results if r['trend'] in ['上升', '下降', '平稳']]
        trend_counts = {
            '上升': trends.count('上升'),
            '下降': trends.count('下降'),
            '平稳': trends.count('平稳')
        }
        
        # 确定综合趋势
        consensus_trend = max(trend_counts, key=trend_counts.get)
        consensus_confidence = trend_counts[consensus_trend] / len(trends) * 100
        
        # 统计程度
        degrees = [r['degree'] for r in results if r['degree'] in ['强', '中等', '弱']]
        if degrees:
            degree_counts = {
                '强': degrees.count('强'),
                '中等': degrees.count('中等'),
                '弱': degrees.count('弱')
            }
            consensus_degree = max(degree_counts, key=degree_counts.get)
        else:
            consensus_degree = '无明显趋势'
        
        return {
            'individual_results': results,
            'consensus_trend': consensus_trend,
            'consensus_degree': consensus_degree,
            'confidence': consensus_confidence,
            'trend_votes': trend_counts
        }


def generate_sample_data():
    """生成示例数据"""
    import datetime
    
    # 示例1: 线性上升趋势
    dates1 = pd.date_range(start='2023-01-01', periods=50, freq='D')
    amounts1 = np.linspace(100, 200, 50) + np.random.normal(0, 5, 50)
    
    # 示例2: 非线性趋势
    dates2 = pd.date_range(start='2023-01-01', periods=50, freq='D')
    x = np.linspace(0, 10, 50)
    amounts2 = 100 + 5 * x**2 + np.random.normal(0, 10, 50)
    
    # 示例3: 平稳趋势
    dates3 = pd.date_range(start='2023-01-01', periods=50, freq='D')
    amounts3 = 150 + np.random.normal(0, 10, 50)
    
    return [
        (dates1, amounts1, "线性上升趋势"),
        (dates2, amounts2, "非线性上升趋势"),
        (dates3, amounts3, "平稳趋势")
    ]


if __name__ == "__main__":
    # 测试示例
    print("=" * 80)
    print("时间序列趋势分析示例")
    print("=" * 80)
    
    samples = generate_sample_data()
    
    for dates, amounts, description in samples:
        print(f"\n\n{'='*80}")
        print(f"测试数据: {description}")
        print(f"{'='*80}\n")
        
        analyzer = TimeSeriesTrendAnalyzer(dates, amounts)
        result = analyzer.comprehensive_analysis()
        
        print("各方法分析结果:")
        print("-" * 80)
        for r in result['individual_results']:
            print(f"\n{r['method']}:")
            print(f"  趋势: {r['trend']}")
            print(f"  程度: {r['degree']}")
            for key, value in r.items():
                if key not in ['method', 'trend', 'degree']:
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
        
        print("\n" + "=" * 80)
        print("综合结论:")
        print("=" * 80)
        print(f"趋势方向: {result['consensus_trend']}")
        print(f"趋势程度: {result['consensus_degree']}")
        print(f"置信度: {result['confidence']:.1f}%")
        print(f"投票分布: {result['trend_votes']}")
