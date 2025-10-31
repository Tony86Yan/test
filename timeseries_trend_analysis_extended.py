"""
?????????????????
??????????????????????/??/????????
???????????????
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
import warnings
import logging

warnings.filterwarnings('ignore')


logger = logging.getLogger()
logging.basicConfig(
    filename='./logs/timeseries_trend_analysis.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8'
)


def calculate_churn_features(df):
    """
    ??SQL???????????
    
    Args:
        df: DataFrame with columns ['merchant_id', 'merchant_name', '_shop_id', '_shop_name', 
                                     'check_date', 'shop_create_date', 'date', 'amount', 'transaction_count']
    
    Returns:
        dict: ?????????
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # ??????????????????????
    current_date = df['date'].max()
    
    # ??????
    last_month_start = current_date - pd.DateOffset(months=1)
    month_before_last_start = current_date - pd.DateOffset(months=2)
    
    df['month_label'] = df['date'].apply(lambda x: 
        'Last 1 Month' if x >= last_month_start 
        else 'The Month Before Last' if x >= month_before_last_start 
        else 'Earlier'
    )
    
    # ???????????
    df_recent = df[df['month_label'].isin(['Last 1 Month', 'The Month Before Last'])].copy()
    
    if len(df_recent) == 0:
        return None
    
    # ?shop?????????
    shop_month_stats = df_recent.groupby(['_shop_id', 'month_label']).agg({
        'amount': ['sum', 'mean'],
        'transaction_count': 'sum'
    }).reset_index()
    
    shop_month_stats.columns = ['_shop_id', 'month_label', 'total_amount', 'mean_amount', 'total_count']
    
    # ???????Active Score?
    active_scores = []
    
    for shop_id in df_recent['_shop_id'].unique():
        shop_data = df_recent[df_recent['_shop_id'] == shop_id].copy()
        
        for month_label in ['Last 1 Month', 'The Month Before Last']:
            month_data = shop_data[shop_data['month_label'] == month_label].copy()
            
            if len(month_data) == 0:
                continue
            
            # ??????pt_diff?
            if month_label == 'Last 1 Month':
                reference_date = current_date
            else:
                reference_date = last_month_start
            
            month_data['pt_diff'] = (reference_date - month_data['date']).dt.days
            
            # ???shop???????
            mean_amount = shop_month_stats[
                (shop_month_stats['_shop_id'] == shop_id) & 
                (shop_month_stats['month_label'] == month_label)
            ]['mean_amount'].values
            
            if len(mean_amount) > 0 and mean_amount[0] > 0:
                mean_amt = mean_amount[0]
                # ??????: sum(POW(0.95, pt_diff) * (amount/mean_amount))
                month_data['score_component'] = np.power(0.95, month_data['pt_diff']) * (month_data['amount'] / mean_amt)
                score = month_data['score_component'].sum()
                
                active_scores.append({
                    '_shop_id': shop_id,
                    'month_label': month_label,
                    'active_score': score
                })
    
    active_score_df = pd.DataFrame(active_scores)
    
    # ????
    result_df = shop_month_stats.merge(active_score_df, on=['_shop_id', 'month_label'], how='left')
    result_df['active_score'] = result_df['active_score'].fillna(0)
    
    # ??????shop????????????
    pivot_amount = result_df.pivot_table(
        index='_shop_id', 
        columns='month_label', 
        values='total_amount', 
        fill_value=0
    )
    
    pivot_score = result_df.pivot_table(
        index='_shop_id', 
        columns='month_label', 
        values='active_score', 
        fill_value=0
    )
    
    # ????
    features = pd.DataFrame()
    features['shop_id'] = pivot_amount.index
    
    # ??????
    features['last_m_before_last_tx_amount'] = pivot_amount.get('The Month Before Last', 0).values
    features['last_m_tx_amount'] = pivot_amount.get('Last 1 Month', 0).values
    
    # ??????
    features['last_m_before_last_active_score'] = pivot_score.get('The Month Before Last', 0).values
    features['last_m_active_score'] = pivot_score.get('Last 1 Month', 0).values
    
    # ??????
    features['month_tx_amount_decay_score'] = (features['last_m_before_last_tx_amount'] + 1000) / (features['last_m_tx_amount'] + 1000)
    features['suspected_churn_rate'] = (features['last_m_before_last_active_score'] + 1) / (features['last_m_active_score'] + 1)
    
    # ????????
    merchant_info = df_recent.groupby('_shop_id').first()[['merchant_id', 'merchant_name', '_shop_name', 'check_date', 'shop_create_date']].reset_index()
    features = features.merge(merchant_info, left_on='shop_id', right_on='_shop_id', how='left')
    
    return features


class TimeSeriesTrendAnalyzer:
    """?????????"""

    def __init__(self, dates, amounts, df, churn_features=None):
        """
        ??????

        Args:
            dates: ???????
            amounts: ??????????
            df: ?????
            churn_features: ??????????????
        """
        self.df = df
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.df['time_index'] = range(len(self.df))
        self.churn_features = churn_features

    def method1_linear_regression(self):
        """
        ??1: ??????
        - ???????
        - ?????R??????????
        """
        X = self.df['time_index'].values.reshape(-1, 1)
        y = self.df['amount'].values

        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(y, model.predict(X))

        # ???????p?
        y_pred = model.predict(X)
        residuals = y - y_pred
        std_error = np.sqrt(np.sum(residuals ** 2) / (len(y) - 2))

        # ???????
        x_mean = np.mean(X)
        slope_se = std_error / np.sqrt(np.sum((X - x_mean) ** 2))
        t_stat = slope / slope_se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 2))

        # ????
        if p_value > 0.05:
            trend = "??"
            degree = "?????"
        else:
            if slope > 0:
                trend = "??"
            else:
                trend = "??"

            # ?????R?????
            normalized_slope = abs(slope) / (np.std(y) + 1e-10)
            if r2 > 0.7 and normalized_slope > 0.1:
                degree = "?"
            elif r2 > 0.4 or normalized_slope > 0.05:
                degree = "??"
            else:
                degree = "?"

        return {
            'method': '????',
            'trend': trend,
            'degree': degree,
            'slope': slope,
            'r2': r2,
            'p_value': p_value,
            'normalized_slope': abs(slope) / (np.std(y) + 1e-10)
        }

    def method2_mann_kendall(self):
        """
        ??2: Mann-Kendall??
        - ?????????????
        - ???????????????
        """
        data = self.df['amount'].values
        n = len(data)

        # ??S???
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(data[j] - data[i])

        # ????
        var_s = n * (n - 1) * (2 * n + 5) / 18

        # ??Z???
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0

        # ??p?
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Kendall's Tau (????)
        tau = s / (0.5 * n * (n - 1))

        # ????
        if p_value > 0.05:
            trend = "??"
            degree = "?????"
        else:
            if tau > 0:
                trend = "??"
            else:
                trend = "??"

            # ??tau?????
            abs_tau = abs(tau)
            if abs_tau > 0.5:
                degree = "?"
            elif abs_tau > 0.3:
                degree = "??"
            else:
                degree = "?"

        return {
            'method': 'Mann-Kendall??',
            'trend': trend,
            'degree': degree,
            'tau': tau,
            'z_score': z,
            'p_value': p_value
        }

    def method3_polynomial_regression(self, degree=2):
        """
        ??3: ?????
        - ?????????
        - ????????????????
        """
        X = self.df['time_index'].values.reshape(-1, 1)
        y = self.df['amount'].values

        # ?????
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)

        # ???????????
        if degree == 2:
            b = model.coef_[1]
            c = model.coef_[2]

            # ???????
            mid_point = len(X) / 2
            derivative = b + 2 * c * mid_point

            # ?????????
            if abs(c) > 0.001:
                if c > 0:
                    curvature = "????" if derivative > 0 else "????"
                else:
                    curvature = "????" if derivative > 0 else "????"
            else:
                curvature = "????"
        else:
            derivative = np.mean(np.gradient(y_pred))
            curvature = "?????"

        # ??????
        start_val = y_pred[0]
        end_val = y_pred[-1]
        change_rate = (end_val - start_val) / (abs(start_val) + 1e-10)

        if abs(change_rate) < 0.05:
            trend = "??"
            degree_str = "????"
        else:
            if end_val > start_val:
                trend = "??"
            else:
                trend = "??"

            if r2 > 0.7 and abs(change_rate) > 0.2:
                degree_str = "?"
            elif r2 > 0.4 or abs(change_rate) > 0.1:
                degree_str = "??"
            else:
                degree_str = "?"

        return {
            'method': f'{degree}??????',
            'trend': trend,
            'degree': degree_str,
            'curvature': curvature,
            'r2': r2,
            'change_rate': change_rate
        }

    def method4_moving_average_slope(self, window=7):
        """
        ??4: ????????
        - ????????????
        - ???????????
        """
        data = self.df['amount'].values

        # ??????
        ma = pd.Series(data).rolling(window=window, center=True).mean()

        # ?????????
        slopes = np.gradient(ma.dropna())

        avg_slope = np.nanmean(slopes)
        slope_std = np.nanstd(slopes)

        # ?????
        positive_slopes = np.sum(slopes > 0)
        negative_slopes = np.sum(slopes < 0)
        total_slopes = len(slopes)

        consistency = max(positive_slopes, negative_slopes) / total_slopes

        # ????
        if consistency < 0.6:
            trend = "??"
            degree = "???????"
        else:
            if avg_slope > 0:
                trend = "??"
            else:
                trend = "??"

            # ???????????????
            if consistency > 0.8 and slope_std < np.abs(avg_slope):
                degree = "?"
            elif consistency > 0.7:
                degree = "??"
            else:
                degree = "?"

        return {
            'method': f'??????(??={window})',
            'trend': trend,
            'degree': degree,
            'avg_slope': avg_slope,
            'consistency': consistency,
            'slope_std': slope_std
        }

    def method5_spearman_correlation(self):
        """
        ??5: Spearman?????
        - ??????????
        - ?????????????
        """
        time_index = self.df['time_index'].values
        amounts = self.df['amount'].values

        rho, p_value = stats.spearmanr(time_index, amounts)

        # ????
        if p_value > 0.05:
            trend = "??"
            degree = "?????"
        else:
            if rho > 0:
                trend = "??"
            else:
                trend = "??"

            abs_rho = abs(rho)
            if abs_rho > 0.7:
                degree = "?"
            elif abs_rho > 0.4:
                degree = "??"
            else:
                degree = "?"

        return {
            'method': 'Spearman???',
            'trend': trend,
            'degree': degree,
            'rho': rho,
            'p_value': p_value
        }

    def method6_detrended_fluctuation(self):
        """
        ??6: ??????? (???)
        - ?????????
        - ??????????
        """
        data = self.df['amount'].values

        # ??????
        mean_val = np.mean(data)
        cumsum = np.cumsum(data - mean_val)

        # ????????
        X = np.arange(len(cumsum)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, cumsum)

        trend_strength = abs(model.coef_[0]) / (np.std(data) + 1e-10)

        # ???????
        residuals = cumsum - model.predict(X)
        fluctuation = np.std(residuals)

        # ????
        if trend_strength < 0.1:
            trend = "??"
            degree = "????"
        else:
            if model.coef_[0] > 0:
                trend = "??"
            else:
                trend = "??"

            if trend_strength > 0.5:
                degree = "?"
            elif trend_strength > 0.2:
                degree = "??"
            else:
                degree = "?"

        return {
            'method': '???????',
            'trend': trend,
            'degree': degree,
            'trend_strength': trend_strength,
            'fluctuation': fluctuation
        }

    def method7_change_point_detection(self):
        """
        ??7: ????
        - ??????????
        - ??????
        """
        data = self.df['amount'].values
        n = len(data)

        if n < 10:
            return {
                'method': '????',
                'trend': '?????',
                'degree': 'N/A'
            }

        # ?????????????????
        best_split = n // 2
        max_variance_diff = 0

        for i in range(n // 4, 3 * n // 4):
            var1 = np.var(data[:i])
            var2 = np.var(data[i:])
            var_diff = abs(var1 - var2)

            if var_diff > max_variance_diff:
                max_variance_diff = var_diff
                best_split = i

        # ???????
        seg1_slope = np.polyfit(range(best_split), data[:best_split], 1)[0]
        seg2_slope = np.polyfit(range(len(data[best_split:])), data[best_split:], 1)[0]

        # ????
        overall_slope = (seg1_slope + seg2_slope) / 2

        if abs(overall_slope) < 0.01 * np.std(data):
            trend = "??"
            degree = "????"
        else:
            if overall_slope > 0:
                trend = "??"
            else:
                trend = "??"

            # ????????
            if np.sign(seg1_slope) == np.sign(seg2_slope):
                degree = "????"
            else:
                degree = "????"

        return {
            'method': '????',
            'trend': trend,
            'degree': degree,
            'change_point': self.df['date'].iloc[best_split],
            'seg1_slope': seg1_slope,
            'seg2_slope': seg2_slope
        }

    def method8_suspected_churn_rate(self):
        """
        ??8: ??Suspected Churn Rate(Last 2 Months)?????
        - ??????????????
        - ?????????????
        """
        if self.churn_features is None:
            return {
                'method': 'Suspected Churn Rate??',
                'trend': 'N/A',
                'degree': '?????',
                'churn_rate': None
            }
        
        churn_rate = self.churn_features.get('suspected_churn_rate', None)
        
        if churn_rate is None or pd.isna(churn_rate):
            return {
                'method': 'Suspected Churn Rate??',
                'trend': 'N/A',
                'degree': '?????',
                'churn_rate': None
            }
        
        # ???????????
        if churn_rate > 2:
            trend = "??"
            if churn_rate > 4:
                degree = "?"
            elif churn_rate > 3:
                degree = "??"
            else:
                degree = "?"
        elif churn_rate < 0.4:
            trend = "??"
            if churn_rate < 0.2:
                degree = "?"
            elif churn_rate < 0.3:
                degree = "??"
            else:
                degree = "?"
        else:
            trend = "??"
            degree = "????"
        
        # ??????
        if churn_rate > 4:
            risk_level = "???"
        elif churn_rate > 2:
            risk_level = "????"
        elif churn_rate < 0.4:
            risk_level = "????"
        else:
            risk_level = "??"
        
        return {
            'method': 'Suspected Churn Rate??',
            'trend': trend,
            'degree': degree,
            'churn_rate': churn_rate,
            'risk_level': risk_level,
            'last_m_before_last_active_score': self.churn_features.get('last_m_before_last_active_score'),
            'last_m_active_score': self.churn_features.get('last_m_active_score')
        }

    def method9_month_tx_amount_decay(self):
        """
        ??9: ??Month TX Amount Decay Score?????
        - ??????????????????
        - ?????????????
        """
        if self.churn_features is None:
            return {
                'method': 'Month TX Amount Decay??',
                'trend': 'N/A',
                'degree': '?????',
                'decay_score': None
            }
        
        decay_score = self.churn_features.get('month_tx_amount_decay_score', None)
        
        if decay_score is None or pd.isna(decay_score):
            return {
                'method': 'Month TX Amount Decay??',
                'trend': 'N/A',
                'degree': '?????',
                'decay_score': None
            }
        
        # ???????????
        if decay_score > 2:
            trend = "??"
            if decay_score > 5:
                degree = "?"
            elif decay_score > 3.5:
                degree = "??"
            else:
                degree = "?"
        elif decay_score < 0.35:
            trend = "??"
            if decay_score < 0.12:
                degree = "?"
            elif decay_score < 0.24:
                degree = "??"
            else:
                degree = "?"
        else:
            trend = "??"
            degree = "????"
        
        # ??????
        if decay_score > 5:
            business_status = "????"
        elif decay_score > 2:
            business_status = "????"
        elif decay_score < 0.35:
            business_status = "????"
        else:
            business_status = "??"
        
        return {
            'method': 'Month TX Amount Decay??',
            'trend': trend,
            'degree': degree,
            'decay_score': decay_score,
            'business_status': business_status,
            'last_m_before_last_tx_amount': self.churn_features.get('last_m_before_last_tx_amount'),
            'last_m_tx_amount': self.churn_features.get('last_m_tx_amount')
        }

    def comprehensive_analysis(self):
        """
        ??????????????????
        """
        results = []

        # ????????
        results.append(self.method1_linear_regression())
        results.append(self.method2_mann_kendall())
        results.append(self.method3_polynomial_regression(degree=2))
        results.append(self.method3_polynomial_regression(degree=3))
        results.append(self.method4_moving_average_slope(window=5))
        results.append(self.method5_spearman_correlation())
        results.append(self.method6_detrended_fluctuation())
        results.append(self.method7_change_point_detection())
        
        # ????
        results.append(self.method8_suspected_churn_rate())
        results.append(self.method9_month_tx_amount_decay())

        # ??????
        trends = [r['trend'] for r in results if r['trend'] in ['??', '??', '??']]
        
        if len(trends) == 0:
            return {
                'individual_results': results,
                'consensus_trend': 'N/A',
                'consensus_degree': 'N/A',
                'confidence': 0,
                'trend_votes': {}
            }
        
        trend_counts = {
            '??': trends.count('??'),
            '??': trends.count('??'),
            '??': trends.count('??')
        }

        # ??????
        consensus_trend = max(trend_counts, key=trend_counts.get)
        consensus_confidence = trend_counts[consensus_trend] / len(trends) * 100

        # ????
        degrees = [r['degree'] for r in results if r['degree'] in ['?', '??', '?']]
        if degrees:
            degree_counts = {
                '?': degrees.count('?'),
                '??': degrees.count('??'),
                '?': degrees.count('?')
            }
            consensus_degree = max(degree_counts, key=degree_counts.get)
        else:
            consensus_degree = '?????'

        return {
            'individual_results': results,
            'consensus_trend': consensus_trend,
            'consensus_degree': consensus_degree,
            'confidence': consensus_confidence,
            'trend_votes': trend_counts
        }


def _preprocess_data(transaction_data, min_pt, max_pt):
    """?????"""
    df = transaction_data.copy()
    df['date'] = pd.to_datetime(df['date'])

    # ????????
    date_range = pd.date_range(
        start=min_pt,
        end=max_pt,
        freq='D'
    )

    daily_agg = (
        df.set_index('date')
        .reindex(date_range, fill_value=0)
        .rename_axis('date')
        .reset_index()
    )

    # ????????? DataFrame ???????
    for col in ['merchant_id', 'merchant_name', '_shop_id', '_shop_name', 'check_date', 'shop_create_date']:
        if col in df.columns:
            daily_agg[col] = df[col].iloc[0]

    return daily_agg


def main():
    """????????"""
    import os
    
    # ??????
    os.makedirs('./logs', exist_ok=True)
    
    print("=" * 80)
    print("???????????????")
    print("=" * 80)

    # ???????
    # mer_df = pd.read_csv("your_data.csv", sep="@@@")
    # ????????
    
    # ????
    np.random.seed(42)
    dates = pd.date_range(start='2025-07-03', periods=120, freq='D')
    
    # ????????????
    base_amount = 10000
    trend = -50  # ????50
    amounts = [base_amount + trend * i + np.random.normal(0, 500) for i in range(len(dates))]
    amounts = [max(0, a) for a in amounts]  # ????
    
    mer_df = pd.DataFrame({
        'merchant_id': ['BC00001234'] * len(dates),
        'merchant_name': ['????'] * len(dates),
        '_shop_id': ['BC00001234001'] * len(dates),
        '_shop_name': ['????'] * len(dates),
        'check_date': ['2025-01-01'] * len(dates),
        'shop_create_date': ['2025-01-01'] * len(dates),
        'date': dates,
        'amount': amounts,
        'transaction_count': np.random.randint(10, 100, len(dates))
    })
    
    print(f"\n????:")
    print(mer_df.head(10))
    
    # ??????
    min_pt = mer_df['date'].min()
    max_pt = mer_df['date'].max()
    print(f"\n????: {min_pt} ? {max_pt}")
    
    # ???????
    print("\n?????????...")
    churn_features_df = calculate_churn_features(mer_df)
    
    if churn_features_df is not None and len(churn_features_df) > 0:
        print("\n?????:")
        print(churn_features_df[['shop_id', 'month_tx_amount_decay_score', 'suspected_churn_rate']])
        
        # ?????????
        for shop_id in churn_features_df['shop_id'].values:
            shop_data = mer_df[mer_df['_shop_id'] == shop_id].copy()
            shop_features = churn_features_df[churn_features_df['shop_id'] == shop_id].iloc[0].to_dict()
            
            print(f"\n\n{'=' * 80}")
            print(f"????: {shop_id}")
            print(f"{'=' * 80}\n")
            
            # ?????
            processed_data = _preprocess_data(shop_data, min_pt, max_pt)
            
            # ?????
            analyzer = TimeSeriesTrendAnalyzer(
                dates=processed_data['date'],
                amounts=processed_data['amount'],
                df=processed_data,
                churn_features=shop_features
            )
            
            # ??????
            result = analyzer.comprehensive_analysis()
            
            # ????
            print("???????:")
            print("-" * 80)
            for r in result['individual_results']:
                print(f"\n{r['method']}:")
                print(f"  ??: {r['trend']}")
                print(f"  ??: {r['degree']}")
                for key, value in r.items():
                    if key not in ['method', 'trend', 'degree']:
                        if isinstance(value, float):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
            
            print("\n" + "=" * 80)
            print("????:")
            print("=" * 80)
            print(f"????: {result['consensus_trend']}")
            print(f"????: {result['consensus_degree']}")
            print(f"???: {result['confidence']:.1f}%")
            print(f"????: {result['trend_votes']}")
            
            # ?????
            logging.info(f"\n\n{'=' * 80}")
            logging.info(f"??: {shop_id}")
            logging.info(f"????: {result['consensus_trend']}, ??: {result['consensus_degree']}, ???: {result['confidence']:.1f}%")
            logging.info(f"{'=' * 80}\n")
    else:
        print("?????????")


if __name__ == "__main__":
    main()
