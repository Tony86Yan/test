"""
商户流失预测系统
使用机器学习识别并预测商户是否疑似流失
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class MerchantChurnPredictor:
    """商户流失预测器"""
    
    def __init__(self, min_date, max_date):
        """
        初始化预测器
        
        Args:
            min_date: 数据最小日期
            max_date: 数据最大日期
        """
        self.min_date = pd.to_datetime(min_date)
        self.max_date = pd.to_datetime(max_date)
        self.model = None
        self.scaler = StandardScaler()
        
    def preprocess_data(self, transaction_data):
        """
        数据预处理
        
        Args:
            transaction_data: 商户交易数据DataFrame
            
        Returns:
            预处理后的每日数据
        """
        df = transaction_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # 创建完整日期序列
        date_range = pd.date_range(
            start=self.min_date,
            end=self.max_date,
            freq='D'
        )
        
        # 按日聚合
        daily_data = df.groupby('date').agg({
            'transaction_count': 'sum',
            'amount': 'sum'
        }).reindex(date_range, fill_value=0)
        
        daily_data.index.name = 'date'
        daily_data.reset_index(inplace=True)
        
        return daily_data
    
    def classify_merchant_frequency(self, daily_data):
        """
        分类商户交易频率（高频、中频、低频）
        
        Args:
            daily_data: 每日交易数据
            
        Returns:
            频率类型: 'high', 'medium', 'low'
        """
        # 计算有交易的天数占比
        total_days = len(daily_data)
        active_days = (daily_data['transaction_count'] > 0).sum()
        active_ratio = active_days / total_days
        
        # 计算平均每周活跃天数
        avg_weekly_active_days = active_ratio * 7
        
        if avg_weekly_active_days >= 5:
            # 每周至少5天有交易 -> 高频（如超市）
            return 'high'
        elif avg_weekly_active_days >= 1:
            # 每周至少1天有交易 -> 中频
            return 'medium'
        else:
            # 每周少于1天交易 -> 低频
            return 'low'
    
    def apply_moving_average(self, daily_data, frequency_type):
        """
        根据商户频率应用移动平均平滑数据
        
        Args:
            daily_data: 每日交易数据
            frequency_type: 频率类型 ('high', 'medium', 'low')
            
        Returns:
            平滑后的数据
        """
        df = daily_data.copy()
        
        # 根据频率选择窗口大小
        if frequency_type == 'high':
            window = 7  # 高频商户使用7天窗口
        else:
            window = 30  # 中频和低频商户使用30天窗口
        
        # 应用移动平均
        df['amount_ma'] = df['amount'].rolling(window=window, min_periods=1).mean()
        df['transaction_count_ma'] = df['transaction_count'].rolling(window=window, min_periods=1).mean()
        
        # 同时计算移动标准差（用于识别波动）
        df['amount_std'] = df['amount'].rolling(window=window, min_periods=1).std().fillna(0)
        df['transaction_count_std'] = df['transaction_count'].rolling(window=window, min_periods=1).std().fillna(0)
        
        return df
    
    def engineer_features(self, smoothed_data, lookback_days=30):
        """
        特征工程：提取商户交易趋势特征
        
        Args:
            smoothed_data: 平滑后的数据
            lookback_days: 回溯天数，用于计算特征
            
        Returns:
            特征字典
        """
        df = smoothed_data.copy()
        
        if len(df) < lookback_days:
            lookback_days = len(df)
        
        # 取最近的数据
        recent_data = df.tail(lookback_days).copy()
        
        features = {}
        
        # 1. 基础统计特征
        features['avg_amount'] = recent_data['amount_ma'].mean()
        features['avg_transaction_count'] = recent_data['transaction_count_ma'].mean()
        features['std_amount'] = recent_data['amount_std'].mean()
        features['std_transaction_count'] = recent_data['transaction_count_std'].mean()
        
        # 2. 趋势特征（使用线性回归拟合）
        if len(recent_data) > 1:
            X_trend = np.arange(len(recent_data)).reshape(-1, 1)
            
            # 金额趋势
            y_amount = recent_data['amount_ma'].values
            if np.std(y_amount) > 0:
                from sklearn.linear_model import LinearRegression
                lr_amount = LinearRegression()
                lr_amount.fit(X_trend, y_amount)
                features['amount_trend_slope'] = lr_amount.coef_[0]
                features['amount_trend_r2'] = lr_amount.score(X_trend, y_amount)
            else:
                features['amount_trend_slope'] = 0
                features['amount_trend_r2'] = 0
            
            # 交易笔数趋势
            y_count = recent_data['transaction_count_ma'].values
            if np.std(y_count) > 0:
                lr_count = LinearRegression()
                lr_count.fit(X_trend, y_count)
                features['count_trend_slope'] = lr_count.coef_[0]
                features['count_trend_r2'] = lr_count.score(X_trend, y_count)
            else:
                features['count_trend_slope'] = 0
                features['count_trend_r2'] = 0
        else:
            features['amount_trend_slope'] = 0
            features['amount_trend_r2'] = 0
            features['count_trend_slope'] = 0
            features['count_trend_r2'] = 0
        
        # 3. 周期性特征
        # 比较最近一周vs前三周
        if len(recent_data) >= 28:
            last_week = recent_data.tail(7)
            prev_3weeks = recent_data.iloc[-28:-7]
            
            features['amount_decay_ratio'] = (
                last_week['amount_ma'].mean() / (prev_3weeks['amount_ma'].mean() + 1e-6)
            )
            features['count_decay_ratio'] = (
                last_week['transaction_count_ma'].mean() / (prev_3weeks['transaction_count_ma'].mean() + 1e-6)
            )
        else:
            features['amount_decay_ratio'] = 1.0
            features['count_decay_ratio'] = 1.0
        
        # 4. 活跃度特征
        features['active_days_ratio'] = (recent_data['transaction_count'] > 0).sum() / len(recent_data)
        features['zero_transaction_days'] = (recent_data['transaction_count'] == 0).sum()
        
        # 5. 最近活动特征
        # 最后一次交易距今天数
        last_transaction_idx = recent_data[recent_data['transaction_count'] > 0].index
        if len(last_transaction_idx) > 0:
            features['days_since_last_transaction'] = len(recent_data) - (
                recent_data.index.get_loc(last_transaction_idx[-1]) + 1
            )
        else:
            features['days_since_last_transaction'] = lookback_days
        
        # 6. 波动性特征
        features['coefficient_of_variation_amount'] = (
            features['std_amount'] / (features['avg_amount'] + 1e-6)
        )
        features['coefficient_of_variation_count'] = (
            features['std_transaction_count'] / (features['avg_transaction_count'] + 1e-6)
        )
        
        # 7. 非线性趋势检测（二次项系数）
        if len(recent_data) > 2:
            X_poly = np.column_stack([X_trend, X_trend**2])
            
            # 金额的二次趋势
            if np.std(y_amount) > 0:
                from sklearn.linear_model import LinearRegression
                lr_poly_amount = LinearRegression()
                lr_poly_amount.fit(X_poly, y_amount)
                features['amount_quadratic_coef'] = lr_poly_amount.coef_[1]
            else:
                features['amount_quadratic_coef'] = 0
            
            # 交易笔数的二次趋势
            if np.std(y_count) > 0:
                lr_poly_count = LinearRegression()
                lr_poly_count.fit(X_poly, y_count)
                features['count_quadratic_coef'] = lr_poly_count.coef_[1]
            else:
                features['count_quadratic_coef'] = 0
        else:
            features['amount_quadratic_coef'] = 0
            features['count_quadratic_coef'] = 0
        
        return features
    
    def identify_trend(self, features):
        """
        识别商户交易趋势
        
        Args:
            features: 特征字典
            
        Returns:
            趋势类型: 'increasing', 'decreasing', 'stable'
        """
        amount_slope = features['amount_trend_slope']
        count_slope = features['count_trend_slope']
        
        # 归一化斜率
        amount_normalized_slope = amount_slope / (features['avg_amount'] + 1e-6)
        count_normalized_slope = count_slope / (features['avg_transaction_count'] + 1e-6)
        
        # 综合判断
        avg_normalized_slope = (amount_normalized_slope + count_normalized_slope) / 2
        
        # 阈值可以根据实际情况调整
        if avg_normalized_slope > 0.05:
            return 'increasing'
        elif avg_normalized_slope < -0.05:
            return 'decreasing'
        else:
            return 'stable'
    
    def prepare_training_data(self, shop_dict, churn_labels):
        """
        准备训练数据
        
        Args:
            shop_dict: 商户交易数据字典 {shop_id: DataFrame}
            churn_labels: 商户流失标签字典 {shop_id: 0/1}
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        feature_list = []
        label_list = []
        shop_ids = []
        
        for shop_id, transaction_data in shop_dict.items():
            if shop_id not in churn_labels:
                continue
            
            try:
                # 预处理数据
                daily_data = self.preprocess_data(transaction_data)
                
                # 分类频率
                frequency_type = self.classify_merchant_frequency(daily_data)
                
                # 应用移动平均
                smoothed_data = self.apply_moving_average(daily_data, frequency_type)
                
                # 特征工程
                features = self.engineer_features(smoothed_data)
                
                feature_list.append(features)
                label_list.append(churn_labels[shop_id])
                shop_ids.append(shop_id)
            except Exception as e:
                print(f"处理商户 {shop_id} 时出错: {e}")
                continue
        
        # 转换为DataFrame
        X = pd.DataFrame(feature_list)
        y = np.array(label_list)
        
        feature_names = X.columns.tolist()
        
        # 处理缺失值和无穷值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, feature_names
    
    def train(self, X_train, y_train):
        """
        训练逻辑回归模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 训练逻辑回归模型
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # 处理类别不平衡
        )
        self.model.fit(X_train_scaled, y_train)
        
        print("模型训练完成！")
    
    def evaluate(self, X_test, y_test):
        """
        评估模型
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
        """
        X_test_scaled = self.scaler.transform(X_test)
        
        # 预测
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # 评估指标
        print("\n模型评估结果：")
        print("=" * 60)
        print("\n分类报告：")
        print(classification_report(y_test, y_pred, target_names=['未流失', '流失']))
        
        print(f"\nAUC-ROC分数: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        print("\n混淆矩阵：")
        cm = confusion_matrix(y_test, y_pred)
        print(f"真负例: {cm[0,0]}, 假正例: {cm[0,1]}")
        print(f"假负例: {cm[1,0]}, 真正例: {cm[1,1]}")
        
        return y_pred, y_pred_proba
    
    def predict_churn(self, transaction_data):
        """
        预测单个商户的流失概率
        
        Args:
            transaction_data: 商户交易数据DataFrame
            
        Returns:
            流失概率, 趋势类型, 频率类型, 特征字典
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")
        
        # 预处理数据
        daily_data = self.preprocess_data(transaction_data)
        
        # 分类频率
        frequency_type = self.classify_merchant_frequency(daily_data)
        
        # 应用移动平均
        smoothed_data = self.apply_moving_average(daily_data, frequency_type)
        
        # 特征工程
        features = self.engineer_features(smoothed_data)
        
        # 识别趋势
        trend = self.identify_trend(features)
        
        # 预测
        X = pd.DataFrame([features])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.transform(X)
        
        churn_probability = self.model.predict_proba(X_scaled)[0, 1]
        
        return churn_probability, trend, frequency_type, features
    
    def get_feature_importance(self, feature_names):
        """
        获取特征重要性
        
        Args:
            feature_names: 特征名称列表
            
        Returns:
            特征重要性DataFrame
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        # 逻辑回归的系数可以作为特征重要性
        coefficients = self.model.coef_[0]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        return feature_importance
    
    def batch_predict(self, shop_dict, top_n=None):
        """
        批量预测商户流失概率
        
        Args:
            shop_dict: 商户交易数据字典 {shop_id: DataFrame}
            top_n: 返回流失概率最高的前N个商户
            
        Returns:
            预测结果DataFrame
        """
        results = []
        
        for shop_id, transaction_data in shop_dict.items():
            try:
                churn_prob, trend, freq_type, features = self.predict_churn(transaction_data)
                
                results.append({
                    'shop_id': shop_id,
                    'churn_probability': churn_prob,
                    'trend': trend,
                    'frequency_type': freq_type,
                    'avg_amount': features['avg_amount'],
                    'avg_transaction_count': features['avg_transaction_count'],
                    'days_since_last_transaction': features['days_since_last_transaction'],
                    'active_days_ratio': features['active_days_ratio']
                })
            except Exception as e:
                print(f"预测商户 {shop_id} 时出错: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('churn_probability', ascending=False)
        
        if top_n:
            results_df = results_df.head(top_n)
        
        return results_df
