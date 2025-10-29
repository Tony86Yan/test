import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

min_pt = ""
max_pt = ""


class MerchantChurnPredictor:
    """
    商户流失预测系统
    结合传统统计方法和机器学习模型
    """
    
    def __init__(self, observation_days=60, decay_rate=0.95):
        """
        参数：
        observation_days: 观察期天数（默认60天）
        decay_rate: 时间衰减率（默认0.95）
        """
        self.observation_days = observation_days
        self.decay_rate = decay_rate
        self.time_weights = self._calculate_time_weights()
        
        # 机器学习模型
        self.lr_model = None
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        
        # 商户分类标准（基于加权活跃率）
        self.merchant_categories = {
            'very_active': {
                'min_weighted_rate': 0.60,
                'max_weighted_rate': 1.00,
                'name': '高频商户',
                'short_window': 7,   # 高频商户使用更短的窗口
                'long_window': 15,
                'stable_threshold': 0.2,
            },
            'active': {
                'min_weighted_rate': 0.40,
                'max_weighted_rate': 0.60,
                'name': '中高频商户',
                'short_window': 7,
                'long_window': 30,
                'stable_threshold': 0.5,
            },
            'moderate': {
                'min_weighted_rate': 0.25,
                'max_weighted_rate': 0.40,
                'name': '中频商户',
                'short_window': 15,
                'long_window': 30,
                'stable_threshold': 0.9,
            },
            'low_active': {
                'min_weighted_rate': 0.10,
                'max_weighted_rate': 0.25,
                'name': '低频商户',
                'short_window': 15,
                'long_window': 30,
                'stable_threshold': 1.4,
            },
            'very_low': {
                'min_weighted_rate': 0.00,
                'max_weighted_rate': 0.10,
                'name': '极低频商户',
                'short_window': 15,
                'long_window': 30,
                'stable_threshold': 2,
            }
        }
    
    def _calculate_time_weights(self):
        """预计算时间衰减权重"""
        weights = np.array([
            self.decay_rate ** i
            for i in range(self.observation_days)
        ])
        return weights[::-1]
    
    def _preprocess_data(self, transaction_data):
        """数据预处理"""
        df = transaction_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # 创建完整日期序列
        date_range = pd.date_range(
            start=min_pt,
            end=max_pt,
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
    
    def calculate_weighted_active_rate(self, df):
        """计算加权活跃率"""
        if len(df) == 0:
            return 0.0
        
        df_recent = df.tail(self.observation_days).copy()
        n_days = len(df_recent)
        
        if n_days == 0:
            return 0.0
        
        weights = self.time_weights[-n_days:]
        is_active = (df_recent['transaction_count'] > 0).astype(float).values
        
        weighted_sum = np.sum(is_active * weights)
        weights_sum = np.sum(weights)
        
        return weighted_sum / weights_sum if weights_sum > 0 else 0
    
    def classify_merchant(self, transaction_data):
        """商户分类"""
        df = self._preprocess_data(transaction_data)
        df_recent = df.tail(self.observation_days)
        
        if len(df_recent) < 3:
            return {
                'category': 'insufficient_data',
                'category_name': '数据不足',
                'weighted_rate': 0.0,
                'active_days': 0,
                'category_info': None
            }
        
        weighted_rate = self.calculate_weighted_active_rate(df_recent)
        active_days = (df_recent['transaction_count'] > 0).sum()
        
        category = 'very_low'
        for cat_name, cat_info in self.merchant_categories.items():
            if (cat_info['min_weighted_rate'] < weighted_rate <= 
                cat_info['max_weighted_rate']):
                category = cat_name
                break
        
        category_info = self.merchant_categories[category]
        
        return {
            'category': category,
            'category_name': category_info['name'],
            'weighted_rate': weighted_rate,
            'active_days': active_days,
            'category_info': category_info
        }
    
    def extract_features(self, transaction_data):
        """
        特征工程：从交易数据中提取关键特征
        
        返回特征字典
        """
        df = self._preprocess_data(transaction_data)
        df_recent = df.tail(self.observation_days)
        
        if len(df_recent) < 3:
            return None
        
        # 商户分类
        classification = self.classify_merchant(transaction_data)
        category_info = classification['category_info']
        
        if category_info is None:
            return None
        
        # 根据商户频率自适应调整窗口大小
        short_window = category_info['short_window']
        long_window = category_info['long_window']
        
        # 基础统计特征
        features = {}
        
        # 1. 活跃度特征
        features['weighted_active_rate'] = classification['weighted_rate']
        features['active_days'] = classification['active_days']
        features['active_rate'] = classification['active_days'] / len(df_recent)
        
        # 2. 交易量特征（最近60天）
        features['total_amount_60d'] = df_recent['amount'].sum()
        features['total_count_60d'] = df_recent['transaction_count'].sum()
        features['avg_amount_per_day'] = df_recent['amount'].mean()
        features['avg_count_per_day'] = df_recent['transaction_count'].mean()
        
        # 3. 交易量特征（只考虑活跃日）
        active_days_df = df_recent[df_recent['transaction_count'] > 0]
        if len(active_days_df) > 0:
            features['avg_amount_per_active_day'] = active_days_df['amount'].mean()
            features['avg_count_per_active_day'] = active_days_df['transaction_count'].mean()
            features['avg_amount_per_txn'] = (
                features['total_amount_60d'] / features['total_count_60d'] 
                if features['total_count_60d'] > 0 else 0
            )
        else:
            features['avg_amount_per_active_day'] = 0
            features['avg_count_per_active_day'] = 0
            features['avg_amount_per_txn'] = 0
        
        # 4. 自适应移动平均特征（根据商户频率调整窗口）
        # 短期MA（使用category特定的窗口）
        features['ma_amount_short'] = self._adaptive_weighted_ma(
            df_recent['amount'].values, short_window
        )
        features['ma_count_short'] = self._adaptive_weighted_ma(
            df_recent['transaction_count'].values, short_window
        )
        
        # 长期MA（使用category特定的窗口）
        features['ma_amount_long'] = self._adaptive_weighted_ma(
            df_recent['amount'].values, long_window
        )
        features['ma_count_long'] = self._adaptive_weighted_ma(
            df_recent['transaction_count'].values, long_window
        )
        
        # 5. 趋势特征（短期vs长期）
        if features['ma_amount_long'] > 0:
            features['amount_trend'] = (
                (features['ma_amount_short'] - features['ma_amount_long']) / 
                features['ma_amount_long']
            )
        else:
            features['amount_trend'] = 0
        
        if features['ma_count_long'] > 0:
            features['count_trend'] = (
                (features['ma_count_short'] - features['ma_count_long']) / 
                features['ma_count_long']
            )
        else:
            features['count_trend'] = 0
        
        # 6. 波动性特征
        features['amount_std'] = df_recent['amount'].std()
        features['count_std'] = df_recent['transaction_count'].std()
        features['amount_cv'] = (
            features['amount_std'] / features['avg_amount_per_day'] 
            if features['avg_amount_per_day'] > 0 else 0
        )
        
        # 7. 时间段对比特征
        # 最近30天 vs 之前30天
        recent_30 = df_recent.tail(30)
        early_30 = df_recent.head(30)
        
        recent_amount = recent_30['amount'].sum()
        early_amount = early_30['amount'].sum()
        
        if early_amount > 0:
            features['amount_change_rate'] = (recent_amount - early_amount) / early_amount
        else:
            features['amount_change_rate'] = 0
        
        recent_active = (recent_30['transaction_count'] > 0).sum()
        early_active = (early_30['transaction_count'] > 0).sum()
        
        if early_active > 0:
            features['active_days_change_rate'] = (recent_active - early_active) / early_active
        else:
            features['active_days_change_rate'] = 0
        
        # 8. 最近活跃度下降特征
        # 最近7天、15天、30天的活跃率
        for days in [7, 15, 30]:
            last_n = df_recent.tail(days)
            active_n = (last_n['transaction_count'] > 0).sum()
            features[f'active_rate_last_{days}d'] = active_n / days
        
        # 9. 连续无交易天数
        # 从最后一天开始往前数，连续多少天没有交易
        consecutive_inactive = 0
        for i in range(len(df_recent) - 1, -1, -1):
            if df_recent.iloc[i]['transaction_count'] == 0:
                consecutive_inactive += 1
            else:
                break
        features['consecutive_inactive_days'] = consecutive_inactive
        
        # 10. 商户分类编码
        category_encoding = {
            'very_active': 5,
            'active': 4,
            'moderate': 3,
            'low_active': 2,
            'very_low': 1,
            'insufficient_data': 0
        }
        features['merchant_category_code'] = category_encoding[classification['category']]
        
        return features
    
    def _adaptive_weighted_ma(self, data, window):
        """
        自适应加权移动平均
        根据商户频率使用不同的窗口长度
        """
        if len(data) < window:
            window = len(data)
        
        if window <= 0:
            return 0
        
        recent_data = data[-window:]
        
        # 去除极值（可选）
        if len(recent_data) > 2:
            mean_val = recent_data.mean()
            max_val = recent_data.max()
            min_val = recent_data.min()
            arr_new = recent_data.copy()
            
            max_idx = np.where(arr_new == max_val)[0][0]
            arr_new[max_idx] = mean_val
            
            min_idx = np.where(arr_new == min_val)[0][0]
            arr_new[min_idx] = mean_val
            recent_data = arr_new
        
        # 加权平均
        recent_weights = self.time_weights[-window:]
        weighted_sum = np.sum(recent_data * recent_weights)
        weights_sum = np.sum(recent_weights)
        
        return weighted_sum / weights_sum if weights_sum > 0 else 0
    
    def prepare_training_data(self, merchants_data, churn_labels=None):
        """
        准备训练数据
        
        参数：
        merchants_data: dict, {merchant_id: DataFrame}
        churn_labels: dict, {merchant_id: 0/1}, 0=未流失, 1=流失
                      如果为None，则自动标注（基于趋势）
        
        返回：
        X: 特征矩阵
        y: 标签向量
        merchant_ids: 商户ID列表
        feature_names: 特征名称列表
        """
        features_list = []
        labels_list = []
        merchant_ids_list = []
        
        for merchant_id, merchant_data in merchants_data.items():
            features = self.extract_features(merchant_data)
            
            if features is None:
                continue
            
            # 自动标注：基于趋势判断是否流失
            if churn_labels is None:
                # 综合判断流失条件
                is_churn = (
                    (features['amount_trend'] < -0.3) or  # 交易额下降>30%
                    (features['consecutive_inactive_days'] > 14) or  # 连续14天无交易
                    (features['active_rate_last_30d'] < 0.2 and features['merchant_category_code'] >= 3) or  # 高频商户活跃率骤降
                    (features['amount_change_rate'] < -0.5)  # 近期vs早期交易额下降>50%
                )
                label = 1 if is_churn else 0
            else:
                label = churn_labels.get(merchant_id, 0)
            
            features_list.append(features)
            labels_list.append(label)
            merchant_ids_list.append(merchant_id)
        
        # 转换为DataFrame
        X = pd.DataFrame(features_list)
        y = np.array(labels_list)
        
        feature_names = X.columns.tolist()
        
        return X, y, merchant_ids_list, feature_names
    
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """
        训练多个机器学习模型
        
        返回：
        dict: 包含各模型的评估结果
        """
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # 1. Logistic Regression
        print("Training Logistic Regression...")
        self.lr_model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced'  # 处理类别不平衡
        )
        self.lr_model.fit(X_train_scaled, y_train)
        
        y_pred_lr = self.lr_model.predict(X_test_scaled)
        y_prob_lr = self.lr_model.predict_proba(X_test_scaled)[:, 1]
        
        results['Logistic Regression'] = {
            'predictions': y_pred_lr,
            'probabilities': y_prob_lr,
            'accuracy': (y_pred_lr == y_test).mean(),
            'roc_auc': roc_auc_score(y_test, y_prob_lr),
            'classification_report': classification_report(y_test, y_pred_lr),
            'confusion_matrix': confusion_matrix(y_test, y_pred_lr),
            'feature_importance': dict(zip(X.columns, np.abs(self.lr_model.coef_[0])))
        }
        
        # 2. Random Forest
        print("Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            class_weight='balanced'
        )
        self.rf_model.fit(X_train, y_train)
        
        y_pred_rf = self.rf_model.predict(X_test)
        y_prob_rf = self.rf_model.predict_proba(X_test)[:, 1]
        
        results['Random Forest'] = {
            'predictions': y_pred_rf,
            'probabilities': y_prob_rf,
            'accuracy': (y_pred_rf == y_test).mean(),
            'roc_auc': roc_auc_score(y_test, y_prob_rf),
            'classification_report': classification_report(y_test, y_pred_rf),
            'confusion_matrix': confusion_matrix(y_test, y_pred_rf),
            'feature_importance': dict(zip(X.columns, self.rf_model.feature_importances_))
        }
        
        # 3. XGBoost (如果可用)
        if XGBOOST_AVAILABLE:
            print("Training XGBoost...")
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss'
            )
            self.xgb_model.fit(X_train, y_train)
            
            y_pred_xgb = self.xgb_model.predict(X_test)
            y_prob_xgb = self.xgb_model.predict_proba(X_test)[:, 1]
            
            results['XGBoost'] = {
                'predictions': y_pred_xgb,
                'probabilities': y_prob_xgb,
                'accuracy': (y_pred_xgb == y_test).mean(),
                'roc_auc': roc_auc_score(y_test, y_prob_xgb),
                'classification_report': classification_report(y_test, y_pred_xgb),
                'confusion_matrix': confusion_matrix(y_test, y_pred_xgb),
                'feature_importance': dict(zip(X.columns, self.xgb_model.feature_importances_))
            }
        
        # 保存测试集用于后续分析
        self.X_test = X_test
        self.y_test = y_test
        
        return results
    
    def predict_churn(self, transaction_data, model_type='ensemble'):
        """
        预测单个商户的流失概率
        
        参数：
        transaction_data: DataFrame
        model_type: 'lr', 'rf', 'xgb', 'ensemble'
        
        返回：
        dict: 包含流失概率和风险等级
        """
        features = self.extract_features(transaction_data)
        
        if features is None:
            return {
                'churn_probability': None,
                'risk_level': 'unknown',
                'error': '数据不足'
            }
        
        # 转换为DataFrame并标准化
        X = pd.DataFrame([features])
        
        if model_type == 'lr' and self.lr_model is not None:
            X_scaled = self.scaler.transform(X)
            churn_prob = self.lr_model.predict_proba(X_scaled)[0, 1]
        
        elif model_type == 'rf' and self.rf_model is not None:
            churn_prob = self.rf_model.predict_proba(X)[0, 1]
        
        elif model_type == 'xgb' and self.xgb_model is not None:
            churn_prob = self.xgb_model.predict_proba(X)[0, 1]
        
        elif model_type == 'ensemble':
            # 集成预测：取多个模型的平均
            probs = []
            
            if self.lr_model is not None:
                X_scaled = self.scaler.transform(X)
                probs.append(self.lr_model.predict_proba(X_scaled)[0, 1])
            
            if self.rf_model is not None:
                probs.append(self.rf_model.predict_proba(X)[0, 1])
            
            if self.xgb_model is not None:
                probs.append(self.xgb_model.predict_proba(X)[0, 1])
            
            if len(probs) == 0:
                return {
                    'churn_probability': None,
                    'risk_level': 'unknown',
                    'error': '模型未训练'
                }
            
            churn_prob = np.mean(probs)
        
        else:
            return {
                'churn_probability': None,
                'risk_level': 'unknown',
                'error': f'模型{model_type}不可用'
            }
        
        # 风险等级划分
        if churn_prob >= 0.7:
            risk_level = '高风险'
        elif churn_prob >= 0.4:
            risk_level = '中风险'
        else:
            risk_level = '低风险'
        
        return {
            'churn_probability': churn_prob,
            'risk_level': risk_level,
            'features': features
        }
    
    def batch_predict(self, merchants_data, model_type='ensemble'):
        """批量预测商户流失"""
        results = []
        
        for merchant_id, merchant_data in merchants_data.items():
            prediction = self.predict_churn(merchant_data, model_type)
            
            if prediction['churn_probability'] is not None:
                result = {
                    'merchant_id': merchant_id,
                    'churn_probability': prediction['churn_probability'],
                    'risk_level': prediction['risk_level']
                }
                
                # 添加关键特征
                if 'features' in prediction:
                    features = prediction['features']
                    result.update({
                        'total_amount_60d': features.get('total_amount_60d', 0),
                        'active_rate': features.get('active_rate', 0),
                        'amount_trend': features.get('amount_trend', 0),
                        'consecutive_inactive_days': features.get('consecutive_inactive_days', 0),
                        'merchant_category_code': features.get('merchant_category_code', 0)
                    })
                
                results.append(result)
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('churn_probability', ascending=False)
        
        return df_results


def print_model_evaluation(results):
    """打印模型评估结果"""
    print("\n" + "="*80)
    print("模型评估结果")
    print("="*80)
    
    for model_name, metrics in results.items():
        print(f"\n【{model_name}】")
        print(f"准确率: {metrics['accuracy']:.3f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"\n混淆矩阵:")
        print(metrics['confusion_matrix'])
        print(f"\n分类报告:")
        print(metrics['classification_report'])
        
        # 特征重要性（Top 10）
        print(f"\nTop 10 重要特征:")
        importance = sorted(
            metrics['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for feat, imp in importance:
            print(f"  {feat}: {imp:.4f}")


# 示例使用
if __name__ == "__main__":
    print("="*80)
    print("商户流失预测系统 - 机器学习模型")
    print("="*80)
    
    # 这里使用您提供的数据加载代码
    # 请根据实际情况修改文件路径
    print("\n注意：请将数据文件路径修改为实际路径")
    print("示例：mer_df = pd.read_csv('your_data_path.csv', sep='@@@')")
