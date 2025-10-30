"""
商户流失预测示例代码
演示如何使用MerchantChurnPredictor进行商户流失预测
"""

import pandas as pd
import numpy as np
from merchant_churn_predictor import MerchantChurnPredictor


def generate_sample_data():
    """
    生成示例数据用于演示
    在实际使用中，请从CSV文件读取真实数据
    """
    print("生成示例商户交易数据...")
    
    np.random.seed(42)
    shop_dict = {}
    churn_labels = {}
    
    start_date = pd.to_datetime('2025-07-03')
    end_date = pd.to_datetime('2025-10-30')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 生成不同类型的商户数据
    
    # 1. 高频活跃商户（未流失） - 每天都有稳定交易
    for i in range(1, 11):
        shop_id = f'high_active_{i}'
        dates = []
        amounts = []
        counts = []
        
        for date in date_range:
            dates.append(date.strftime('%Y-%m-%d'))
            # 稳定的交易金额和笔数，带一些随机波动
            amounts.append(np.random.normal(50000, 5000))
            counts.append(np.random.randint(80, 120))
        
        shop_dict[shop_id] = pd.DataFrame({
            'date': dates,
            'amount': amounts,
            'transaction_count': counts
        })
        churn_labels[shop_id] = 0  # 未流失
    
    # 2. 高频流失商户 - 交易量逐渐下降
    for i in range(1, 11):
        shop_id = f'high_churn_{i}'
        dates = []
        amounts = []
        counts = []
        
        for idx, date in enumerate(date_range):
            dates.append(date.strftime('%Y-%m-%d'))
            # 交易量逐渐下降
            decay_factor = max(0, 1 - idx / len(date_range))
            amounts.append(np.random.normal(50000 * decay_factor, 5000))
            counts.append(max(0, int(np.random.normal(100 * decay_factor, 10))))
        
        shop_dict[shop_id] = pd.DataFrame({
            'date': dates,
            'amount': amounts,
            'transaction_count': counts
        })
        churn_labels[shop_id] = 1  # 流失
    
    # 3. 中频活跃商户（未流失） - 每周3-4天有交易
    for i in range(1, 11):
        shop_id = f'medium_active_{i}'
        dates = []
        amounts = []
        counts = []
        
        for date in date_range:
            # 70%的概率有交易
            if np.random.random() < 0.7:
                dates.append(date.strftime('%Y-%m-%d'))
                amounts.append(np.random.normal(30000, 3000))
                counts.append(np.random.randint(40, 80))
        
        shop_dict[shop_id] = pd.DataFrame({
            'date': dates,
            'amount': amounts,
            'transaction_count': counts
        })
        churn_labels[shop_id] = 0  # 未流失
    
    # 4. 中频流失商户 - 交易频率逐渐降低
    for i in range(1, 11):
        shop_id = f'medium_churn_{i}'
        dates = []
        amounts = []
        counts = []
        
        for idx, date in enumerate(date_range):
            # 交易概率逐渐降低
            prob = max(0.1, 0.7 - 0.6 * idx / len(date_range))
            if np.random.random() < prob:
                dates.append(date.strftime('%Y-%m-%d'))
                amounts.append(np.random.normal(30000, 3000))
                counts.append(np.random.randint(40, 80))
        
        shop_dict[shop_id] = pd.DataFrame({
            'date': dates,
            'amount': amounts,
            'transaction_count': counts
        })
        churn_labels[shop_id] = 1  # 流失
    
    # 5. 低频活跃商户（未流失） - 每月固定时间交易
    for i in range(1, 6):
        shop_id = f'low_active_{i}'
        dates = []
        amounts = []
        counts = []
        
        for date in date_range:
            # 每月的5号左右有交易
            if date.day in [5, 6, 7]:
                dates.append(date.strftime('%Y-%m-%d'))
                amounts.append(np.random.normal(100000, 10000))
                counts.append(np.random.randint(200, 300))
        
        shop_dict[shop_id] = pd.DataFrame({
            'date': dates,
            'amount': amounts,
            'transaction_count': counts
        })
        churn_labels[shop_id] = 0  # 未流失
    
    # 6. 低频流失商户 - 已经停止交易
    for i in range(1, 6):
        shop_id = f'low_churn_{i}'
        dates = []
        amounts = []
        counts = []
        
        # 只在前半段时间有交易
        for date in date_range[:len(date_range)//2]:
            if date.day in [5, 6, 7]:
                dates.append(date.strftime('%Y-%m-%d'))
                amounts.append(np.random.normal(100000, 10000))
                counts.append(np.random.randint(200, 300))
        
        shop_dict[shop_id] = pd.DataFrame({
            'date': dates,
            'amount': amounts,
            'transaction_count': counts
        })
        churn_labels[shop_id] = 1  # 流失
    
    print(f"生成完成！共 {len(shop_dict)} 个商户")
    print(f"流失商户: {sum(churn_labels.values())}")
    print(f"活跃商户: {len(churn_labels) - sum(churn_labels.values())}")
    
    return shop_dict, churn_labels, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def load_real_data(csv_path):
    """
    从CSV文件加载真实数据
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        shop_dict, min_date, max_date
    """
    print(f"从 {csv_path} 加载数据...")
    
    # 读取数据
    mer_df = pd.read_csv(csv_path, sep="@@@")
    
    # 处理数据
    shop_df = mer_df.drop(columns=['merchant_id', 'merchant_name', '_shop_name', 
                                    'check_date', 'shop_create_date'])
    shop_df = shop_df.rename(columns={
        "pt": "date", 
        "amount": "amount", 
        "number": "transaction_count"
    })
    
    # 过滤日期
    shop_df = shop_df[shop_df['date'] >= '2025-07-03']
    min_pt = shop_df['date'].min()
    max_pt = shop_df['date'].max()
    
    print(f"数据日期范围: {min_pt} 至 {max_pt}")
    
    # 构建商户字典
    shop_dict = {}
    for row in shop_df.values:
        key = row[0]  # _shop_id
        if key not in shop_dict.keys():
            shop_dict[key] = []
        shop_dict[key].append([row[1], row[3], row[2]])  # [date, transaction_count, amount]
    
    shop_dict = {
        k: pd.DataFrame(v, columns=['date', 'transaction_count', 'amount']) 
        for k, v in shop_dict.items()
    }
    
    print(f"加载完成！共 {len(shop_dict)} 个商户")
    
    return shop_dict, min_pt, max_pt


def main():
    """主函数"""
    
    print("=" * 80)
    print("商户流失预测系统 - 示例演示")
    print("=" * 80)
    
    # 方式1: 使用示例数据
    print("\n方式1: 使用生成的示例数据")
    print("-" * 80)
    shop_dict, churn_labels, min_date, max_date = generate_sample_data()
    
    # 方式2: 从CSV文件加载真实数据（取消注释以使用）
    # print("\n方式2: 从CSV文件加载真实数据")
    # print("-" * 80)
    # csv_path = "mer_trans_data.csv"
    # shop_dict, min_date, max_date = load_real_data(csv_path)
    # 
    # # 真实数据需要手动标注流失标签
    # # 这里演示如何定义流失：最近30天无交易
    # churn_labels = {}
    # for shop_id, df in shop_dict.items():
    #     df_sorted = df.sort_values('date')
    #     if len(df_sorted) > 0:
    #         last_date = pd.to_datetime(df_sorted['date'].iloc[-1])
    #         days_since_last = (pd.to_datetime(max_date) - last_date).days
    #         churn_labels[shop_id] = 1 if days_since_last > 30 else 0
    
    # 初始化预测器
    print(f"\n初始化预测器（日期范围: {min_date} 至 {max_date}）...")
    predictor = MerchantChurnPredictor(min_date, max_date)
    
    # 准备训练数据
    print("\n准备训练数据...")
    X_train, X_test, y_train, y_test, feature_names = predictor.prepare_training_data(
        shop_dict, churn_labels
    )
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    print(f"特征数量: {len(feature_names)}")
    
    # 训练模型
    print("\n训练逻辑回归模型...")
    predictor.train(X_train, y_train)
    
    # 评估模型
    print("\n评估模型性能...")
    y_pred, y_pred_proba = predictor.evaluate(X_test, y_test)
    
    # 查看特征重要性
    print("\n特征重要性（前10个）:")
    print("=" * 60)
    feature_importance = predictor.get_feature_importance(feature_names)
    print(feature_importance.head(10).to_string(index=False))
    
    # 批量预测所有商户
    print("\n\n批量预测所有商户的流失概率...")
    print("=" * 60)
    results = predictor.batch_predict(shop_dict, top_n=20)
    print("\n流失风险最高的前20个商户:")
    print(results.to_string(index=False))
    
    # 单个商户详细预测示例
    print("\n\n单个商户详细预测示例:")
    print("=" * 60)
    sample_shop_id = list(shop_dict.keys())[0]
    churn_prob, trend, freq_type, features = predictor.predict_churn(
        shop_dict[sample_shop_id]
    )
    
    print(f"商户ID: {sample_shop_id}")
    print(f"流失概率: {churn_prob:.2%}")
    print(f"交易趋势: {trend}")
    print(f"频率类型: {freq_type}")
    print(f"\n关键特征:")
    print(f"  - 平均交易金额: {features['avg_amount']:.2f}")
    print(f"  - 平均交易笔数: {features['avg_transaction_count']:.2f}")
    print(f"  - 距上次交易天数: {features['days_since_last_transaction']:.0f}")
    print(f"  - 活跃天数占比: {features['active_days_ratio']:.2%}")
    print(f"  - 金额趋势斜率: {features['amount_trend_slope']:.2f}")
    print(f"  - 笔数趋势斜率: {features['count_trend_slope']:.2f}")
    
    # 风险分级建议
    print("\n\n商户风险分级建议:")
    print("=" * 60)
    results['risk_level'] = pd.cut(
        results['churn_probability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['低风险', '中风险', '高风险']
    )
    
    risk_summary = results.groupby('risk_level').agg({
        'shop_id': 'count',
        'churn_probability': 'mean'
    }).rename(columns={'shop_id': '商户数量', 'churn_probability': '平均流失概率'})
    
    print(risk_summary)
    
    print("\n高风险商户（流失概率 > 60%）需要重点关注和干预！")
    high_risk = results[results['churn_probability'] > 0.6]
    print(f"共有 {len(high_risk)} 个高风险商户")
    
    print("\n" + "=" * 80)
    print("预测完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
