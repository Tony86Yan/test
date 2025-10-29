"""
商户流失预测完整示例
整合传统统计方法和机器学习模型
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from merchant_churn_prediction import MerchantChurnPredictor, print_model_evaluation

min_pt = ""
max_pt = ""


def generate_merchant_data_with_trend(active_days_recent, active_days_early, 
                                      trend_type='stable', base_amount=10000):
    """
    生成模拟商户数据
    
    参数：
    active_days_recent: 最近30天的活跃天数
    active_days_early: 前30天的活跃天数
    trend_type: 'up', 'down', 'stable'
    base_amount: 基础交易金额
    """
    dates = pd.date_range(start='2024-09-01', periods=60, freq='D')
    
    transaction_counts = np.zeros(60)
    amounts = np.zeros(60)
    
    # 前30天
    early_active_indices = np.random.choice(
        30, size=min(active_days_early, 30), replace=False
    )
    
    # 后30天
    recent_active_indices = np.random.choice(
        range(30, 60), size=min(active_days_recent, 30), replace=False
    )
    
    base_count = 100
    
    # 前30天的交易
    for idx in early_active_indices:
        transaction_counts[idx] = base_count + np.random.normal(0, 15)
        amounts[idx] = base_amount + np.random.normal(0, 1000)
    
    # 后30天的交易（体现趋势）
    for i, idx in enumerate(sorted(recent_active_indices)):
        if trend_type == 'up':
            factor = 1 + (i / len(recent_active_indices)) * 0.6
        elif trend_type == 'down':
            factor = 1 - (i / len(recent_active_indices)) * 0.5
        else:  # stable
            factor = 1 + np.random.uniform(-0.1, 0.1)
        
        transaction_counts[idx] = base_count * factor + np.random.normal(0, 15)
        amounts[idx] = base_amount * factor + np.random.normal(0, 1000)
    
    transaction_counts = np.maximum(transaction_counts, 0)
    amounts = np.maximum(amounts, 0)
    
    return pd.DataFrame({
        'date': dates,
        'transaction_count': transaction_counts,
        'amount': amounts
    })


def main():
    """主函数"""
    
    print("="*80)
    print("商户流失预测系统 - 完整示例")
    print("="*80)
    
    # ========== 第一步：生成模拟数据 ==========
    print("\n【第一步】生成模拟商户数据...")
    
    # 模拟不同类型的商户
    merchants_data = {
        # 高风险商户（流失）
        'M001_高频下降': generate_merchant_data_with_trend(10, 28, 'down', 50000),
        'M002_中频下降': generate_merchant_data_with_trend(8, 18, 'down', 30000),
        'M003_长期不活跃': generate_merchant_data_with_trend(2, 15, 'down', 20000),
        
        # 中等风险商户
        'M004_活跃度下降': generate_merchant_data_with_trend(15, 25, 'down', 40000),
        'M005_交易波动': generate_merchant_data_with_trend(12, 12, 'stable', 25000),
        
        # 低风险商户（正常）
        'M006_高频稳定': generate_merchant_data_with_trend(28, 27, 'stable', 60000),
        'M007_高频上升': generate_merchant_data_with_trend(30, 25, 'up', 55000),
        'M008_中频稳定': generate_merchant_data_with_trend(18, 17, 'stable', 35000),
        'M009_低频稳定': generate_merchant_data_with_trend(8, 7, 'stable', 15000),
        'M010_增长商户': generate_merchant_data_with_trend(22, 15, 'up', 45000),
    }
    
    # 设置全局日期范围
    global min_pt, max_pt
    min_pt = '2024-09-01'
    max_pt = '2024-10-30'
    
    print(f"生成了 {len(merchants_data)} 个模拟商户")
    
    # ========== 第二步：创建预测器并提取特征 ==========
    print("\n【第二步】创建预测器并提取特征...")
    
    predictor = MerchantChurnPredictor(observation_days=60, decay_rate=0.95)
    
    # 准备训练数据（自动标注）
    X, y, merchant_ids, feature_names = predictor.prepare_training_data(merchants_data)
    
    print(f"提取特征数量: {len(feature_names)}")
    print(f"样本数量: {len(X)}")
    print(f"流失商户数: {y.sum()}, 正常商户数: {(y==0).sum()}")
    
    # ========== 第三步：训练模型 ==========
    print("\n【第三步】训练机器学习模型...")
    
    results = predictor.train_models(X, y, test_size=0.3, random_state=42)
    
    # 打印评估结果
    print_model_evaluation(results)
    
    # ========== 第四步：批量预测 ==========
    print("\n" + "="*80)
    print("【第四步】批量预测商户流失风险")
    print("="*80)
    
    predictions_df = predictor.batch_predict(merchants_data, model_type='ensemble')
    
    print("\n商户流失风险排名（按流失概率降序）:")
    print("-"*80)
    
    # 格式化输出
    display_df = predictions_df.copy()
    display_df.columns = [
        '商户ID', '流失概率', '风险等级', '60天交易额', 
        '活跃率', '金额趋势', '连续不活跃天数', '商户等级'
    ]
    
    # 格式化数值显示
    display_df['流失概率'] = display_df['流失概率'].apply(lambda x: f'{x:.1%}')
    display_df['60天交易额'] = display_df['60天交易额'].apply(lambda x: f'{x:,.0f}')
    display_df['活跃率'] = display_df['活跃率'].apply(lambda x: f'{x:.1%}')
    display_df['金额趋势'] = display_df['金额趋势'].apply(lambda x: f'{x:+.1%}')
    
    print(display_df.to_string(index=False))
    
    # ========== 第五步：详细分析高风险商户 ==========
    print("\n" + "="*80)
    print("【第五步】高风险商户详细分析")
    print("="*80)
    
    high_risk_merchants = predictions_df[
        predictions_df['risk_level'] == '高风险'
    ]['merchant_id'].tolist()
    
    if high_risk_merchants:
        print(f"\n发现 {len(high_risk_merchants)} 个高风险商户:")
        
        for merchant_id in high_risk_merchants:
            print(f"\n--- {merchant_id} ---")
            
            # 获取详细预测结果
            prediction = predictor.predict_churn(
                merchants_data[merchant_id], 
                model_type='ensemble'
            )
            
            features = prediction['features']
            
            print(f"流失概率: {prediction['churn_probability']:.1%}")
            print(f"风险等级: {prediction['risk_level']}")
            print(f"\n关键指标:")
            print(f"  - 60天交易总额: {features['total_amount_60d']:,.0f}")
            print(f"  - 活跃率: {features['active_rate']:.1%}")
            print(f"  - 加权活跃率: {features['weighted_active_rate']:.1%}")
            print(f"  - 金额趋势: {features['amount_trend']:+.1%}")
            print(f"  - 交易笔数趋势: {features['count_trend']:+.1%}")
            print(f"  - 连续不活跃天数: {features['consecutive_inactive_days']:.0f}")
            print(f"  - 近期vs早期交易额变化: {features['amount_change_rate']:+.1%}")
            
            # 商户分类信息
            classification = predictor.classify_merchant(merchants_data[merchant_id])
            print(f"\n商户分类: {classification['category_name']}")
            print(f"  - 使用的短期窗口: {classification['category_info']['short_window']}天")
            print(f"  - 使用的长期窗口: {classification['category_info']['long_window']}天")
    else:
        print("\n未发现高风险商户")
    
    # ========== 第六步：特征重要性分析 ==========
    print("\n" + "="*80)
    print("【第六步】特征重要性综合分析")
    print("="*80)
    
    # 汇总所有模型的特征重要性
    all_importance = {}
    
    for model_name, metrics in results.items():
        for feat, imp in metrics['feature_importance'].items():
            if feat not in all_importance:
                all_importance[feat] = []
            all_importance[feat].append(imp)
    
    # 计算平均重要性
    avg_importance = {
        feat: np.mean(imps) 
        for feat, imps in all_importance.items()
    }
    
    # 排序并显示Top 15
    sorted_importance = sorted(
        avg_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:15]
    
    print("\nTop 15 最重要特征（综合所有模型）:")
    print("-"*60)
    for i, (feat, imp) in enumerate(sorted_importance, 1):
        print(f"{i:2d}. {feat:35s} {imp:.4f}")
    
    # ========== 第七步：业务建议 ==========
    print("\n" + "="*80)
    print("【第七步】业务建议")
    print("="*80)
    
    # 统计各风险等级商户数量
    risk_distribution = predictions_df['risk_level'].value_counts().to_dict()
    
    print(f"\n风险分布:")
    for level, count in risk_distribution.items():
        print(f"  {level}: {count} 个商户")
    
    # 计算潜在损失
    high_risk_df = predictions_df[predictions_df['risk_level'] == '高风险']
    if len(high_risk_df) > 0:
        potential_loss = high_risk_df['total_amount_60d'].sum()
        print(f"\n潜在流失交易额（高风险商户60天总交易额）: {potential_loss:,.0f}")
    
    print("\n建议采取的行动:")
    print("1. 【高风险商户】")
    print("   - 立即联系，了解业务变化原因")
    print("   - 提供优惠政策或定制化服务")
    print("   - 安排客户经理定期回访")
    
    print("\n2. 【中风险商户】")
    print("   - 关注交易趋势变化")
    print("   - 定期发送产品更新和优惠信息")
    print("   - 收集反馈，改进服务")
    
    print("\n3. 【低风险商户】")
    print("   - 保持现有服务质量")
    print("   - 探索交叉销售机会")
    print("   - 鼓励推荐新商户")
    
    # ========== 第八步：模型对比 ==========
    print("\n" + "="*80)
    print("【第八步】不同模型预测对比")
    print("="*80)
    
    # 选择一个商户，对比不同模型的预测结果
    sample_merchant = 'M001_高频下降'
    print(f"\n以商户 {sample_merchant} 为例，对比不同模型:")
    print("-"*60)
    
    for model_type in ['lr', 'rf', 'xgb', 'ensemble']:
        pred = predictor.predict_churn(
            merchants_data[sample_merchant], 
            model_type=model_type
        )
        
        if pred['churn_probability'] is not None:
            print(f"{model_type.upper():12s}: 流失概率 {pred['churn_probability']:.1%}, "
                  f"风险等级 {pred['risk_level']}")
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == "__main__":
    # 设置随机种子以保证可复现性
    np.random.seed(42)
    
    # 运行主函数
    main()
