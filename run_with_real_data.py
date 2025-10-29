"""
使用实际数据的商户流失预测脚本
适配用户提供的数据格式
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from merchant_churn_prediction import MerchantChurnPredictor, print_model_evaluation

min_pt = ""
max_pt = ""


def load_and_prepare_data(file_path, sep="@@@", min_date='2025-07-03'):
    """
    加载并准备实际数据
    
    参数：
    file_path: CSV文件路径
    sep: 分隔符
    min_date: 最小日期过滤
    
    返回：
    shop_dict: {shop_id: DataFrame}
    """
    print(f"正在加载数据: {file_path}")
    
    # 读取数据
    # merchant_id@@@merchant_name@@@_shop_id@@@_shop_name@@@check_date@@@shop_create_date@@@pt@@@amount@@@number
    mer_df = pd.read_csv(file_path, sep=sep)
    
    print(f"原始数据行数: {len(mer_df)}")
    print(f"列名: {mer_df.columns.tolist()}")
    
    # 数据预处理
    shop_df = mer_df.drop(
        columns=['merchant_id', 'merchant_name', '_shop_name', 'check_date', 'shop_create_date'],
        errors='ignore'
    )
    
    # 重命名列
    shop_df = shop_df.rename(columns={
        "pt": "date",
        "amount": "amount",
        "number": "transaction_count"
    })
    
    # 日期过滤
    shop_df = shop_df[shop_df['date'] >= min_date]
    
    # 设置全局日期范围
    global min_pt, max_pt
    min_pt = shop_df['date'].min()
    max_pt = shop_df['date'].max()
    
    print(f"日期范围: {min_pt} 至 {max_pt}")
    print(f"过滤后数据行数: {len(shop_df)}")
    
    # 转换为字典格式
    shop_dict = {}
    for row in shop_df.values:
        key = row[0]  # _shop_id
        
        if key not in shop_dict:
            shop_dict[key] = []
        
        shop_dict[key].append([row[1], row[3], row[2]])  # [date, transaction_count, amount]
    
    # 转换为DataFrame
    shop_dict = {
        k: pd.DataFrame(v, columns=['date', 'transaction_count', 'amount']) 
        for k, v in shop_dict.items()
    }
    
    print(f"商户数量: {len(shop_dict)}")
    
    return shop_dict


def analyze_merchants(shop_dict, sample_size=None, train_model=True):
    """
    分析商户流失风险
    
    参数：
    shop_dict: 商户数据字典
    sample_size: 采样数量（用于大数据集）
    train_model: 是否训练模型
    """
    
    # 如果数据量太大，进行采样
    if sample_size and len(shop_dict) > sample_size:
        print(f"\n数据量较大，随机采样 {sample_size} 个商户进行分析...")
        merchant_ids = np.random.choice(
            list(shop_dict.keys()), 
            size=sample_size, 
            replace=False
        )
        shop_dict_sample = {k: shop_dict[k] for k in merchant_ids}
    else:
        shop_dict_sample = shop_dict
    
    # 创建预测器
    predictor = MerchantChurnPredictor(observation_days=60, decay_rate=0.95)
    
    # ========== 阶段1：特征提取和数据准备 ==========
    print("\n" + "="*80)
    print("阶段1：特征提取")
    print("="*80)
    
    X, y, merchant_ids, feature_names = predictor.prepare_training_data(shop_dict_sample)
    
    print(f"\n成功提取特征的商户数: {len(X)}")
    print(f"特征数量: {len(feature_names)}")
    print(f"自动标注结果 - 流失: {y.sum()}, 正常: {(y==0).sum()}")
    
    # 显示特征列表
    print("\n提取的特征列表:")
    for i, feat in enumerate(feature_names, 1):
        print(f"  {i:2d}. {feat}")
    
    # ========== 阶段2：训练模型（可选） ==========
    if train_model and len(X) >= 10:  # 至少需要10个样本
        print("\n" + "="*80)
        print("阶段2：训练机器学习模型")
        print("="*80)
        
        # 确保有足够的正负样本
        if y.sum() > 0 and (y==0).sum() > 0:
            test_size = min(0.3, 3 / len(X))  # 动态调整测试集大小
            results = predictor.train_models(X, y, test_size=test_size, random_state=42)
            
            print_model_evaluation(results)
        else:
            print("\n警告：所有样本标签相同，无法训练分类模型")
            train_model = False
    else:
        print("\n跳过模型训练（样本数量不足）")
        train_model = False
    
    # ========== 阶段3：批量预测 ==========
    print("\n" + "="*80)
    print("阶段3：批量预测商户流失风险")
    print("="*80)
    
    if train_model:
        predictions_df = predictor.batch_predict(shop_dict_sample, model_type='ensemble')
    else:
        # 如果没有训练模型，只提取特征
        print("\n未训练模型，仅显示特征分析...")
        features_list = []
        
        for merchant_id, merchant_data in shop_dict_sample.items():
            features = predictor.extract_features(merchant_data)
            if features:
                features['merchant_id'] = merchant_id
                features_list.append(features)
        
        predictions_df = pd.DataFrame(features_list)
        
        # 添加简单的风险评估（基于规则）
        def assess_risk(row):
            if (row['amount_trend'] < -0.3 or 
                row['consecutive_inactive_days'] > 14 or
                row['amount_change_rate'] < -0.5):
                return '高风险'
            elif (row['amount_trend'] < -0.1 or 
                  row['consecutive_inactive_days'] > 7):
                return '中风险'
            else:
                return '低风险'
        
        predictions_df['risk_level'] = predictions_df.apply(assess_risk, axis=1)
        predictions_df = predictions_df.sort_values('amount_trend')
    
    # ========== 阶段4：结果展示 ==========
    print("\n商户流失风险分析结果:")
    print("-"*80)
    
    # 选择关键列显示
    if 'churn_probability' in predictions_df.columns:
        display_cols = [
            'merchant_id', 'churn_probability', 'risk_level', 
            'total_amount_60d', 'active_rate', 'amount_trend', 
            'consecutive_inactive_days'
        ]
    else:
        display_cols = [
            'merchant_id', 'risk_level', 'total_amount_60d', 
            'active_rate', 'amount_trend', 'consecutive_inactive_days'
        ]
    
    display_df = predictions_df[
        [col for col in display_cols if col in predictions_df.columns]
    ].head(20).copy()
    
    # 格式化显示
    if 'churn_probability' in display_df.columns:
        display_df['churn_probability'] = display_df['churn_probability'].apply(
            lambda x: f'{x:.1%}'
        )
    
    display_df['total_amount_60d'] = display_df['total_amount_60d'].apply(
        lambda x: f'{x:,.0f}'
    )
    display_df['active_rate'] = display_df['active_rate'].apply(
        lambda x: f'{x:.1%}'
    )
    display_df['amount_trend'] = display_df['amount_trend'].apply(
        lambda x: f'{x:+.1%}'
    )
    
    print(display_df.to_string(index=False))
    
    # ========== 阶段5：统计汇总 ==========
    print("\n" + "="*80)
    print("阶段5：统计汇总")
    print("="*80)
    
    risk_dist = predictions_df['risk_level'].value_counts()
    print("\n风险分布:")
    for level, count in risk_dist.items():
        pct = count / len(predictions_df) * 100
        print(f"  {level}: {count} 个 ({pct:.1f}%)")
    
    # 高风险商户统计
    high_risk_df = predictions_df[predictions_df['risk_level'] == '高风险']
    if len(high_risk_df) > 0:
        print(f"\n高风险商户分析:")
        print(f"  数量: {len(high_risk_df)} 个")
        print(f"  60天交易总额: {high_risk_df['total_amount_60d'].sum():,.0f}")
        print(f"  平均活跃率: {high_risk_df['active_rate'].mean():.1%}")
        print(f"  平均金额趋势: {high_risk_df['amount_trend'].mean():+.1%}")
    
    # 按交易额分层统计
    print("\n按交易额分层的流失风险:")
    predictions_df['amount_tier'] = pd.qcut(
        predictions_df['total_amount_60d'], 
        q=4, 
        labels=['低', '中低', '中高', '高'],
        duplicates='drop'
    )
    
    tier_risk = pd.crosstab(
        predictions_df['amount_tier'], 
        predictions_df['risk_level'],
        normalize='index'
    ) * 100
    
    print(tier_risk.round(1))
    
    # 保存结果
    output_file = '/workspace/merchant_churn_predictions.csv'
    predictions_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n完整预测结果已保存至: {output_file}")
    
    return predictor, predictions_df


def main():
    """主函数"""
    
    print("="*80)
    print("商户流失预测系统 - 实际数据分析")
    print("="*80)
    
    # ========== 配置参数 ==========
    # 请修改为您的实际数据文件路径
    DATA_FILE = "C:\\Users\\Xiangdong.Yan\\doc\\数据中心\\ml\\mer_trans_data.csv"
    
    # 如果文件不存在，提示用户
    import os
    if not os.path.exists(DATA_FILE):
        print(f"\n错误：数据文件不存在: {DATA_FILE}")
        print("\n请修改 DATA_FILE 变量为您的实际数据文件路径")
        print("然后重新运行脚本")
        
        # 使用示例数据演示
        print("\n" + "="*80)
        print("使用示例数据进行演示...")
        print("="*80)
        
        from merchant_analysis_example import main as run_example
        run_example()
        return
    
    # ========== 加载数据 ==========
    try:
        shop_dict = load_and_prepare_data(
            file_path=DATA_FILE,
            sep="@@@",
            min_date='2025-07-03'
        )
    except Exception as e:
        print(f"\n加载数据时出错: {str(e)}")
        print("\n请检查：")
        print("1. 文件路径是否正确")
        print("2. 文件分隔符是否为 '@@@'")
        print("3. 列名是否包含: _shop_id, pt, amount, number")
        return
    
    # ========== 分析商户 ==========
    # 如果商户数量太多，可以设置 sample_size 进行采样
    # sample_size=1000 表示随机抽取1000个商户分析
    predictor, predictions_df = analyze_merchants(
        shop_dict,
        sample_size=None,  # None表示分析所有商户
        train_model=True
    )
    
    # ========== 详细分析高风险商户 ==========
    print("\n" + "="*80)
    print("高风险商户详细分析（Top 5）")
    print("="*80)
    
    if 'churn_probability' in predictions_df.columns:
        top_risk = predictions_df.nlargest(5, 'churn_probability')
    else:
        top_risk = predictions_df[
            predictions_df['risk_level'] == '高风险'
        ].nlargest(5, 'total_amount_60d')
    
    for idx, row in top_risk.iterrows():
        merchant_id = row['merchant_id']
        
        print(f"\n--- 商户 {merchant_id} ---")
        
        # 获取详细预测
        if merchant_id in shop_dict:
            prediction = predictor.predict_churn(
                shop_dict[merchant_id],
                model_type='ensemble'
            )
            
            if prediction.get('churn_probability'):
                print(f"流失概率: {prediction['churn_probability']:.1%}")
            
            print(f"风险等级: {prediction.get('risk_level', row['risk_level'])}")
            
            if 'features' in prediction:
                features = prediction['features']
                print(f"\n关键指标:")
                print(f"  60天交易额: {features['total_amount_60d']:,.0f}")
                print(f"  活跃率: {features['active_rate']:.1%}")
                print(f"  金额趋势: {features['amount_trend']:+.1%}")
                print(f"  交易笔数趋势: {features['count_trend']:+.1%}")
                print(f"  连续不活跃天数: {features['consecutive_inactive_days']:.0f}")
                print(f"  近期vs早期变化: {features['amount_change_rate']:+.1%}")
                
                # 自适应窗口信息
                classification = predictor.classify_merchant(shop_dict[merchant_id])
                print(f"\n商户类型: {classification['category_name']}")
                print(f"  短期窗口: {classification['category_info']['short_window']}天")
                print(f"  长期窗口: {classification['category_info']['long_window']}天")
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()
