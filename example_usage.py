"""
时间序列趋势分析使用示例
演示如何使用TimeSeriesTrendAnalyzer分析实际数据
"""

import pandas as pd
import numpy as np
from timeseries_trend_analysis import TimeSeriesTrendAnalyzer


def example1_csv_data():
    """示例1: 从CSV文件读取数据"""
    print("\n" + "="*80)
    print("示例1: 从CSV文件读取数据")
    print("="*80)
    
    # 创建示例CSV数据
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    amounts = 1000 + np.cumsum(np.random.normal(5, 20, 100))
    
    df = pd.DataFrame({
        'date': dates,
        'amount': amounts
    })
    
    # 保存为CSV
    df.to_csv('/tmp/sales_data.csv', index=False)
    print("已创建示例数据文件: /tmp/sales_data.csv")
    
    # 读取CSV并分析
    df = pd.read_csv('/tmp/sales_data.csv')
    analyzer = TimeSeriesTrendAnalyzer(df['date'], df['amount'])
    
    # 使用线性回归方法
    result = analyzer.method1_linear_regression()
    print(f"\n线性回归分析:")
    print(f"  趋势: {result['trend']}")
    print(f"  程度: {result['degree']}")
    print(f"  斜率: {result['slope']:.2f}")
    print(f"  R²: {result['r2']:.4f}")
    print(f"  p值: {result['p_value']:.4f}")


def example2_compare_methods():
    """示例2: 比较不同方法的结果"""
    print("\n" + "="*80)
    print("示例2: 比较多种分析方法")
    print("="*80)
    
    # 创建有明显上升趋势的数据
    dates = pd.date_range(start='2023-01-01', periods=50, freq='W')
    amounts = 500 + 10 * np.arange(50) + np.random.normal(0, 15, 50)
    
    analyzer = TimeSeriesTrendAnalyzer(dates, amounts)
    
    # 应用多种方法
    methods = [
        ('线性回归', analyzer.method1_linear_regression),
        ('Mann-Kendall检验', analyzer.method2_mann_kendall),
        ('二次多项式', lambda: analyzer.method3_polynomial_regression(2)),
        ('Spearman相关', analyzer.method5_spearman_correlation),
    ]
    
    print("\n方法对比:")
    print("-" * 80)
    for name, method in methods:
        result = method()
        print(f"{name:20s} | 趋势: {result['trend']:6s} | 程度: {result['degree']}")


def example3_nonlinear_trend():
    """示例3: 识别非线性趋势"""
    print("\n" + "="*80)
    print("示例3: 非线性趋势识别")
    print("="*80)
    
    # 创建二次函数趋势的数据
    dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
    x = np.linspace(0, 10, 60)
    amounts = 100 + 5*x + 0.5*x**2 + np.random.normal(0, 5, 60)
    
    analyzer = TimeSeriesTrendAnalyzer(dates, amounts)
    
    # 比较线性和非线性方法
    linear = analyzer.method1_linear_regression()
    poly2 = analyzer.method3_polynomial_regression(degree=2)
    poly3 = analyzer.method3_polynomial_regression(degree=3)
    
    print("\n线性回归:")
    print(f"  R²: {linear['r2']:.4f}")
    print(f"  趋势: {linear['trend']} ({linear['degree']})")
    
    print("\n二次多项式回归:")
    print(f"  R²: {poly2['r2']:.4f}")
    print(f"  趋势: {poly2['trend']} ({poly2['degree']})")
    print(f"  曲率特征: {poly2['curvature']}")
    
    print("\n三次多项式回归:")
    print(f"  R²: {poly3['r2']:.4f}")
    print(f"  趋势: {poly3['trend']} ({poly3['degree']})")


def example4_comprehensive_analysis():
    """示例4: 综合分析（推荐方法）"""
    print("\n" + "="*80)
    print("示例4: 综合分析（推荐使用）")
    print("="*80)
    
    # 创建带噪声的数据
    dates = pd.date_range(start='2023-01-01', periods=80, freq='D')
    trend = np.linspace(100, 180, 80)
    noise = np.random.normal(0, 10, 80)
    amounts = trend + noise
    
    analyzer = TimeSeriesTrendAnalyzer(dates, amounts)
    
    # 综合分析
    result = analyzer.comprehensive_analysis()
    
    print("\n综合分析结果:")
    print("=" * 80)
    print(f"📊 趋势方向: {result['consensus_trend']}")
    print(f"📈 趋势程度: {result['consensus_degree']}")
    print(f"✅ 置信度: {result['confidence']:.1f}%")
    print(f"\n投票分布:")
    for trend, count in result['trend_votes'].items():
        print(f"  {trend}: {count} 票")
    
    print("\n各方法详细结果:")
    print("-" * 80)
    for r in result['individual_results']:
        print(f"\n{r['method']}:")
        print(f"  ➤ 趋势: {r['trend']}")
        print(f"  ➤ 程度: {r['degree']}")


def example5_change_point():
    """示例5: 检测趋势变化点"""
    print("\n" + "="*80)
    print("示例5: 趋势变化点检测")
    print("="*80)
    
    # 创建有变点的数据：前半段上升，后半段下降
    dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
    amounts = np.concatenate([
        np.linspace(100, 150, 30) + np.random.normal(0, 5, 30),
        np.linspace(150, 120, 30) + np.random.normal(0, 5, 30)
    ])
    
    analyzer = TimeSeriesTrendAnalyzer(dates, amounts)
    
    result = analyzer.method7_change_point_detection()
    
    print("\n变点检测结果:")
    print(f"  整体趋势: {result['trend']}")
    print(f"  趋势特征: {result['degree']}")
    if 'change_point' in result:
        print(f"  变化点: {result['change_point']}")
        print(f"  第一段斜率: {result['seg1_slope']:.4f}")
        print(f"  第二段斜率: {result['seg2_slope']:.4f}")


def example6_moving_average():
    """示例6: 移动平均斜率分析"""
    print("\n" + "="*80)
    print("示例6: 移动平均斜率分析（适合噪声数据）")
    print("="*80)
    
    # 创建高噪声数据
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    trend = np.linspace(200, 280, 100)
    noise = np.random.normal(0, 20, 100)  # 高噪声
    amounts = trend + noise
    
    analyzer = TimeSeriesTrendAnalyzer(dates, amounts)
    
    # 尝试不同的窗口大小
    for window in [3, 5, 10]:
        result = analyzer.method4_moving_average_slope(window=window)
        print(f"\n窗口大小 = {window}:")
        print(f"  趋势: {result['trend']}")
        print(f"  程度: {result['degree']}")
        print(f"  一致性: {result['consistency']:.2%}")


def example7_custom_data():
    """示例7: 使用自己的数据"""
    print("\n" + "="*80)
    print("示例7: 自定义数据格式")
    print("="*80)
    
    # 方式1: 使用列表
    dates_list = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
    amounts_list = [100, 105, 103, 108, 112]
    
    analyzer = TimeSeriesTrendAnalyzer(dates_list, amounts_list)
    result = analyzer.method1_linear_regression()
    print(f"\n方式1 - 列表格式:")
    print(f"  趋势: {result['trend']}, 程度: {result['degree']}")
    
    # 方式2: 使用numpy数组
    dates_array = pd.date_range('2023-01-01', periods=10, freq='D')
    amounts_array = np.array([100, 102, 105, 103, 108, 110, 109, 113, 115, 118])
    
    analyzer = TimeSeriesTrendAnalyzer(dates_array, amounts_array)
    result = analyzer.method1_linear_regression()
    print(f"\n方式2 - NumPy数组:")
    print(f"  趋势: {result['trend']}, 程度: {result['degree']}")
    
    # 方式3: 从DataFrame
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=15, freq='D'),
        'amount': np.random.randint(100, 200, 15)
    })
    
    analyzer = TimeSeriesTrendAnalyzer(df['date'].values, df['amount'].values)
    result = analyzer.method1_linear_regression()
    print(f"\n方式3 - DataFrame:")
    print(f"  趋势: {result['trend']}, 程度: {result['degree']}")


def example8_practical_scenario():
    """示例8: 实际业务场景"""
    print("\n" + "="*80)
    print("示例8: 实际业务场景 - 月度销售额分析")
    print("="*80)
    
    # 模拟12个月的销售数据
    months = pd.date_range(start='2023-01-01', periods=12, freq='MS')
    sales = np.array([
        120000, 125000, 130000, 128000, 135000, 140000,
        145000, 150000, 148000, 155000, 160000, 165000
    ])
    
    analyzer = TimeSeriesTrendAnalyzer(months, sales)
    result = analyzer.comprehensive_analysis()
    
    print("\n📊 销售趋势分析报告")
    print("=" * 80)
    print(f"分析期间: {months[0].strftime('%Y-%m')} 至 {months[-1].strftime('%Y-%m')}")
    print(f"数据点数: {len(months)} 个月")
    print(f"\n核心结论:")
    print(f"  • 趋势方向: {result['consensus_trend']}")
    print(f"  • 趋势强度: {result['consensus_degree']}")
    print(f"  • 分析置信度: {result['confidence']:.1f}%")
    
    # 具体指标
    linear = [r for r in result['individual_results'] if r['method'] == '线性回归'][0]
    print(f"\n关键指标:")
    print(f"  • 月均增长: {linear['slope']:.0f} 元/月")
    print(f"  • 拟合度(R²): {linear['r2']:.2%}")
    print(f"  • 统计显著性: p = {linear['p_value']:.4f}")
    
    if result['consensus_trend'] == '上升':
        print(f"\n💡 业务建议:")
        if result['consensus_degree'] == '强':
            print(f"  ✓ 销售增长强劲，建议扩大市场投入")
            print(f"  ✓ 考虑增加库存以应对需求增长")
        elif result['consensus_degree'] == '中等':
            print(f"  ✓ 销售稳步增长，保持现有策略")
            print(f"  ✓ 关注市场变化，适时调整")
        else:
            print(f"  ⚠ 增长趋势较弱，需要分析原因")


if __name__ == "__main__":
    print("\n")
    print("🎯 " + "="*76 + " 🎯")
    print("   时间序列趋势分析 - 实用示例集")
    print("🎯 " + "="*76 + " 🎯")
    
    # 运行所有示例
    example1_csv_data()
    example2_compare_methods()
    example3_nonlinear_trend()
    example4_comprehensive_analysis()
    example5_change_point()
    example6_moving_average()
    example7_custom_data()
    example8_practical_scenario()
    
    print("\n" + "="*80)
    print("✅ 所有示例运行完成！")
    print("="*80)
    print("\n💡 提示:")
    print("  1. 对于大多数情况，推荐使用 comprehensive_analysis() 方法")
    print("  2. 如果数据有明显非线性特征，使用多项式回归")
    print("  3. 如果数据有异常值，优先使用Mann-Kendall或Spearman方法")
    print("  4. 对于噪声大的数据，使用移动平均斜率方法")
    print("\n📚 更多详细说明请参考: TIMESERIES_TREND_METHODS.md\n")
