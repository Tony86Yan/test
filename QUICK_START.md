# 快速开始指南

## 1分钟上手

### 基本用法

```python
from timeseries_trend_analysis import TimeSeriesTrendAnalyzer

# 你的数据
dates = ['2023-01-01', '2023-01-02', '2023-01-03', ...]
amounts = [100, 105, 103, 108, 112, ...]

# 创建分析器
analyzer = TimeSeriesTrendAnalyzer(dates, amounts)

# 综合分析（推荐）
result = analyzer.comprehensive_analysis()

# 查看结果
print(f"趋势: {result['consensus_trend']}")        # 上升/下降/平稳
print(f"程度: {result['consensus_degree']}")      # 强/中等/弱
print(f"置信度: {result['confidence']:.1f}%")    # 可靠程度
```

## 常用场景

### 场景1: 快速判断趋势

```python
# 使用线性回归（最快）
result = analyzer.method1_linear_regression()
print(f"{result['trend']} ({result['degree']})")
```

### 场景2: 数据有异常值

```python
# 使用Mann-Kendall检验（抗异常值）
result = analyzer.method2_mann_kendall()
print(f"Kendall's Tau: {result['tau']:.3f}")
print(f"{result['trend']} ({result['degree']})")
```

### 场景3: 非线性趋势

```python
# 使用多项式回归
result = analyzer.method3_polynomial_regression(degree=2)
print(f"曲率: {result['curvature']}")
print(f"{result['trend']} ({result['degree']})")
```

### 场景4: 噪声数据

```python
# 使用移动平均斜率
result = analyzer.method4_moving_average_slope(window=5)
print(f"一致性: {result['consistency']:.1%}")
print(f"{result['trend']} ({result['degree']})")
```

### 场景5: 检测趋势转折

```python
# 使用变点检测
result = analyzer.method7_change_point_detection()
if 'change_point' in result:
    print(f"变化点: {result['change_point']}")
    print(f"第一段: 斜率={result['seg1_slope']:.2f}")
    print(f"第二段: 斜率={result['seg2_slope']:.2f}")
```

## 数据格式

### 支持的输入格式

```python
# 格式1: Python列表
dates = ['2023-01-01', '2023-01-02', ...]
amounts = [100, 105, 103, ...]

# 格式2: NumPy数组
import numpy as np
dates = pd.date_range('2023-01-01', periods=30, freq='D')
amounts = np.array([100, 105, 103, ...])

# 格式3: Pandas Series
import pandas as pd
df = pd.read_csv('data.csv')
dates = df['date']
amounts = df['amount']

# 创建分析器（所有格式通用）
analyzer = TimeSeriesTrendAnalyzer(dates, amounts)
```

## 结果解读

### 趋势方向
- **上升**: 数值随时间增加
- **下降**: 数值随时间减少
- **平稳**: 无明显趋势

### 趋势程度
- **强**: 趋势明显，方向一致，R²>0.7 或 |τ|>0.5
- **中等**: 趋势可见，略有波动，R²>0.4 或 |τ|>0.3
- **弱**: 趋势不明显，波动较大
- **无显著趋势**: 统计上不显著 (p≥0.05)

### 置信度
- **>70%**: 高可信度，多数方法一致
- **50-70%**: 中等可信度
- **<50%**: 低可信度，建议人工审查

## 7种方法对比

| 方法 | 代码 | 适用场景 |
|------|------|---------|
| 线性回归 | `method1_linear_regression()` | 线性趋势，快速分析 |
| Mann-Kendall | `method2_mann_kendall()` | 有异常值，单调趋势 |
| 多项式回归 | `method3_polynomial_regression(degree=2)` | 非线性，曲线趋势 |
| 移动平均 | `method4_moving_average_slope(window=5)` | 噪声大，需要平滑 |
| Spearman | `method5_spearman_correlation()` | 异常值，单调关系 |
| 去趋势波动 | `method6_detrended_fluctuation()` | 长期趋势，复杂数据 |
| 变点检测 | `method7_change_point_detection()` | 趋势转折 |

## 实用技巧

### 技巧1: 获取所有指标

```python
result = analyzer.comprehensive_analysis()

# 查看每个方法的详细结果
for r in result['individual_results']:
    print(f"\n{r['method']}:")
    for key, value in r.items():
        if key not in ['method']:
            print(f"  {key}: {value}")
```

### 技巧2: 自定义判断阈值

```python
result = analyzer.method1_linear_regression()

# 自定义判断逻辑
if result['p_value'] < 0.01:  # 更严格的显著性
    if abs(result['slope']) > 1.0:  # 自定义斜率阈值
        print("显著趋势")
```

### 技巧3: 批量分析多个时间序列

```python
# 假设有多个产品的销售数据
products = ['产品A', '产品B', '产品C']
all_results = {}

for product in products:
    dates = get_dates_for(product)
    amounts = get_amounts_for(product)
    
    analyzer = TimeSeriesTrendAnalyzer(dates, amounts)
    all_results[product] = analyzer.comprehensive_analysis()

# 比较结果
for product, result in all_results.items():
    print(f"{product}: {result['consensus_trend']} ({result['consensus_degree']})")
```

### 技巧4: 从CSV文件读取

```python
import pandas as pd

# 读取CSV
df = pd.read_csv('data.csv')

# 确保日期格式正确
df['date'] = pd.to_datetime(df['date'])

# 创建分析器
analyzer = TimeSeriesTrendAnalyzer(df['date'], df['amount'])
result = analyzer.comprehensive_analysis()
```

## 常见问题

### Q1: 数据点太少怎么办？
**A**: 至少需要10个数据点，推荐20个以上。数据点少时使用线性回归或Spearman相关。

### Q2: 如何处理缺失值？
**A**: 在创建分析器前处理：
```python
df = df.dropna()  # 删除缺失值
# 或
df = df.fillna(method='ffill')  # 前向填充
```

### Q3: 如何处理异常值？
**A**: 
- 方法1: 使用抗异常值的方法（Mann-Kendall、Spearman）
- 方法2: 预先清洗异常值
```python
# 使用3σ规则清洗异常值
mean, std = amounts.mean(), amounts.std()
amounts_clean = amounts[(amounts > mean - 3*std) & (amounts < mean + 3*std)]
```

### Q4: 日期不等间隔怎么办？
**A**: 代码自动处理不等间隔日期，但最好重采样为等间隔：
```python
df = df.set_index('date').resample('D').mean()  # 重采样为每日
```

### Q5: 如何选择多项式的次数？
**A**: 
- degree=2: 识别加速/减速（推荐）
- degree=3: 识别更复杂的曲线
- 不要超过3次，容易过拟合

### Q6: 不同方法结果不一致怎么办？
**A**: 
- 查看置信度，<70%说明数据特征复杂
- 检查原始数据是否有问题
- 结合业务背景判断
- 可视化数据验证

## 下一步

- 📖 查看[详细文档](TIMESERIES_TREND_METHODS.md)了解每种方法的原理
- 💻 运行[示例代码](example_usage.py)查看8个实用案例
- 🎯 将代码应用到自己的数据

## 示例输出

```
综合分析结果:
================================================================================
📊 趋势方向: 上升
📈 趋势程度: 强
✅ 置信度: 87.5%

投票分布:
  上升: 7 票
  下降: 0 票
  平稳: 1 票
  
关键指标:
  • 线性斜率: 5.23
  • 拟合优度: 0.9542
  • Kendall's Tau: 0.8765
  • p值: < 0.0001
```

---

**需要帮助？** 查看 [完整文档](TIMESERIES_TREND_METHODS.md) 或运行示例程序。
