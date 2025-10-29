# 商户流失预测系统 - 使用指南

## 快速开始（3分钟）

### 方法1：一键运行（推荐）

```bash
python quick_start.py
```

按照交互式提示选择：
1. 使用示例数据演示
2. 使用实际数据分析
3. 查看已有结果

### 方法2：直接运行示例

```bash
python merchant_analysis_example.py
```

这将使用模拟数据展示完整功能。

## 核心功能说明

### 1. 自适应窗口调整

系统会自动识别商户交易频率，并调整Moving Average窗口：

```
高频商户（活跃率≥60%）  → 短期7天，长期15天
中频商户（活跃率25-40%） → 短期15天，长期30天
低频商户（活跃率10-25%） → 短期15天，长期30天
```

**原理**：高频商户交易密集，用短窗口快速捕捉变化；低频商户交易稀疏，用长窗口平滑噪声。

### 2. Logistic Regression预测

```python
from merchant_churn_prediction import MerchantChurnPredictor

# 创建预测器
predictor = MerchantChurnPredictor(
    observation_days=60,  # 观察期60天
    decay_rate=0.95       # 时间衰减率
)

# 训练模型
X, y, ids, features = predictor.prepare_training_data(merchants_data)
results = predictor.train_models(X, y)

# 预测单个商户
prediction = predictor.predict_churn(
    merchant_data,
    model_type='lr'  # 'lr', 'rf', 'xgb', 'ensemble'
)

print(f"流失概率: {prediction['churn_probability']:.1%}")
```

### 3. 特征提取

系统自动提取30+特征：

**活跃度特征**
- 加权活跃率（时间衰减）
- 连续不活跃天数
- 最近7/15/30天活跃率

**交易量特征**
- 60天总交易额/笔数
- 平均单笔金额
- 活跃日平均交易额

**趋势特征**（自适应窗口）
- 短期vs长期MA差异
- 近30天vs前30天变化率
- 金额趋势、笔数趋势

**波动性特征**
- 交易额标准差
- 变异系数

### 4. 流失判断标准

自动标注使用以下规则：

```python
is_churn = (
    (amount_trend < -0.3) or           # 交易额下降>30%
    (consecutive_inactive_days > 14) or # 连续14天无交易
    (active_rate_last_30d < 0.2 and     # 高频商户活跃率骤降
     merchant_category_code >= 3) or
    (amount_change_rate < -0.5)        # 近期vs早期下降>50%
)
```

可以根据业务需求调整阈值。

## 使用实际数据

### 数据格式要求

CSV文件，分隔符 `@@@`，包含以下列：

```
merchant_id      - 商户ID
merchant_name    - 商户名称
_shop_id        - 门店ID
_shop_name      - 门店名称
check_date      - 检查日期
shop_create_date - 门店创建日期
pt              - 交易日期
amount          - 交易金额
number          - 交易笔数
```

### 示例代码

```python
from run_with_real_data import load_and_prepare_data, analyze_merchants

# 加载数据
shop_dict = load_and_prepare_data(
    file_path="your_data.csv",
    sep="@@@",
    min_date='2025-07-03'
)

# 分析（可选采样）
predictor, predictions_df = analyze_merchants(
    shop_dict,
    sample_size=1000,  # None表示全部
    train_model=True
)

# 结果保存在 /workspace/merchant_churn_predictions.csv
```

## 可视化分析

```python
from visualization_analysis import ChurnVisualization

# 创建可视化对象
viz = ChurnVisualization()

# 生成完整报告
viz.generate_comprehensive_report(
    predictions_df,
    model_results=results,
    output_dir='/workspace/'
)
```

生成的图表包括：
- 风险分布图
- 趋势分析图
- 商户分类分析图
- 特征重要性图
- 混淆矩阵

## 输出结果

### 1. CSV文件

`/workspace/merchant_churn_predictions.csv`

| 字段 | 说明 |
|------|------|
| merchant_id | 商户ID |
| churn_probability | 流失概率 |
| risk_level | 风险等级（高/中/低） |
| total_amount_60d | 60天交易总额 |
| active_rate | 活跃率 |
| amount_trend | 金额趋势 |
| consecutive_inactive_days | 连续不活跃天数 |

### 2. 图表文件

- `risk_distribution.png` - 风险分布
- `trend_analysis.png` - 趋势分析
- `category_analysis.png` - 商户分类分析
- `feature_importance_*.png` - 特征重要性
- `confusion_matrix_*.png` - 混淆矩阵

## 模型对比

| 模型 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| Logistic Regression | 可解释性强、快速 | 只能捕捉线性关系 | 快速评估、特征分析 |
| Random Forest | 非线性、鲁棒 | 计算慢 | 复杂模式识别 |
| XGBoost | 高准确率 | 需要调参 | 最终预测 |
| Ensemble | 稳定性好 | 计算慢 | 生产环境推荐 |

## 参数调优

### 调整观察期

```python
predictor = MerchantChurnPredictor(
    observation_days=90,  # 默认60
    decay_rate=0.95
)
```

### 调整窗口大小

修改 `merchant_churn_prediction.py`：

```python
self.merchant_categories = {
    'very_active': {
        'short_window': 5,   # 改为5天
        'long_window': 10,   # 改为10天
        ...
    }
}
```

### 调整流失阈值

在 `prepare_training_data()` 方法中修改：

```python
is_churn = (
    (features['amount_trend'] < -0.2) or  # 改为-0.2
    (features['consecutive_inactive_days'] > 10) or  # 改为10天
    ...
)
```

## 常见问题

**Q1: 为什么高频商户用短窗口？**

A: 高频商户每天都有交易，7天窗口已经包含足够信息，能更快反映趋势变化。长窗口会延迟变化信号。

**Q2: 如何处理数据不足？**

A: 系统自动跳过数据少于3天的商户。建议至少有30天数据。

**Q3: 模型准确率低怎么办？**

A: 
1. 使用真实流失标签（而非自动标注）
2. 增加训练样本
3. 添加更多特征（如行业、地区）
4. 调整阈值参数

**Q4: 能否实时预测？**

A: 可以。训练好模型后，使用 `predict_churn()` 实时预测新商户。

**Q5: 如何解释预测结果？**

A: 查看特征重要性排名，关注：
- 连续不活跃天数（最重要）
- 金额趋势
- 活跃率变化

## 业务应用建议

### 1. 预警机制

```python
# 每日运行
high_risk = predictions_df[
    (predictions_df['churn_probability'] > 0.7) &
    (predictions_df['total_amount_60d'] > 100000)  # 高价值商户
]

# 发送预警邮件
send_alert(high_risk)
```

### 2. 分级干预

- **高风险**：客户经理电话联系
- **中风险**：发送优惠券
- **低风险**：常规维护

### 3. 效果评估

跟踪干预后的流失率变化：

```python
# 记录干预
interventions = {
    'merchant_001': '2025-11-01',  # 干预日期
    ...
}

# 30天后评估
actual_churn = check_actual_churn(interventions)
success_rate = calculate_retention_rate(actual_churn)
```

## 技术支持

- 查看详细文档：`README_CHURN_PREDICTION.md`
- 示例代码：`merchant_analysis_example.py`
- 可视化示例：`visualization_analysis.py`

## 文件结构

```
/workspace/
├── merchant_churn_prediction.py      # 核心预测类
├── merchant_analysis_example.py      # 示例演示
├── run_with_real_data.py            # 实际数据脚本
├── visualization_analysis.py         # 可视化分析
├── quick_start.py                   # 快速开始脚本
├── requirements.txt                 # 依赖包
├── README_CHURN_PREDICTION.md       # 详细文档
└── USAGE_GUIDE.md                   # 本文档
```

## 下一步

1. 运行 `python quick_start.py` 体验功能
2. 准备实际数据
3. 根据业务调整参数
4. 部署到生产环境

---

**版本**: 1.0  
**更新日期**: 2025-10-29
