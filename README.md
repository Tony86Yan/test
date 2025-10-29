# 商户流失预测系统

基于机器学习的商户流失预测解决方案，整合Logistic Regression、Random Forest、XGBoost等多种模型，实现自适应窗口Moving Average分析。

## 🎯 核心特性

### 1. 自适应窗口调整
根据商户交易频率（高频/中频/低频）自动调整Moving Average窗口长度：
- **高频商户**（活跃率≥60%）：使用7天短期窗口快速捕捉变化
- **低频商户**（活跃率10-25%）：使用30天长期窗口平滑噪声

### 2. 时间衰减权重
采用指数衰减权重（decay_rate=0.95），越近的交易权重越高，更快反映趋势变化。

### 3. 机器学习模型
- **Logistic Regression**: 可解释性强，特征重要性分析
- **Random Forest**: 捕捉非线性关系
- **XGBoost**: 高准确率梯度提升
- **Ensemble**: 集成预测，提高稳定性

### 4. 丰富的特征工程
自动提取30+特征：
- 活跃度特征（加权活跃率、连续不活跃天数等）
- 交易量特征（60天总额、平均单笔金额等）
- 自适应MA特征（根据商户频率动态调整窗口）
- 趋势特征（短期vs长期、近期vs早期）
- 波动性特征（标准差、变异系数）

## 📦 安装

### 环境要求
- Python 3.7+
- pip

### 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 方法1: 一键启动（推荐新手）

```bash
python3 quick_start.py
```

按照交互式提示选择：
1. 使用示例数据演示
2. 使用实际数据分析
3. 查看已有预测结果

### 方法2: 运行示例

```bash
python3 merchant_analysis_example.py
```

使用模拟数据展示完整功能，包括：
- 商户分类
- 特征提取
- 模型训练
- 批量预测
- 可视化分析

### 方法3: 使用实际数据

1. 准备CSV数据文件（分隔符为`@@@`）
2. 修改`run_with_real_data.py`中的文件路径
3. 运行分析：

```bash
python3 run_with_real_data.py
```

## 💻 代码示例

### 基础使用

```python
from merchant_churn_prediction import MerchantChurnPredictor

# 创建预测器
predictor = MerchantChurnPredictor(
    observation_days=60,  # 观察期60天
    decay_rate=0.95       # 时间衰减率
)

# 准备数据（字典格式）
merchants_data = {
    'merchant_001': pd.DataFrame({
        'date': [...],
        'transaction_count': [...],
        'amount': [...]
    }),
    # 更多商户...
}

# 训练模型
X, y, ids, features = predictor.prepare_training_data(merchants_data)
results = predictor.train_models(X, y)

# 批量预测
predictions_df = predictor.batch_predict(merchants_data, model_type='ensemble')

# 查看高风险商户
high_risk = predictions_df[predictions_df['risk_level'] == '高风险']
print(high_risk)
```

### 单个商户预测

```python
# 预测单个商户
prediction = predictor.predict_churn(
    merchant_data,
    model_type='ensemble'  # 可选: 'lr', 'rf', 'xgb', 'ensemble'
)

print(f"流失概率: {prediction['churn_probability']:.1%}")
print(f"风险等级: {prediction['risk_level']}")
```

### 可视化分析

```python
from visualization_analysis import ChurnVisualization

viz = ChurnVisualization()

# 生成完整报告（包含多个图表）
viz.generate_comprehensive_report(
    predictions_df,
    model_results=results,
    output_dir='/workspace/'
)
```

## 📊 数据格式

### 输入数据格式

CSV文件，分隔符`@@@`，包含以下列：

```
merchant_id      - 商户ID
merchant_name    - 商户名称  
_shop_id        - 门店ID
_shop_name      - 门店名称
check_date      - 检查日期
shop_create_date - 门店创建日期
pt              - 交易日期（必需）
amount          - 交易金额（必需）
number          - 交易笔数（必需）
```

### 输出结果

**CSV文件**: `merchant_churn_predictions.csv`

| 列名 | 说明 |
|------|------|
| merchant_id | 商户ID |
| churn_probability | 流失概率（0-1） |
| risk_level | 风险等级（高风险/中风险/低风险） |
| total_amount_60d | 60天交易总额 |
| active_rate | 活跃率 |
| amount_trend | 金额趋势（负值表示下降） |
| consecutive_inactive_days | 连续不活跃天数 |

**图表文件**:
- `risk_distribution.png` - 风险分布
- `trend_analysis.png` - 趋势分析
- `category_analysis.png` - 商户分类分析
- `feature_importance_*.png` - 特征重要性
- `confusion_matrix_*.png` - 混淆矩阵

## 🔧 参数调整

### 调整观察期

```python
predictor = MerchantChurnPredictor(
    observation_days=90,  # 改为90天
    decay_rate=0.95
)
```

### 调整商户分类窗口

修改`merchant_churn_prediction.py`中的`merchant_categories`字典：

```python
self.merchant_categories = {
    'very_active': {
        'short_window': 5,   # 自定义短期窗口
        'long_window': 10,   # 自定义长期窗口
        ...
    }
}
```

### 调整流失判断阈值

在`prepare_training_data()`方法中修改：

```python
is_churn = (
    (features['amount_trend'] < -0.2) or  # 自定义阈值
    (features['consecutive_inactive_days'] > 10) or
    ...
)
```

## 📚 文档

| 文档 | 说明 |
|------|------|
| **README.md** | 本文档，快速开始 |
| **USAGE_GUIDE.md** | 详细使用指南 |
| **README_CHURN_PREDICTION.md** | 完整技术文档 |
| **PROJECT_SUMMARY.md** | 项目总结和设计思路 |

## 📁 项目结构

```
/workspace/
├── merchant_churn_prediction.py      # 核心预测类
├── merchant_analysis_example.py      # 示例演示
├── run_with_real_data.py            # 实际数据分析脚本
├── visualization_analysis.py         # 可视化分析
├── quick_start.py                   # 一键启动脚本
├── requirements.txt                 # Python依赖
├── README.md                        # 本文档
├── USAGE_GUIDE.md                   # 使用指南
├── README_CHURN_PREDICTION.md       # 技术文档
└── PROJECT_SUMMARY.md               # 项目总结
```

## 🎓 核心算法

### 自适应窗口Moving Average

1. **商户分类**: 根据加权活跃率将商户分为5类（高频、中高频、中频、低频、极低频）
2. **窗口选择**: 不同类别使用不同的MA窗口长度
3. **时间权重**: 应用指数衰减权重（0.95^i）
4. **加权计算**: weighted_ma = Σ(data * weights) / Σ(weights)

### 流失判断规则

```python
is_churn = (
    (amount_trend < -0.3) or           # 交易额下降>30%
    (consecutive_inactive_days > 14) or # 连续14天无交易
    (active_rate_last_30d < 0.2 and     # 高频商户活跃率骤降
     merchant_category_code >= 3) or
    (amount_change_rate < -0.5)        # 近期vs早期下降>50%
)
```

## 🔍 特征重要性（示例）

根据实验，Top 5最重要特征：

1. **consecutive_inactive_days** - 连续不活跃天数
2. **amount_trend** - 金额趋势
3. **amount_change_rate** - 近期vs早期交易额变化
4. **active_rate_last_30d** - 最近30天活跃率
5. **weighted_active_rate** - 加权活跃率

## 💡 业务应用

### 预警机制

```python
# 识别高风险高价值商户
critical = predictions_df[
    (predictions_df['churn_probability'] > 0.7) &
    (predictions_df['total_amount_60d'] > 100000)
]

# 发送预警
for _, merchant in critical.iterrows():
    send_alert(merchant['merchant_id'], merchant['churn_probability'])
```

### 分级干预

- **高风险**（流失概率≥70%）：客户经理电话 + 费率优惠
- **中风险**（流失概率40-70%）：发送优惠券 + 产品推荐  
- **低风险**（流失概率<40%）：常规维护 + 交叉销售

## 🚨 常见问题

**Q: 为什么高频商户使用短窗口？**

A: 高频商户每天都有交易，7天窗口已包含足够信息。短窗口能更快捕捉趋势变化，而长窗口会延迟信号。

**Q: 如何处理数据不足的商户？**

A: 系统自动跳过数据少于3天的商户。建议至少有30-60天历史数据。

**Q: 模型准确率不高怎么办？**

A: 
1. 使用真实流失标签（而非自动标注）
2. 增加训练样本数量
3. 添加更多特征（如行业、地区）
4. 调整阈值参数

**Q: 能否实时预测？**

A: 可以。训练好模型后，使用`predict_churn()`方法实时预测新商户。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目仅供内部使用。

## 📧 联系方式

如有问题，请联系数据分析团队。

---

**版本**: 1.0  
**更新日期**: 2025-10-29  
**作者**: AI Assistant

## 🌟 下一步

1. ✅ 安装依赖: `pip install -r requirements.txt`
2. ✅ 运行示例: `python3 merchant_analysis_example.py`
3. ✅ 查看文档: `USAGE_GUIDE.md`
4. ✅ 使用实际数据: 修改`run_with_real_data.py`
5. ✅ 部署到生产环境

**祝使用愉快！** 🎉
