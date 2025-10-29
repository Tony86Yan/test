# 商户流失预测系统

## 项目概述

本项目为第三方支付公司提供了一套完整的商户流失预测解决方案，结合传统统计方法和现代机器学习算法，能够：

1. **自动识别不同交易频率的商户**（高频、中频、低频等）
2. **自适应调整分析窗口**：根据商户交易频率动态调整Moving Average窗口长度
3. **多模型预测**：使用Logistic Regression、Random Forest、XGBoost等多种模型
4. **特征工程**：从交易数据中提取30+个关键特征
5. **风险分级**：将商户分为高、中、低风险三个等级

## 核心特性

### 1. 自适应窗口调整

根据商户交易频率自动调整Moving Average窗口：

| 商户类型 | 活跃率范围 | 短期窗口 | 长期窗口 |
|---------|-----------|---------|---------|
| 高频商户 | ≥60% | 7天 | 15天 |
| 中高频商户 | 40%-60% | 7天 | 30天 |
| 中频商户 | 25%-40% | 15天 | 30天 |
| 低频商户 | 10%-25% | 15天 | 30天 |
| 极低频商户 | <10% | 15天 | 30天 |

**设计原理**：
- 高频商户交易密集，使用较短窗口能更快捕捉趋势变化
- 低频商户交易稀疏，使用较长窗口避免噪声干扰

### 2. 时间衰减权重

使用指数衰减权重，越近的交易权重越高：

```
权重 = decay_rate ^ i
```

- 最近一天权重 = 1.0
- 倒数第二天 = 0.95
- 倒数第三天 = 0.95² = 0.9025
- ...

### 3. 机器学习模型

#### Logistic Regression
- 优点：可解释性强，训练快速
- 特点：处理类别不平衡（class_weight='balanced'）
- 适用：快速评估、特征重要性分析

#### Random Forest
- 优点：非线性关系捕捉、鲁棒性强
- 特点：100棵树，最大深度10
- 适用：复杂模式识别

#### XGBoost
- 优点：高准确率、处理缺失值
- 特点：梯度提升、自动特征选择
- 适用：最终预测模型

#### 集成预测
取多个模型预测概率的平均值，提高稳定性。

### 4. 特征工程（30+特征）

#### 活跃度特征
- `weighted_active_rate`: 加权活跃率
- `active_days`: 活跃天数
- `active_rate`: 简单活跃率
- `consecutive_inactive_days`: 连续不活跃天数

#### 交易量特征
- `total_amount_60d`: 60天交易总额
- `total_count_60d`: 60天交易总笔数
- `avg_amount_per_txn`: 平均单笔金额
- `avg_amount_per_active_day`: 平均每活跃日交易额

#### 自适应移动平均特征
- `ma_amount_short`: 短期金额MA（窗口自适应）
- `ma_amount_long`: 长期金额MA（窗口自适应）
- `ma_count_short`: 短期笔数MA
- `ma_count_long`: 长期笔数MA

#### 趋势特征
- `amount_trend`: 金额趋势（短期vs长期）
- `count_trend`: 笔数趋势
- `amount_change_rate`: 近30天vs前30天变化率
- `active_days_change_rate`: 活跃天数变化率

#### 波动性特征
- `amount_std`: 金额标准差
- `count_std`: 笔数标准差
- `amount_cv`: 金额变异系数

#### 时间段特征
- `active_rate_last_7d`: 最近7天活跃率
- `active_rate_last_15d`: 最近15天活跃率
- `active_rate_last_30d`: 最近30天活跃率

## 安装

### 环境要求
- Python 3.7+
- pandas, numpy, scikit-learn, xgboost

### 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 方法1：使用示例数据

```bash
python merchant_analysis_example.py
```

这将使用模拟数据展示完整的分析流程。

### 方法2：使用实际数据

1. 准备数据文件（CSV格式）：

```
merchant_id@@@merchant_name@@@_shop_id@@@_shop_name@@@check_date@@@shop_create_date@@@pt@@@amount@@@number
```

2. 修改 `run_with_real_data.py` 中的文件路径：

```python
DATA_FILE = "your_data_path.csv"
```

3. 运行分析：

```bash
python run_with_real_data.py
```

### 方法3：在代码中使用

```python
from merchant_churn_prediction import MerchantChurnPredictor

# 创建预测器
predictor = MerchantChurnPredictor(
    observation_days=60,
    decay_rate=0.95
)

# 准备数据
# merchants_data = {merchant_id: DataFrame, ...}

# 提取特征并训练模型
X, y, merchant_ids, feature_names = predictor.prepare_training_data(merchants_data)
results = predictor.train_models(X, y)

# 预测单个商户
prediction = predictor.predict_churn(
    transaction_data,
    model_type='ensemble'
)

print(f"流失概率: {prediction['churn_probability']:.1%}")
print(f"风险等级: {prediction['risk_level']}")

# 批量预测
predictions_df = predictor.batch_predict(merchants_data)
```

## 流失判断标准

系统使用以下规则自动标注流失商户：

```python
is_churn = (
    (amount_trend < -0.3) or           # 交易额下降>30%
    (consecutive_inactive_days > 14) or # 连续14天无交易
    (active_rate_last_30d < 0.2 and     # 高频商户活跃率骤降
     merchant_category_code >= 3) or
    (amount_change_rate < -0.5)        # 近期vs早期下降>50%
)
```

### 风险等级定义

- **高风险**：流失概率 ≥ 70%
- **中风险**：流失概率 40%-70%
- **低风险**：流失概率 < 40%

## 输出结果

### 1. 预测结果CSV

包含以下字段：
- `merchant_id`: 商户ID
- `churn_probability`: 流失概率
- `risk_level`: 风险等级
- `total_amount_60d`: 60天交易总额
- `active_rate`: 活跃率
- `amount_trend`: 金额趋势
- `consecutive_inactive_days`: 连续不活跃天数

### 2. 模型评估报告

- 准确率（Accuracy）
- ROC-AUC分数
- 混淆矩阵
- 分类报告（精确率、召回率、F1分数）
- 特征重要性排名

### 3. 业务洞察

- 风险分布统计
- 潜在流失交易额
- 分层流失风险分析
- 高风险商户详细信息

## 业务建议

### 针对高风险商户
1. **立即行动**：联系商户了解业务变化
2. **激励措施**：提供费率优惠、返现等
3. **专属服务**：安排客户经理定期回访

### 针对中风险商户
1. **监控跟踪**：定期关注交易趋势
2. **主动沟通**：发送产品更新和优惠信息
3. **反馈收集**：了解服务改进需求

### 针对低风险商户
1. **维护关系**：保持现有服务质量
2. **交叉销售**：推荐其他增值服务
3. **推荐奖励**：鼓励转介绍新商户

## 模型优化建议

### 1. 调整窗口参数

根据业务特点调整不同频率商户的窗口长度：

```python
self.merchant_categories = {
    'very_active': {
        'short_window': 7,   # 可调整
        'long_window': 15,   # 可调整
        ...
    }
}
```

### 2. 调整流失判断标准

根据历史流失商户特征调整阈值：

```python
is_churn = (
    (amount_trend < -0.3) or  # 调整阈值
    (consecutive_inactive_days > 14) or  # 调整天数
    ...
)
```

### 3. 添加外部特征

可以添加更多特征：
- 商户行业类别
- 注册时长
- 历史投诉记录
- 竞争对手活动
- 宏观经济指标

### 4. 集成人工标注

使用实际流失标签训练模型：

```python
churn_labels = {
    'merchant_001': 1,  # 已流失
    'merchant_002': 0,  # 正常
    ...
}

X, y, merchant_ids, feature_names = predictor.prepare_training_data(
    merchants_data,
    churn_labels=churn_labels
)
```

## 技术架构

```
├── merchant_churn_prediction.py    # 核心预测类
├── merchant_analysis_example.py    # 示例演示
├── run_with_real_data.py          # 实际数据分析脚本
├── requirements.txt               # 依赖包
└── README_CHURN_PREDICTION.md    # 本文档
```

## 关键算法说明

### 自适应加权移动平均

```python
def _adaptive_weighted_ma(self, data, window):
    # 1. 根据商户频率选择窗口大小
    # 2. 提取最近window个数据点
    recent_data = data[-window:]
    
    # 3. 应用时间衰减权重
    recent_weights = self.time_weights[-window:]
    
    # 4. 计算加权平均
    weighted_sum = np.sum(recent_data * recent_weights)
    weights_sum = np.sum(recent_weights)
    
    return weighted_sum / weights_sum
```

### 时间衰减权重计算

```python
def _calculate_time_weights(self):
    weights = np.array([
        self.decay_rate ** i
        for i in range(self.observation_days)
    ])
    return weights[::-1]  # 反转，使最近的权重最大
```

## 性能优化

1. **大数据集采样**：对于超大数据集，使用采样分析
2. **并行处理**：可以使用multiprocessing并行处理商户
3. **增量更新**：定期更新模型而非每次重新训练
4. **特征缓存**：缓存已提取的特征

## 常见问题

### Q1: 如何调整观察期天数？

```python
predictor = MerchantChurnPredictor(
    observation_days=90,  # 改为90天
    decay_rate=0.95
)
```

### Q2: 如何处理数据不足的商户？

系统会自动跳过数据不足的商户（少于3天数据）。

### Q3: 模型准确率不高怎么办？

1. 增加训练样本数量
2. 使用人工标注的流失标签
3. 添加更多特征
4. 调整模型参数

### Q4: 如何解释预测结果？

查看特征重要性排名，关注：
- 连续不活跃天数
- 金额趋势
- 活跃率变化

## 许可证

本项目仅供内部使用。

## 联系方式

如有问题，请联系数据分析团队。
