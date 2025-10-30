# 商户流失预测系统

基于机器学习的商户流失预测解决方案，使用逻辑回归识别商户交易趋势并预测流失风险。

## 📋 项目背景

第三方支付公司为商户提供一站式支付方案。不同商户有不同的交易模式：
- **高频商户**（如超市）：每天交易笔数和金额都很稳定
- **中频商户**：每周交易笔数和金额都很稳定
- **低频商户**：每月交易笔数和金额都很稳定

本系统通过分析商户的历史交易数据，识别交易趋势并预测商户是否存在流失风险。

## 🎯 核心功能

### 1. 商户频率分类
- 自动识别商户的交易频率（高频、中频、低频）
- 基于活跃天数占比和平均每周活跃天数进行分类

### 2. 数据预处理与平滑
- **高频商户**：使用7天移动平均窗口
- **中频和低频商户**：使用30天移动平均窗口
- 计算移动标准差以识别波动性

### 3. 趋势识别
使用线性回归和多项式回归识别三种趋势：
- **上升趋势**（increasing）：交易量逐渐增长
- **下降趋势**（decreasing）：交易量逐渐减少，流失风险高
- **平稳趋势**（stable）：交易量保持稳定

### 4. 特征工程
提取17个关键特征用于流失预测：

#### 基础统计特征
- 平均交易金额
- 平均交易笔数
- 交易金额标准差
- 交易笔数标准差

#### 趋势特征
- 金额趋势斜率（线性回归系数）
- 金额趋势拟合优度（R²）
- 交易笔数趋势斜率
- 交易笔数趋势拟合优度

#### 衰减特征
- 金额衰减比率（最近一周 vs 前三周）
- 交易笔数衰减比率

#### 活跃度特征
- 活跃天数占比
- 零交易天数
- 距上次交易天数

#### 波动性特征
- 金额变异系数
- 交易笔数变异系数

#### 非线性趋势特征
- 金额二次项系数（检测加速/减速变化）
- 交易笔数二次项系数

### 5. 逻辑回归模型
- 使用标准化特征训练逻辑回归模型
- 自动处理类别不平衡（class_weight='balanced'）
- 输出流失概率（0-1之间）

### 6. 批量预测与风险分级
- 支持批量预测所有商户的流失概率
- 自动风险分级：
  - **低风险**：流失概率 < 30%
  - **中风险**：流失概率 30%-60%
  - **高风险**：流失概率 > 60%

## 🚀 快速开始

### 安装依赖

```bash
pip install pandas numpy scikit-learn
```

### 使用示例

#### 方式1: 使用示例数据（快速测试）

```python
from merchant_churn_predictor import MerchantChurnPredictor
import pandas as pd

# 生成或加载数据
# shop_dict = {shop_id: DataFrame(['date', 'transaction_count', 'amount'])}
# churn_labels = {shop_id: 0/1}

# 初始化预测器
predictor = MerchantChurnPredictor(min_date='2025-07-03', max_date='2025-10-30')

# 准备训练数据
X_train, X_test, y_train, y_test, feature_names = predictor.prepare_training_data(
    shop_dict, churn_labels
)

# 训练模型
predictor.train(X_train, y_train)

# 评估模型
y_pred, y_pred_proba = predictor.evaluate(X_test, y_test)

# 预测单个商户
churn_prob, trend, freq_type, features = predictor.predict_churn(
    shop_dict['shop_001']
)
print(f"流失概率: {churn_prob:.2%}")
print(f"交易趋势: {trend}")
print(f"频率类型: {freq_type}")

# 批量预测
results = predictor.batch_predict(shop_dict, top_n=20)
print(results)
```

#### 方式2: 使用真实CSV数据

```python
# 读取数据
mer_df = pd.read_csv("mer_trans_data.csv", sep="@@@")

# 数据处理
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

# 构建商户字典
shop_dict = {}
for row in shop_df.values:
    key = row[0]  # _shop_id
    if key not in shop_dict.keys():
        shop_dict[key] = []
    shop_dict[key].append([row[1], row[3], row[2]])

shop_dict = {
    k: pd.DataFrame(v, columns=['date', 'transaction_count', 'amount']) 
    for k, v in shop_dict.items()
}

# 定义流失标签（示例：最近30天无交易视为流失）
churn_labels = {}
for shop_id, df in shop_dict.items():
    df_sorted = df.sort_values('date')
    if len(df_sorted) > 0:
        last_date = pd.to_datetime(df_sorted['date'].iloc[-1])
        days_since_last = (pd.to_datetime(max_pt) - last_date).days
        churn_labels[shop_id] = 1 if days_since_last > 30 else 0

# 使用预测器
predictor = MerchantChurnPredictor(min_pt, max_pt)
# ... 后续步骤同方式1
```

### 运行示例代码

```bash
python example_usage.py
```

该脚本会：
1. 生成示例商户数据（包含不同频率和流失状态的商户）
2. 训练逻辑回归模型
3. 评估模型性能
4. 展示特征重要性
5. 批量预测流失风险
6. 生成风险分级报告

## 📊 数据格式

### 输入数据格式

每个商户的交易数据应为DataFrame，包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| date | str/datetime | 交易日期（格式：YYYY-MM-DD） |
| transaction_count | int | 交易笔数 |
| amount | float | 交易金额 |

示例：
```python
shop_data = pd.DataFrame({
    'date': ['2025-10-01', '2025-10-02', '2025-10-03'],
    'transaction_count': [100, 95, 110],
    'amount': [50000, 48000, 52000]
})
```

### 输出结果格式

批量预测结果DataFrame包含以下列：

| 列名 | 说明 |
|------|------|
| shop_id | 商户ID |
| churn_probability | 流失概率（0-1） |
| trend | 交易趋势（increasing/decreasing/stable） |
| frequency_type | 频率类型（high/medium/low） |
| avg_amount | 平均交易金额 |
| avg_transaction_count | 平均交易笔数 |
| days_since_last_transaction | 距上次交易天数 |
| active_days_ratio | 活跃天数占比 |

## 🔧 核心算法

### 1. 频率分类算法

```python
active_ratio = 有交易天数 / 总天数
avg_weekly_active_days = active_ratio * 7

if avg_weekly_active_days >= 5:
    频率 = 高频  # 每周至少5天有交易
elif avg_weekly_active_days >= 1:
    频率 = 中频  # 每周至少1天有交易
else:
    频率 = 低频  # 每周少于1天交易
```

### 2. 移动平均平滑

```python
if 频率 == 高频:
    窗口大小 = 7天
else:
    窗口大小 = 30天

平滑后的值 = 原始值.rolling(window=窗口大小).mean()
```

### 3. 趋势识别算法

使用线性回归拟合最近N天的移动平均值：

```python
X = [0, 1, 2, ..., N-1]  # 时间索引
y = 移动平均值

斜率, R² = 线性回归(X, y)
归一化斜率 = 斜率 / 平均值

if 归一化斜率 > 0.05:
    趋势 = 上升
elif 归一化斜率 < -0.05:
    趋势 = 下降
else:
    趋势 = 平稳
```

### 4. 逻辑回归模型

```python
模型输入: 17个特征 (标准化后)
模型输出: P(流失) = 1 / (1 + e^(-z))

其中: z = w₀ + w₁x₁ + w₂x₂ + ... + w₁₇x₁₇
```

## 📈 模型评估指标

- **准确率（Accuracy）**：整体预测正确的比例
- **精确率（Precision）**：预测为流失的商户中，真正流失的比例
- **召回率（Recall）**：实际流失的商户中，被正确预测的比例
- **F1分数**：精确率和召回率的调和平均
- **AUC-ROC**：模型区分流失和非流失商户的能力

## 💡 使用建议

### 1. 流失标签定义

根据业务需求定义流失，常见方法：
- **时间阈值法**：最近N天无交易视为流失
- **交易量阈值法**：交易量下降超过X%视为流失
- **综合判断法**：结合多个指标综合判断

示例：
```python
# 方法1: 30天无交易视为流失
churn_labels = {}
for shop_id, df in shop_dict.items():
    last_date = pd.to_datetime(df['date'].max())
    days_since_last = (pd.to_datetime(max_date) - last_date).days
    churn_labels[shop_id] = 1 if days_since_last > 30 else 0

# 方法2: 交易量下降超过50%视为流失
churn_labels = {}
for shop_id, df in shop_dict.items():
    recent_avg = df.tail(30)['amount'].mean()
    previous_avg = df.iloc[-60:-30]['amount'].mean()
    decay_ratio = recent_avg / (previous_avg + 1e-6)
    churn_labels[shop_id] = 1 if decay_ratio < 0.5 else 0
```

### 2. 模型调优

- **特征选择**：根据特征重要性选择最相关的特征
- **超参数调优**：调整C值（正则化强度）
- **阈值优化**：根据业务成本调整分类阈值
- **处理不平衡**：使用SMOTE等方法平衡样本

### 3. 实时监控

建议建立监控系统：
- 每日更新商户交易数据
- 每周批量预测流失概率
- 对高风险商户发出预警
- 追踪干预效果

### 4. 业务干预策略

根据流失概率和趋势制定干预策略：

| 流失概率 | 趋势 | 建议措施 |
|---------|------|---------|
| 高 (>60%) | 下降 | 立即联系，提供优惠或解决问题 |
| 高 (>60%) | 平稳 | 了解需求变化，提供增值服务 |
| 中 (30-60%) | 下降 | 主动沟通，了解业务情况 |
| 中 (30-60%) | 平稳 | 定期回访，维护关系 |
| 低 (<30%) | 上升 | 鼓励继续，提供更多支持 |
| 低 (<30%) | 平稳 | 保持正常服务 |

## 🔍 高级功能

### 1. 处理非线性趋势

模型已包含二次项系数特征，可识别加速/减速变化：
- 正的二次系数：加速增长或减速下降
- 负的二次系数：减速增长或加速下降

如需更复杂的非线性建模，可考虑：
- 多项式回归
- 决策树/随机森林
- 梯度提升树（XGBoost, LightGBM）
- 神经网络

### 2. 时间序列特征

可进一步增加时间序列特征：
- 季节性分析（周、月、季度模式）
- 自相关系数
- 指数移动平均（EMA）
- ARIMA残差

### 3. 外部特征

结合业务场景，可添加：
- 商户类型（行业分类）
- 地理位置
- 费率信息
- 客户服务记录
- 竞争对手动态

## 🐛 常见问题

### Q1: 如何处理缺失数据？
A: 系统会自动用0填充缺失的日期，确保时间序列连续。

### Q2: 模型准确率不高怎么办？
A: 
1. 检查流失标签定义是否合理
2. 增加更多特征
3. 收集更多历史数据
4. 尝试其他机器学习算法

### Q3: 如何处理新商户？
A: 新商户缺乏历史数据，建议：
1. 至少观察30天后再预测
2. 使用简单规则（如X天无交易预警）
3. 基于商户类型使用群体模型

### Q4: 预测结果如何应用到业务？
A: 
1. 建立自动化预警系统
2. 与CRM系统集成
3. 为客户经理提供每周风险报告
4. A/B测试不同干预策略的效果

## 📝 文件说明

- `merchant_churn_predictor.py`: 核心预测模块
- `example_usage.py`: 示例代码和演示
- `README_CHURN_PREDICTION.md`: 完整文档（本文件）

## 🚀 扩展方向

1. **实时预测服务**：部署为API服务，支持在线预测
2. **可视化面板**：构建Dashboard展示商户健康度
3. **A/B测试框架**：评估不同干预策略的效果
4. **深度学习模型**：使用LSTM等模型捕捉长期依赖
5. **多维度分析**：整合多个数据源进行综合分析

## 📧 技术支持

如有问题或建议，欢迎反馈！

---

**版本**: 1.0  
**更新日期**: 2025-10-30
