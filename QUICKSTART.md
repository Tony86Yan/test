# 商户流失预测系统 - 快速开始指南

## 🎯 一分钟快速测试

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行示例
```bash
python3 example_usage.py
```

这将：
- ✅ 自动生成50个示例商户数据（包含高频、中频、低频商户）
- ✅ 训练逻辑回归模型
- ✅ 评估模型性能（准确率、AUC-ROC等）
- ✅ 展示特征重要性分析
- ✅ 预测所有商户的流失概率
- ✅ 生成风险分级报告

### 3. 查看结果

示例输出：
```
模型评估结果：
- 准确率: 80%
- AUC-ROC: 0.96
- 成功识别流失和活跃商户

特征重要性（Top 3）：
1. 交易笔数标准差
2. 交易金额标准差  
3. 平均交易笔数

流失风险最高的商户：
- shop_id: high_churn_9
- 流失概率: 99.03%
- 趋势: 下降
- 频率: 高频
```

## 📊 使用真实数据

### 修改 example_usage.py

取消注释以下代码段（第55-75行）：

```python
# 方式2: 从CSV文件加载真实数据（取消注释以使用）
print("\n方式2: 从CSV文件加载真实数据")
print("-" * 80)
csv_path = "mer_trans_data.csv"
shop_dict, min_date, max_date = load_real_data(csv_path)

# 定义流失标签（最近30天无交易视为流失）
churn_labels = {}
for shop_id, df in shop_dict.items():
    df_sorted = df.sort_values('date')
    if len(df_sorted) > 0:
        last_date = pd.to_datetime(df_sorted['date'].iloc[-1])
        days_since_last = (pd.to_datetime(max_date) - last_date).days
        churn_labels[shop_id] = 1 if days_since_last > 30 else 0
```

### CSV数据格式

您的CSV文件应包含以下列（使用 `@@@` 作为分隔符）：

```
merchant_id@@@merchant_name@@@_shop_id@@@_shop_name@@@check_date@@@shop_create_date@@@pt@@@amount@@@number
```

其中：
- `_shop_id`: 商户ID
- `pt`: 交易日期（YYYY-MM-DD）
- `amount`: 交易金额
- `number`: 交易笔数

## 🔧 核心API使用

### 基本流程

```python
from merchant_churn_predictor import MerchantChurnPredictor

# 1. 初始化
predictor = MerchantChurnPredictor(
    min_date='2025-07-03', 
    max_date='2025-10-30'
)

# 2. 准备数据
X_train, X_test, y_train, y_test, feature_names = \
    predictor.prepare_training_data(shop_dict, churn_labels)

# 3. 训练模型
predictor.train(X_train, y_train)

# 4. 评估模型
predictor.evaluate(X_test, y_test)

# 5. 预测单个商户
churn_prob, trend, freq_type, features = \
    predictor.predict_churn(shop_dict['shop_001'])

print(f"流失概率: {churn_prob:.2%}")
print(f"交易趋势: {trend}")  # increasing/decreasing/stable

# 6. 批量预测
results = predictor.batch_predict(shop_dict, top_n=20)
print(results[['shop_id', 'churn_probability', 'trend']])
```

## 📈 输出说明

### 交易趋势类型
- **increasing**: 交易量上升，商户成长中
- **decreasing**: 交易量下降，流失风险高
- **stable**: 交易量稳定

### 频率类型
- **high**: 高频商户（每周≥5天有交易，如超市）
- **medium**: 中频商户（每周1-5天有交易）
- **low**: 低频商户（每周<1天有交易）

### 流失概率
- **0-30%**: 低风险，保持正常服务
- **30-60%**: 中风险，需要定期回访
- **60-100%**: 高风险，需要立即干预

## 💡 业务应用建议

### 1. 定期批量预测（推荐每周一次）
```python
# 获取流失风险最高的商户
high_risk_merchants = results[results['churn_probability'] > 0.6]

# 导出给客户经理
high_risk_merchants.to_csv('weekly_high_risk_merchants.csv', index=False)
```

### 2. 实时监控（每日检查新增高风险）
```python
# 预测所有商户
all_results = predictor.batch_predict(shop_dict)

# 筛选新增高风险商户
new_high_risk = all_results[
    (all_results['churn_probability'] > 0.6) & 
    (all_results['days_since_last_transaction'] > 7)
]

# 发送预警通知
send_alert(new_high_risk)
```

### 3. 效果追踪
```python
# 记录干预前的流失概率
before_intervention = predictor.predict_churn(shop_dict['shop_001'])[0]

# 进行商户干预（优惠、客服跟进等）
# ... 

# 30天后重新预测
after_intervention = predictor.predict_churn(shop_dict['shop_001'])[0]

# 计算改善效果
improvement = before_intervention - after_intervention
print(f"流失概率改善: {improvement:.2%}")
```

## 📚 进一步学习

详细文档请参考：
- 📖 [完整文档](README_CHURN_PREDICTION.md) - 算法原理、特征说明、常见问题
- 💻 [核心代码](merchant_churn_predictor.py) - MerchantChurnPredictor类实现
- 🔬 [示例代码](example_usage.py) - 完整使用示例

## ⚠️ 注意事项

1. **数据质量**: 确保交易数据完整、准确
2. **流失定义**: 根据业务实际情况调整流失标签定义
3. **模型更新**: 定期（如每月）使用新数据重新训练模型
4. **阈值调整**: 根据干预成本和效果调整风险阈值
5. **新商户**: 至少观察30天后再进行预测

## 🆘 遇到问题？

### 常见问题排查

**Q: ImportError: No module named 'pandas'**  
A: 运行 `pip install -r requirements.txt`

**Q: 所有商户预测概率都很接近**  
A: 检查流失标签是否正确定义，增加数据多样性

**Q: 模型准确率低**  
A: 
1. 增加训练数据量（至少50个商户）
2. 调整流失定义标准
3. 增加更多特征
4. 尝试其他算法（随机森林、XGBoost等）

---

**开始使用**: `python3 example_usage.py`  
**完整文档**: [README_CHURN_PREDICTION.md](README_CHURN_PREDICTION.md)
