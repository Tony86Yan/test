# 文件索引

## 核心代码文件

### 1. `merchant_churn_prediction.py`
**功能**: 核心预测类  
**大小**: ~500行  
**主要内容**:
- `MerchantChurnPredictor` 类
- 自适应窗口Moving Average实现
- 时间衰减权重计算
- 特征提取（30+特征）
- Logistic Regression、Random Forest、XGBoost模型
- 批量预测功能

**关键方法**:
```python
- __init__(): 初始化预测器
- extract_features(): 提取30+特征
- classify_merchant(): 商户分类
- train_models(): 训练3个ML模型
- predict_churn(): 预测单个商户
- batch_predict(): 批量预测
```

---

### 2. `merchant_analysis_example.py`
**功能**: 完整示例演示  
**大小**: ~300行  
**主要内容**:
- 生成模拟商户数据
- 完整分析流程演示
- 模型评估和对比
- 高风险商户详细分析
- 业务建议生成

**使用场景**: 学习系统功能、测试代码

**运行**:
```bash
python3 merchant_analysis_example.py
```

---

### 3. `run_with_real_data.py`
**功能**: 实际数据分析脚本  
**大小**: ~250行  
**主要内容**:
- 加载CSV数据（分隔符@@@）
- 数据预处理和转换
- 大数据集采样支持
- 完整分析流程
- 结果保存和导出

**使用场景**: 分析真实商户数据

**使用方法**:
1. 修改`DATA_FILE`变量为实际路径
2. 运行: `python3 run_with_real_data.py`

---

### 4. `visualization_analysis.py`
**功能**: 可视化分析  
**大小**: ~350行  
**主要内容**:
- `ChurnVisualization` 类
- 风险分布图
- 趋势分析图
- 商户分类分析图
- 特征重要性图
- 混淆矩阵图
- 综合报告生成

**生成的图表**:
- `risk_distribution.png`
- `trend_analysis.png`
- `category_analysis.png`
- `feature_importance_*.png`
- `confusion_matrix_*.png`

**运行**:
```bash
python3 visualization_analysis.py
```

---

### 5. `quick_start.py`
**功能**: 交互式一键启动  
**大小**: ~200行  
**主要内容**:
- 依赖检查
- 交互式菜单
- 三种运行模式
  1. 示例数据演示
  2. 实际数据分析
  3. 查看已有结果

**使用场景**: 新手快速上手

**运行**:
```bash
python3 quick_start.py
```

---

## 配置文件

### 6. `requirements.txt`
**功能**: Python依赖列表  
**内容**:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0  # 可选
```

**安装**:
```bash
pip install -r requirements.txt
```

---

## 文档文件

### 7. `README.md`
**功能**: 项目主文档  
**内容**:
- 快速开始指南
- 核心特性介绍
- 代码示例
- 参数调整
- 常见问题

**适合**: 所有用户，首先阅读

---

### 8. `USAGE_GUIDE.md`
**功能**: 详细使用指南  
**内容**:
- 3分钟快速开始
- 核心功能详解
- 使用实际数据步骤
- 特征工程说明
- 流失判断标准
- 可视化分析
- 参数调优指南

**适合**: 需要深入了解功能的用户

---

### 9. `README_CHURN_PREDICTION.md`
**功能**: 完整技术文档  
**内容**:
- 项目概述
- 核心特性详解
- 自适应窗口设计
- 时间衰减权重
- 机器学习模型
- 特征工程（30+特征）
- 流失判断标准
- 模型优化建议
- 技术架构
- 关键算法说明

**适合**: 技术人员、需要定制化的用户

---

### 10. `PROJECT_SUMMARY.md`
**功能**: 项目总结和设计思路  
**内容**:
- 项目背景
- 核心创新点
- 技术架构
- 核心算法
- 实验结果
- 业务价值
- 对比传统方法
- 适用场景
- 后续优化方向

**适合**: 管理层、技术决策者

---

### 11. `FILE_INDEX.md`
**功能**: 本文件，文件索引  
**内容**: 所有文件的功能说明和使用方法

---

## 输出文件（运行后生成）

### 12. `merchant_churn_predictions.csv`
**位置**: `/workspace/`  
**生成**: 运行预测后自动生成  
**内容**: 商户流失预测结果
- merchant_id
- churn_probability
- risk_level
- total_amount_60d
- active_rate
- amount_trend
- consecutive_inactive_days
- 等30+列

---

### 13. 图表文件
**位置**: `/workspace/`  
**生成**: 运行可视化分析后生成

| 文件名 | 说明 |
|--------|------|
| `risk_distribution.png` | 风险分布（饼图、直方图、散点图） |
| `trend_analysis.png` | 趋势分析（金额趋势、活跃率等） |
| `category_analysis.png` | 商户分类分析 |
| `feature_importance_Logistic_Regression.png` | LR特征重要性 |
| `feature_importance_Random_Forest.png` | RF特征重要性 |
| `feature_importance_XGBoost.png` | XGB特征重要性 |
| `confusion_matrix_*.png` | 混淆矩阵 |

---

## 文件依赖关系

```
quick_start.py
    ├── merchant_analysis_example.py
    │       └── merchant_churn_prediction.py
    ├── run_with_real_data.py
    │       └── merchant_churn_prediction.py
    └── visualization_analysis.py

merchant_analysis_example.py
    └── merchant_churn_prediction.py

run_with_real_data.py
    ├── merchant_churn_prediction.py
    └── visualization_analysis.py (可选)

visualization_analysis.py
    └── (独立，可单独运行)
```

---

## 使用流程图

```
开始
  │
  ├─→ 新手用户
  │     └─→ python3 quick_start.py
  │           └─→ 选择模式1（示例数据）
  │
  ├─→ 学习功能
  │     └─→ python3 merchant_analysis_example.py
  │
  ├─→ 使用实际数据
  │     └─→ 修改 run_with_real_data.py
  │           └─→ python3 run_with_real_data.py
  │
  └─→ 查看结果
        └─→ python3 visualization_analysis.py
```

---

## 推荐阅读顺序

### 新手用户
1. `README.md` - 快速了解项目
2. `quick_start.py` - 一键运行体验
3. `USAGE_GUIDE.md` - 学习具体用法

### 技术用户
1. `README.md` - 快速了解项目
2. `README_CHURN_PREDICTION.md` - 深入技术细节
3. `merchant_churn_prediction.py` - 阅读源码

### 管理层/决策者
1. `README.md` - 快速了解项目
2. `PROJECT_SUMMARY.md` - 了解价值和创新
3. 运行示例查看效果

---

## 文件大小统计

| 文件 | 代码行数 | 说明 |
|------|---------|------|
| merchant_churn_prediction.py | ~500 | 核心类 |
| merchant_analysis_example.py | ~300 | 示例 |
| run_with_real_data.py | ~250 | 实际数据 |
| visualization_analysis.py | ~350 | 可视化 |
| quick_start.py | ~200 | 启动脚本 |
| **总计** | **~1,600** | **不含注释** |

---

## 快速命令参考

```bash
# 安装依赖
pip install -r requirements.txt

# 一键启动（推荐）
python3 quick_start.py

# 运行示例
python3 merchant_analysis_example.py

# 运行实际数据分析（需先修改文件路径）
python3 run_with_real_data.py

# 生成可视化报告（需先有预测结果）
python3 visualization_analysis.py

# 检查Python版本
python3 --version

# 检查依赖
python3 -c "import pandas, numpy, sklearn; print('OK')"
```

---

## 核心概念速查

| 概念 | 说明 | 代码位置 |
|------|------|---------|
| 自适应窗口 | 根据商户频率调整MA窗口 | `merchant_churn_prediction.py:150-200` |
| 时间衰减权重 | 0.95^i指数衰减 | `merchant_churn_prediction.py:50-60` |
| 商户分类 | 5级分类（高/中高/中/低/极低频） | `merchant_churn_prediction.py:40-90` |
| 特征提取 | 30+特征 | `merchant_churn_prediction.py:200-350` |
| 流失判断 | 基于规则的自动标注 | `merchant_churn_prediction.py:400-420` |
| 模型训练 | LR+RF+XGB | `merchant_churn_prediction.py:450-550` |
| 集成预测 | 均值投票 | `merchant_churn_prediction.py:560-600` |

---

## 数据格式速查

### 输入格式
```csv
merchant_id@@@merchant_name@@@_shop_id@@@pt@@@amount@@@number
M001@@@超市A@@@S001@@@2025-10-01@@@5000.0@@@100
```

### 输出格式
```csv
merchant_id,churn_probability,risk_level,total_amount_60d
M001,0.85,高风险,300000.0
```

---

**更新日期**: 2025-10-29  
**版本**: 1.0
