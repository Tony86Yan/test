"""
商户流失预测可视化分析
提供图表和报告生成功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（根据系统选择）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("Warning: 中文字体设置失败，图表中的中文可能无法正常显示")


class ChurnVisualization:
    """流失预测可视化类"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_risk_distribution(self, predictions_df, save_path=None):
        """
        绘制风险分布图
        
        参数：
        predictions_df: 预测结果DataFrame
        save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. 风险等级分布（饼图）
        risk_counts = predictions_df['risk_level'].value_counts()
        colors = ['#ff4444', '#ffaa44', '#44ff44']
        axes[0, 0].pie(
            risk_counts.values,
            labels=risk_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        axes[0, 0].set_title('商户风险等级分布', fontsize=14, fontweight='bold')
        
        # 2. 流失概率分布（直方图）
        if 'churn_probability' in predictions_df.columns:
            axes[0, 1].hist(
                predictions_df['churn_probability'],
                bins=20,
                edgecolor='black',
                alpha=0.7
            )
            axes[0, 1].axvline(
                predictions_df['churn_probability'].mean(),
                color='red',
                linestyle='--',
                label=f'平均值: {predictions_df["churn_probability"].mean():.2%}'
            )
            axes[0, 1].set_xlabel('流失概率', fontsize=12)
            axes[0, 1].set_ylabel('商户数量', fontsize=12)
            axes[0, 1].set_title('流失概率分布', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
        
        # 3. 交易额 vs 流失概率（散点图）
        if 'churn_probability' in predictions_df.columns:
            scatter = axes[1, 0].scatter(
                predictions_df['total_amount_60d'],
                predictions_df['churn_probability'],
                c=predictions_df['churn_probability'],
                cmap='RdYlGn_r',
                alpha=0.6,
                s=100
            )
            axes[1, 0].set_xlabel('60天交易总额', fontsize=12)
            axes[1, 0].set_ylabel('流失概率', fontsize=12)
            axes[1, 0].set_title('交易额 vs 流失概率', fontsize=14, fontweight='bold')
            axes[1, 0].set_xscale('log')
            plt.colorbar(scatter, ax=axes[1, 0], label='流失概率')
        
        # 4. 风险等级 vs 交易额（箱线图）
        risk_order = ['低风险', '中风险', '高风险']
        risk_order_exists = [r for r in risk_order if r in predictions_df['risk_level'].unique()]
        
        sns.boxplot(
            data=predictions_df,
            x='risk_level',
            y='total_amount_60d',
            order=risk_order_exists,
            ax=axes[1, 1]
        )
        axes[1, 1].set_xlabel('风险等级', fontsize=12)
        axes[1, 1].set_ylabel('60天交易总额', fontsize=12)
        axes[1, 1].set_title('不同风险等级的交易额分布', fontsize=14, fontweight='bold')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"风险分布图已保存至: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance_dict, top_n=15, save_path=None):
        """
        绘制特征重要性图
        
        参数：
        feature_importance_dict: 特征重要性字典
        top_n: 显示前N个特征
        save_path: 保存路径（可选）
        """
        # 排序并选择top N
        sorted_features = sorted(
            feature_importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        features, importance = zip(*sorted_features)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importance, alpha=0.8, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('重要性分数', fontsize=12)
        ax.set_title(f'Top {top_n} 特征重要性', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for i, v in enumerate(importance):
            ax.text(v, i, f' {v:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征重要性图已保存至: {save_path}")
        
        plt.show()
    
    def plot_trend_analysis(self, predictions_df, save_path=None):
        """
        绘制趋势分析图
        
        参数：
        predictions_df: 预测结果DataFrame
        save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. 金额趋势分布
        axes[0, 0].hist(
            predictions_df['amount_trend'],
            bins=30,
            edgecolor='black',
            alpha=0.7,
            color='skyblue'
        )
        axes[0, 0].axvline(0, color='red', linestyle='--', label='零线')
        axes[0, 0].set_xlabel('金额趋势', fontsize=12)
        axes[0, 0].set_ylabel('商户数量', fontsize=12)
        axes[0, 0].set_title('金额趋势分布', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        
        # 2. 活跃率 vs 风险等级
        risk_order = ['低风险', '中风险', '高风险']
        risk_order_exists = [r for r in risk_order if r in predictions_df['risk_level'].unique()]
        
        sns.violinplot(
            data=predictions_df,
            x='risk_level',
            y='active_rate',
            order=risk_order_exists,
            ax=axes[0, 1]
        )
        axes[0, 1].set_xlabel('风险等级', fontsize=12)
        axes[0, 1].set_ylabel('活跃率', fontsize=12)
        axes[0, 1].set_title('不同风险等级的活跃率分布', fontsize=14, fontweight='bold')
        
        # 3. 连续不活跃天数 vs 风险等级
        sns.boxplot(
            data=predictions_df,
            x='risk_level',
            y='consecutive_inactive_days',
            order=risk_order_exists,
            ax=axes[1, 0]
        )
        axes[1, 0].set_xlabel('风险等级', fontsize=12)
        axes[1, 0].set_ylabel('连续不活跃天数', fontsize=12)
        axes[1, 0].set_title('不同风险等级的连续不活跃天数', fontsize=14, fontweight='bold')
        
        # 4. 金额趋势 vs 流失概率
        if 'churn_probability' in predictions_df.columns:
            scatter = axes[1, 1].scatter(
                predictions_df['amount_trend'],
                predictions_df['churn_probability'],
                c=predictions_df['churn_probability'],
                cmap='RdYlGn_r',
                alpha=0.6,
                s=100
            )
            axes[1, 1].axvline(0, color='black', linestyle='--', alpha=0.3)
            axes[1, 1].set_xlabel('金额趋势', fontsize=12)
            axes[1, 1].set_ylabel('流失概率', fontsize=12)
            axes[1, 1].set_title('金额趋势 vs 流失概率', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=axes[1, 1], label='流失概率')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"趋势分析图已保存至: {save_path}")
        
        plt.show()
    
    def plot_merchant_category_analysis(self, predictions_df, save_path=None):
        """
        绘制商户分类分析图
        
        参数：
        predictions_df: 预测结果DataFrame（需包含merchant_category_code）
        save_path: 保存路径（可选）
        """
        if 'merchant_category_code' not in predictions_df.columns:
            print("Warning: 缺少merchant_category_code列，跳过商户分类分析")
            return
        
        # 商户类型映射
        category_map = {
            5: '高频',
            4: '中高频',
            3: '中频',
            2: '低频',
            1: '极低频'
        }
        
        predictions_df['category_name'] = predictions_df['merchant_category_code'].map(category_map)
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. 商户类型分布
        category_counts = predictions_df['category_name'].value_counts()
        axes[0, 0].bar(
            range(len(category_counts)),
            category_counts.values,
            tick_label=category_counts.index,
            color='steelblue',
            alpha=0.8
        )
        axes[0, 0].set_xlabel('商户类型', fontsize=12)
        axes[0, 0].set_ylabel('商户数量', fontsize=12)
        axes[0, 0].set_title('商户类型分布', fontsize=14, fontweight='bold')
        
        # 2. 各类型商户的流失率
        if 'churn_probability' in predictions_df.columns:
            category_churn = predictions_df.groupby('category_name')['churn_probability'].mean().sort_values()
            
            axes[0, 1].barh(
                range(len(category_churn)),
                category_churn.values,
                tick_label=category_churn.index,
                color='coral',
                alpha=0.8
            )
            axes[0, 1].set_xlabel('平均流失概率', fontsize=12)
            axes[0, 1].set_ylabel('商户类型', fontsize=12)
            axes[0, 1].set_title('不同类型商户的平均流失概率', fontsize=14, fontweight='bold')
        
        # 3. 商户类型 vs 风险等级（堆叠柱状图）
        risk_by_category = pd.crosstab(
            predictions_df['category_name'],
            predictions_df['risk_level'],
            normalize='index'
        ) * 100
        
        risk_by_category.plot(
            kind='bar',
            stacked=True,
            ax=axes[1, 0],
            color=['#44ff44', '#ffaa44', '#ff4444']
        )
        axes[1, 0].set_xlabel('商户类型', fontsize=12)
        axes[1, 0].set_ylabel('百分比 (%)', fontsize=12)
        axes[1, 0].set_title('不同类型商户的风险分布', fontsize=14, fontweight='bold')
        axes[1, 0].legend(title='风险等级', loc='upper right')
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)
        
        # 4. 商户类型 vs 交易额
        category_amount = predictions_df.groupby('category_name')['total_amount_60d'].sum().sort_values(ascending=False)
        
        axes[1, 1].bar(
            range(len(category_amount)),
            category_amount.values,
            tick_label=category_amount.index,
            color='lightgreen',
            alpha=0.8
        )
        axes[1, 1].set_xlabel('商户类型', fontsize=12)
        axes[1, 1].set_ylabel('60天交易总额', fontsize=12)
        axes[1, 1].set_title('不同类型商户的交易额贡献', fontsize=14, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"商户分类分析图已保存至: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix, save_path=None):
        """
        绘制混淆矩阵
        
        参数：
        confusion_matrix: sklearn的混淆矩阵
        save_path: 保存路径（可选）
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['正常', '流失'],
            yticklabels=['正常', '流失'],
            ax=ax,
            cbar_kws={'label': '数量'}
        )
        
        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=12)
        ax.set_title('混淆矩阵', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵图已保存至: {save_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self, predictions_df, model_results=None, output_dir='/workspace/'):
        """
        生成综合报告（包含多个图表）
        
        参数：
        predictions_df: 预测结果DataFrame
        model_results: 模型评估结果字典（可选）
        output_dir: 输出目录
        """
        print("\n正在生成可视化报告...")
        
        # 1. 风险分布图
        self.plot_risk_distribution(
            predictions_df,
            save_path=f'{output_dir}risk_distribution.png'
        )
        
        # 2. 趋势分析图
        self.plot_trend_analysis(
            predictions_df,
            save_path=f'{output_dir}trend_analysis.png'
        )
        
        # 3. 商户分类分析图
        self.plot_merchant_category_analysis(
            predictions_df,
            save_path=f'{output_dir}category_analysis.png'
        )
        
        # 4. 特征重要性图（如果有模型结果）
        if model_results:
            for model_name, metrics in model_results.items():
                if 'feature_importance' in metrics:
                    self.plot_feature_importance(
                        metrics['feature_importance'],
                        top_n=15,
                        save_path=f'{output_dir}feature_importance_{model_name.replace(" ", "_")}.png'
                    )
                    break  # 只绘制一个模型的特征重要性
            
            # 5. 混淆矩阵
            for model_name, metrics in model_results.items():
                if 'confusion_matrix' in metrics:
                    self.plot_confusion_matrix(
                        metrics['confusion_matrix'],
                        save_path=f'{output_dir}confusion_matrix_{model_name.replace(" ", "_")}.png'
                    )
                    break
        
        print(f"\n所有图表已保存至: {output_dir}")


def create_summary_table(predictions_df):
    """创建汇总表格"""
    summary = {
        '指标': [],
        '数值': []
    }
    
    # 基础统计
    summary['指标'].append('总商户数')
    summary['数值'].append(len(predictions_df))
    
    # 风险分布
    for risk_level in ['高风险', '中风险', '低风险']:
        count = (predictions_df['risk_level'] == risk_level).sum()
        pct = count / len(predictions_df) * 100
        summary['指标'].append(f'{risk_level}商户数')
        summary['数值'].append(f'{count} ({pct:.1f}%)')
    
    # 交易额统计
    summary['指标'].append('60天总交易额')
    summary['数值'].append(f'{predictions_df["total_amount_60d"].sum():,.0f}')
    
    summary['指标'].append('平均每商户交易额')
    summary['数值'].append(f'{predictions_df["total_amount_60d"].mean():,.0f}')
    
    # 高风险商户统计
    high_risk = predictions_df[predictions_df['risk_level'] == '高风险']
    if len(high_risk) > 0:
        summary['指标'].append('高风险商户交易额')
        summary['数值'].append(f'{high_risk["total_amount_60d"].sum():,.0f}')
        
        summary['指标'].append('潜在流失占比')
        summary['数值'].append(
            f'{high_risk["total_amount_60d"].sum() / predictions_df["total_amount_60d"].sum() * 100:.1f}%'
        )
    
    # 流失概率统计
    if 'churn_probability' in predictions_df.columns:
        summary['指标'].append('平均流失概率')
        summary['数值'].append(f'{predictions_df["churn_probability"].mean():.1%}')
    
    return pd.DataFrame(summary)


# 使用示例
if __name__ == "__main__":
    print("="*80)
    print("商户流失预测可视化分析")
    print("="*80)
    
    # 加载预测结果
    try:
        predictions_df = pd.read_csv('/workspace/merchant_churn_predictions.csv')
        print(f"\n成功加载预测结果，共 {len(predictions_df)} 个商户")
        
        # 创建可视化对象
        viz = ChurnVisualization()
        
        # 生成汇总表格
        print("\n" + "="*80)
        print("汇总统计")
        print("="*80)
        summary_table = create_summary_table(predictions_df)
        print(summary_table.to_string(index=False))
        
        # 生成可视化报告
        viz.generate_comprehensive_report(
            predictions_df,
            output_dir='/workspace/'
        )
        
    except FileNotFoundError:
        print("\n错误：未找到预测结果文件")
        print("请先运行 merchant_analysis_example.py 或 run_with_real_data.py 生成预测结果")
