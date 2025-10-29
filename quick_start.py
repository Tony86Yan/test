"""
快速开始脚本
一键运行完整的商户流失预测分析
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def print_banner():
    """打印欢迎横幅"""
    print("\n" + "="*80)
    print(" " * 20 + "商户流失预测系统 v1.0")
    print("="*80)
    print("\n功能特性:")
    print("  ✓ 自动识别不同频率商户（高频、中频、低频）")
    print("  ✓ 自适应调整Moving Average窗口")
    print("  ✓ 多模型预测（Logistic Regression, Random Forest, XGBoost）")
    print("  ✓ 30+特征自动提取")
    print("  ✓ 风险分级（高、中、低）")
    print("  ✓ 可视化分析报告")
    print("="*80 + "\n")


def check_dependencies():
    """检查依赖包"""
    print("正在检查依赖包...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    optional_packages = {
        'xgboost': 'xgboost'
    }
    
    missing = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (缺失)")
            missing.append(package)
    
    for module, package in optional_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ⚠ {package} (可选，建议安装)")
    
    if missing:
        print(f"\n错误：缺少必需的依赖包: {', '.join(missing)}")
        print("\n请运行以下命令安装:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n依赖检查通过！\n")
    return True


def select_mode():
    """选择运行模式"""
    print("请选择运行模式:")
    print("  1. 使用示例数据（演示功能）")
    print("  2. 使用实际数据（需要提供数据文件路径）")
    print("  3. 仅查看已有预测结果（需要先运行过预测）")
    print("  4. 退出")
    
    while True:
        choice = input("\n请输入选项 (1-4): ").strip()
        
        if choice in ['1', '2', '3', '4']:
            return int(choice)
        else:
            print("无效选项，请重新输入")


def run_example_mode():
    """运行示例模式"""
    print("\n" + "="*80)
    print("模式1：使用示例数据演示")
    print("="*80)
    
    try:
        from merchant_analysis_example import main
        print("\n正在运行示例分析...")
        main()
        
        print("\n示例运行完成！")
        return True
        
    except Exception as e:
        print(f"\n错误：运行示例时出错 - {str(e)}")
        return False


def run_real_data_mode():
    """运行实际数据模式"""
    print("\n" + "="*80)
    print("模式2：使用实际数据分析")
    print("="*80)
    
    # 获取数据文件路径
    print("\n请输入数据文件路径:")
    print("（格式要求：CSV文件，分隔符为@@@）")
    print("（列名：merchant_id, merchant_name, _shop_id, _shop_name, check_date, shop_create_date, pt, amount, number）")
    
    file_path = input("\n文件路径: ").strip().strip('"').strip("'")
    
    if not os.path.exists(file_path):
        print(f"\n错误：文件不存在 - {file_path}")
        return False
    
    # 询问是否采样
    print("\n是否对数据进行采样？")
    print("（如果商户数量很大，建议采样以加快分析速度）")
    
    sample_choice = input("是否采样? (y/n): ").strip().lower()
    
    sample_size = None
    if sample_choice == 'y':
        while True:
            try:
                sample_size = int(input("请输入采样数量（如：1000）: ").strip())
                if sample_size > 0:
                    break
                else:
                    print("请输入正整数")
            except ValueError:
                print("输入无效，请输入数字")
    
    # 运行分析
    try:
        from run_with_real_data import load_and_prepare_data, analyze_merchants
        
        # 设置全局变量
        import merchant_churn_prediction
        
        # 加载数据
        shop_dict = load_and_prepare_data(
            file_path=file_path,
            sep="@@@",
            min_date='2025-07-03'
        )
        
        # 同步日期范围
        merchant_churn_prediction.min_pt = min(
            pd.DataFrame(v)['date'].min() for v in shop_dict.values()
        )
        merchant_churn_prediction.max_pt = max(
            pd.DataFrame(v)['date'].max() for v in shop_dict.values()
        )
        
        # 分析
        predictor, predictions_df = analyze_merchants(
            shop_dict,
            sample_size=sample_size,
            train_model=True
        )
        
        # 询问是否生成可视化报告
        viz_choice = input("\n是否生成可视化报告? (y/n): ").strip().lower()
        
        if viz_choice == 'y':
            from visualization_analysis import ChurnVisualization
            
            viz = ChurnVisualization()
            viz.generate_comprehensive_report(
                predictions_df,
                output_dir='/workspace/'
            )
        
        print("\n实际数据分析完成！")
        return True
        
    except Exception as e:
        print(f"\n错误：分析数据时出错 - {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def view_results_mode():
    """查看已有结果模式"""
    print("\n" + "="*80)
    print("模式3：查看已有预测结果")
    print("="*80)
    
    result_file = '/workspace/merchant_churn_predictions.csv'
    
    if not os.path.exists(result_file):
        print(f"\n错误：未找到预测结果文件")
        print("请先运行模式1或模式2生成预测结果")
        return False
    
    try:
        predictions_df = pd.read_csv(result_file)
        
        print(f"\n成功加载预测结果，共 {len(predictions_df)} 个商户")
        
        # 显示基本统计
        from visualization_analysis import create_summary_table
        
        summary_table = create_summary_table(predictions_df)
        
        print("\n" + "="*80)
        print("汇总统计")
        print("="*80)
        print(summary_table.to_string(index=False))
        
        # 显示高风险商户
        print("\n" + "="*80)
        print("高风险商户（Top 10）")
        print("="*80)
        
        high_risk = predictions_df[
            predictions_df['risk_level'] == '高风险'
        ].head(10)
        
        if len(high_risk) > 0:
            display_cols = [
                'merchant_id', 'total_amount_60d', 'active_rate',
                'amount_trend', 'consecutive_inactive_days'
            ]
            
            if 'churn_probability' in high_risk.columns:
                display_cols.insert(1, 'churn_probability')
            
            display_df = high_risk[
                [col for col in display_cols if col in high_risk.columns]
            ].copy()
            
            print(display_df.to_string(index=False))
        else:
            print("未发现高风险商户")
        
        # 询问是否生成可视化
        viz_choice = input("\n是否生成可视化报告? (y/n): ").strip().lower()
        
        if viz_choice == 'y':
            from visualization_analysis import ChurnVisualization
            
            viz = ChurnVisualization()
            viz.generate_comprehensive_report(
                predictions_df,
                output_dir='/workspace/'
            )
        
        return True
        
    except Exception as e:
        print(f"\n错误：查看结果时出错 - {str(e)}")
        return False


def main():
    """主函数"""
    print_banner()
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 选择模式
    mode = select_mode()
    
    if mode == 1:
        run_example_mode()
    elif mode == 2:
        run_real_data_mode()
    elif mode == 3:
        view_results_mode()
    elif mode == 4:
        print("\n再见！")
        return
    
    # 询问是否继续
    while True:
        continue_choice = input("\n是否继续使用其他功能? (y/n): ").strip().lower()
        
        if continue_choice == 'y':
            mode = select_mode()
            
            if mode == 1:
                run_example_mode()
            elif mode == 2:
                run_real_data_mode()
            elif mode == 3:
                view_results_mode()
            elif mode == 4:
                break
        else:
            break
    
    print("\n" + "="*80)
    print("感谢使用商户流失预测系统！")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n程序出错: {str(e)}")
        import traceback
        traceback.print_exc()
