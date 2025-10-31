# -*- coding: utf-8 -*-
"""
?????????????????????
"""

import numpy as np
import pandas as pd
import os
import sys

# ?????????UTF-8???????
if sys.version_info >= (3, 7):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

from timeseries_trend_analysis_extended import (
    calculate_churn_features,
    TimeSeriesTrendAnalyzer,
    _preprocess_data
)


def create_sample_data():
    """??????"""
    np.random.seed(42)
    
    # ??120??????4???
    dates = pd.date_range(start='2025-07-03', periods=120, freq='D')
    
    # ??1: ???????
    shop1_amounts = [5000 + 50 * i + np.random.normal(0, 300) for i in range(len(dates))]
    shop1_amounts = [max(0, a) for a in shop1_amounts]
    
    shop1_df = pd.DataFrame({
        'merchant_id': ['BC00001234'] * len(dates),
        'merchant_name': ['?????'] * len(dates),
        '_shop_id': ['BC00001234001'] * len(dates),
        '_shop_name': ['????'] * len(dates),
        'check_date': ['2025-01-01'] * len(dates),
        'shop_create_date': ['2025-01-01'] * len(dates),
        'date': dates,
        'amount': shop1_amounts,
        'transaction_count': np.random.randint(20, 50, len(dates))
    })
    
    # ??2: ???????
    shop2_amounts = [15000 - 100 * i + np.random.normal(0, 500) for i in range(len(dates))]
    shop2_amounts = [max(0, a) for a in shop2_amounts]
    
    shop2_df = pd.DataFrame({
        'merchant_id': ['BC00005678'] * len(dates),
        'merchant_name': ['?????'] * len(dates),
        '_shop_id': ['BC00005678001'] * len(dates),
        '_shop_name': ['????'] * len(dates),
        'check_date': ['2025-01-01'] * len(dates),
        'shop_create_date': ['2025-01-01'] * len(dates),
        'date': dates,
        'amount': shop2_amounts,
        'transaction_count': np.random.randint(5, 30, len(dates))
    })
    
    # ??3: ?????
    shop3_amounts = [8000 + np.random.normal(0, 800) for _ in range(len(dates))]
    shop3_amounts = [max(0, a) for a in shop3_amounts]
    
    shop3_df = pd.DataFrame({
        'merchant_id': ['BC00009999'] * len(dates),
        'merchant_name': ['?????'] * len(dates),
        '_shop_id': ['BC00009999001'] * len(dates),
        '_shop_name': ['????'] * len(dates),
        'check_date': ['2025-01-01'] * len(dates),
        'shop_create_date': ['2025-01-01'] * len(dates),
        'date': dates,
        'amount': shop3_amounts,
        'transaction_count': np.random.randint(15, 35, len(dates))
    })
    
    # ??????
    all_data = pd.concat([shop1_df, shop2_df, shop3_df], ignore_index=True)
    return all_data


def analyze_single_shop(mer_df, shop_id, min_pt, max_pt, churn_features_df):
    """??????"""
    print(f"\n{'=' * 80}")
    print(f"????: {shop_id}")
    print(f"{'=' * 80}\n")
    
    # ??????
    shop_data = mer_df[mer_df['_shop_id'] == shop_id].copy()
    
    if len(shop_data) == 0:
        print(f"??: ?? {shop_id} ????")
        return
    
    # ???????
    shop_features_row = churn_features_df[churn_features_df['shop_id'] == shop_id]
    
    if len(shop_features_row) == 0:
        print(f"??: ?? {shop_id} ???????")
        shop_features = None
    else:
        shop_features = shop_features_row.iloc[0].to_dict()
        
        # ??????
        print("??????:")
        print("-" * 80)
        print(f"??????: {shop_features.get('last_m_before_last_tx_amount', 0):,.2f}")
        print(f"??????: {shop_features.get('last_m_tx_amount', 0):,.2f}")
        print(f"????????: {shop_features.get('month_tx_amount_decay_score', 0):.4f}")
        print(f"??????: {shop_features.get('last_m_before_last_active_score', 0):.4f}")
        print(f"??????: {shop_features.get('last_m_active_score', 0):.4f}")
        print(f"?????: {shop_features.get('suspected_churn_rate', 0):.4f}")
        print()
    
    # ?????
    processed_data = _preprocess_data(shop_data, min_pt, max_pt)
    
    # ?????
    analyzer = TimeSeriesTrendAnalyzer(
        dates=processed_data['date'],
        amounts=processed_data['amount'],
        df=processed_data,
        churn_features=shop_features
    )
    
    # ??????
    result = analyzer.comprehensive_analysis()
    
    # ????????
    print("???????:")
    print("-" * 80)
    
    for r in result['individual_results']:
        print(f"\n{r['method']}:")
        print(f"  ??: {r['trend']}")
        print(f"  ??: {r['degree']}")
        
        # ????????
        if 'churn_rate' in r and r['churn_rate'] is not None:
            print(f"  ???: {r['churn_rate']:.4f}")
            print(f"  ????: {r.get('risk_level', 'N/A')}")
        
        if 'decay_score' in r and r['decay_score'] is not None:
            print(f"  ????: {r['decay_score']:.4f}")
            print(f"  ????: {r.get('business_status', 'N/A')}")
        
        if 'r2' in r:
            print(f"  R?: {r['r2']:.4f}")
        
        if 'p_value' in r:
            print(f"  P?: {r['p_value']:.4f}")
    
    # ??????
    print("\n" + "=" * 80)
    print("????:")
    print("=" * 80)
    print(f"????: {result['consensus_trend']}")
    print(f"????: {result['consensus_degree']}")
    print(f"???: {result['confidence']:.1f}%")
    print(f"????: {result['trend_votes']}")
    print()
    
    # ??????
    print("????:")
    print("-" * 80)
    
    if result['consensus_trend'] == '??':
        if result['consensus_degree'] == '?':
            print("??  ??????????????????????????")
            if shop_features and shop_features.get('suspected_churn_rate', 0) > 4:
                print("   - ???????????????????")
            if shop_features and shop_features.get('month_tx_amount_decay_score', 0) > 5:
                print("   - ?????????????????????")
        elif result['consensus_degree'] == '??':
            print("??  ???????????????????????????")
        else:
            print("??  ???????????????????")
    
    elif result['consensus_trend'] == '??':
        if result['consensus_degree'] == '?':
            print("? ????????????????????")
        elif result['consensus_degree'] == '??':
            print("? ???????????????")
        else:
            print("??  ????????????")
    
    else:
        print("??  ?????????????????")
    
    print()


def main():
    """???"""
    # ??????
    os.makedirs('./logs', exist_ok=True)
    
    print("=" * 80)
    print("?????????? - ????")
    print("=" * 80)
    
    # 1. ??????
    print("\n?? 1: ??????...")
    mer_df = create_sample_data()
    print(f"??? {len(mer_df)} ???????? {mer_df['_shop_id'].nunique()} ???")
    
    # 2. ??????
    min_pt = mer_df['date'].min()
    max_pt = mer_df['date'].max()
    print(f"????: {min_pt.date()} ? {max_pt.date()}")
    
    # 3. ???????
    print("\n?? 2: ?????????...")
    churn_features_df = calculate_churn_features(mer_df)
    
    if churn_features_df is None or len(churn_features_df) == 0:
        print("??: ?????????")
        return
    
    print(f"???? {len(churn_features_df)} ?????????")
    print("\n???????:")
    print(churn_features_df[['shop_id', 'month_tx_amount_decay_score', 'suspected_churn_rate']].to_string())
    
    # 4. ??????
    print("\n?? 3: ??????...")
    
    for shop_id in churn_features_df['shop_id'].values:
        analyze_single_shop(mer_df, shop_id, min_pt, max_pt, churn_features_df)
    
    # 5. ??????
    print("\n" + "=" * 80)
    print("????")
    print("=" * 80)
    
    # ????????
    summary = []
    for shop_id in churn_features_df['shop_id'].values:
        shop_data = mer_df[mer_df['_shop_id'] == shop_id].copy()
        shop_features = churn_features_df[churn_features_df['shop_id'] == shop_id].iloc[0].to_dict()
        
        processed_data = _preprocess_data(shop_data, min_pt, max_pt)
        analyzer = TimeSeriesTrendAnalyzer(
            dates=processed_data['date'],
            amounts=processed_data['amount'],
            df=processed_data,
            churn_features=shop_features
        )
        
        result = analyzer.comprehensive_analysis()
        
        summary.append({
            'shop_id': shop_id,
            'shop_name': shop_features.get('_shop_name', 'N/A'),
            'trend': result['consensus_trend'],
            'degree': result['consensus_degree'],
            'confidence': result['confidence'],
            'churn_rate': shop_features.get('suspected_churn_rate', 0),
            'decay_score': shop_features.get('month_tx_amount_decay_score', 0)
        })
    
    summary_df = pd.DataFrame(summary)
    print("\n??????:")
    print(summary_df.to_string(index=False))
    
    print("\n????????????? ./logs/timeseries_trend_analysis.log")


if __name__ == "__main__":
    main()
