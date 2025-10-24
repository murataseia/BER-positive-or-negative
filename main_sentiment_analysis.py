#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口コミレビュー感情分析システム
BERTを使用してポジティブ/ネガティブを分類
"""

import os
import sys
import argparse
from typing import List, Dict
import pandas as pd

# 自作モジュールのインポート
from sentiment_analyzer import JapaneseSentimentAnalyzer
from data_processor import ReviewDataProcessor

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='口コミレビュー感情分析システム')
    parser.add_argument('--input', '-i', type=str, help='入力CSVファイルのパス')
    parser.add_argument('--output', '-o', type=str, default='sentiment_results.csv', help='出力CSVファイルのパス')
    parser.add_argument('--visualize', '-v', action='store_true', help='結果を可視化する')
    parser.add_argument('--sample', '-s', action='store_true', help='サンプルデータを使用する')
    parser.add_argument('--text', '-t', type=str, help='単一のテキストを分析する')
    
    args = parser.parse_args()
    
    print("=== 口コミレビュー感情分析システム ===")
    print("BERTを使用してポジティブ/ネガティブを分類します\n")
    
    # 感情分析器の初期化
    print("感情分析器を初期化中...")
    analyzer = JapaneseSentimentAnalyzer()
    
    # データプロセッサの初期化
    processor = ReviewDataProcessor()
    
    if args.text:
        # 単一テキストの分析
        print(f"分析対象テキスト: {args.text}")
        result = analyzer.analyze_sentiment(args.text)
        
        print("\n=== 分析結果 ===")
        print(f"テキスト: {result['text']}")
        
        # 感情を日本語で表示
        sentiment_jp = {
            'positive': 'ポジティブ（良い）',
            'negative': 'ネガティブ（悪い）',
            'neutral': 'ニュートラル（普通）'
        }
        sentiment_display = sentiment_jp.get(result['sentiment'], result['sentiment'])
        print(f"感情: {sentiment_display}")
        
        # 信頼度を日本語で表示
        confidence_percent = result['confidence'] * 100
        if confidence_percent >= 80:
            confidence_level = "非常に高い"
        elif confidence_percent >= 60:
            confidence_level = "高い"
        elif confidence_percent >= 40:
            confidence_level = "中程度"
        else:
            confidence_level = "低い"
        
        print(f"信頼度: {confidence_percent:.1f}% ({confidence_level})")
        print(f"詳細スコア: {result['scores']}")
        
    elif args.sample:
        # サンプルデータを使用
        print("サンプルデータを使用して分析を実行します...")
        sample_data = processor.create_sample_data()
        
        # レビューテキストを抽出
        reviews = [item['review'] for item in sample_data]
        
        # 感情分析を実行
        print("感情分析を実行中...")
        results = analyzer.analyze_batch(reviews)
        
        # 結果を表示
        print("\n=== 分析結果 ===")
        for i, result in enumerate(results):
            print(f"\nレビュー {i+1}:")
            print(f"  テキスト: {result['text'][:50]}...")
            
            # 感情を日本語で表示
            sentiment_jp = {
                'positive': 'ポジティブ（良い）',
                'negative': 'ネガティブ（悪い）',
                'neutral': 'ニュートラル（普通）'
            }
            sentiment_display = sentiment_jp.get(result['sentiment'], result['sentiment'])
            print(f"  感情: {sentiment_display}")
            
            # 信頼度を日本語で表示
            confidence_percent = result['confidence'] * 100
            if confidence_percent >= 80:
                confidence_level = "非常に高い"
            elif confidence_percent >= 60:
                confidence_level = "高い"
            elif confidence_percent >= 40:
                confidence_level = "中程度"
            else:
                confidence_level = "低い"
            
            print(f"  信頼度: {confidence_percent:.1f}% ({confidence_level})")
        
        # サマリー統計
        summary = analyzer.get_sentiment_summary(results)
        print(f"\n=== サマリー統計 ===")
        print(f"総レビュー数: {summary['total_reviews']}件")
        print(f"ポジティブ（良い）: {summary['positive_count']}件 ({summary['positive_ratio']:.1%})")
        print(f"ネガティブ（悪い）: {summary['negative_count']}件 ({summary['negative_ratio']:.1%})")
        print(f"ニュートラル（普通）: {summary['neutral_count']}件 ({summary['neutral_ratio']:.1%})")
        
        # 平均信頼度を日本語で表示
        avg_confidence_percent = summary['average_confidence'] * 100
        if avg_confidence_percent >= 80:
            avg_confidence_level = "非常に高い"
        elif avg_confidence_percent >= 60:
            avg_confidence_level = "高い"
        elif avg_confidence_percent >= 40:
            avg_confidence_level = "中程度"
        else:
            avg_confidence_level = "低い"
        
        print(f"平均信頼度: {avg_confidence_percent:.1f}% ({avg_confidence_level})")
        
        # 結果をCSVに保存
        analyzer.export_results(results, args.output)
        
        # 可視化
        if args.visualize:
            print("\n結果を可視化中...")
            analyzer.visualize_results(results, 'sentiment_analysis_visualization.png')
        
    elif args.input:
        # CSVファイルからデータを読み込み
        if not os.path.exists(args.input):
            print(f"エラー: ファイル '{args.input}' が見つかりません。")
            return
        
        print(f"ファイル '{args.input}' からデータを読み込み中...")
        data = processor.load_data_from_csv(args.input)
        
        if not data:
            print("データの読み込みに失敗しました。")
            return
        
        # レビューテキストを抽出
        reviews = [item['review'] for item in data]
        
        # 感情分析を実行
        print("感情分析を実行中...")
        results = analyzer.analyze_batch(reviews)
        
        # 結果を表示
        print(f"\n=== 分析結果 ({len(results)}件) ===")
        for i, result in enumerate(results[:5]):  # 最初の5件のみ表示
            print(f"\nレビュー {i+1}:")
            print(f"  テキスト: {result['text'][:50]}...")
            
            # 感情を日本語で表示
            sentiment_jp = {
                'positive': 'ポジティブ（良い）',
                'negative': 'ネガティブ（悪い）',
                'neutral': 'ニュートラル（普通）'
            }
            sentiment_display = sentiment_jp.get(result['sentiment'], result['sentiment'])
            print(f"  感情: {sentiment_display}")
            
            # 信頼度を日本語で表示
            confidence_percent = result['confidence'] * 100
            if confidence_percent >= 80:
                confidence_level = "非常に高い"
            elif confidence_percent >= 60:
                confidence_level = "高い"
            elif confidence_percent >= 40:
                confidence_level = "中程度"
            else:
                confidence_level = "低い"
            
            print(f"  信頼度: {confidence_percent:.1f}% ({confidence_level})")
        
        if len(results) > 5:
            print(f"\n... 他 {len(results) - 5} 件の結果")
        
        # サマリー統計
        summary = analyzer.get_sentiment_summary(results)
        print(f"\n=== サマリー統計 ===")
        print(f"総レビュー数: {summary['total_reviews']}件")
        print(f"ポジティブ（良い）: {summary['positive_count']}件 ({summary['positive_ratio']:.1%})")
        print(f"ネガティブ（悪い）: {summary['negative_count']}件 ({summary['negative_ratio']:.1%})")
        print(f"ニュートラル（普通）: {summary['neutral_count']}件 ({summary['neutral_ratio']:.1%})")
        
        # 平均信頼度を日本語で表示
        avg_confidence_percent = summary['average_confidence'] * 100
        if avg_confidence_percent >= 80:
            avg_confidence_level = "非常に高い"
        elif avg_confidence_percent >= 60:
            avg_confidence_level = "高い"
        elif avg_confidence_percent >= 40:
            avg_confidence_level = "中程度"
        else:
            avg_confidence_level = "低い"
        
        print(f"平均信頼度: {avg_confidence_percent:.1f}% ({avg_confidence_level})")
        
        # 結果をCSVに保存
        analyzer.export_results(results, args.output)
        
        # 可視化
        if args.visualize:
            print("\n結果を可視化中...")
            analyzer.visualize_results(results, 'sentiment_analysis_visualization.png')
    
    else:
        # 対話モード
        print("対話モードで感情分析を実行します。")
        print("分析したいテキストを入力してください（終了するには 'quit' と入力）:")
        
        while True:
            text = input("\nテキスト: ").strip()
            
            if text.lower() in ['quit', 'exit', '終了']:
                print("分析を終了します。")
                break
            
            if not text:
                print("テキストを入力してください。")
                continue
            
            # 感情分析を実行
            result = analyzer.analyze_sentiment(text)
            
            print(f"\n分析結果:")
            
            # 感情を日本語で表示
            sentiment_jp = {
                'positive': 'ポジティブ（良い）',
                'negative': 'ネガティブ（悪い）',
                'neutral': 'ニュートラル（普通）'
            }
            sentiment_display = sentiment_jp.get(result['sentiment'], result['sentiment'])
            print(f"  感情: {sentiment_display}")
            
            # 信頼度を日本語で表示
            confidence_percent = result['confidence'] * 100
            if confidence_percent >= 80:
                confidence_level = "非常に高い"
            elif confidence_percent >= 60:
                confidence_level = "高い"
            elif confidence_percent >= 40:
                confidence_level = "中程度"
            else:
                confidence_level = "低い"
            
            print(f"  信頼度: {confidence_percent:.1f}% ({confidence_level})")
            print(f"  詳細スコア: {result['scores']}")

def demo():
    """デモンストレーション関数"""
    print("=== デモンストレーション ===")
    
    # 感情分析器の初期化
    analyzer = JapaneseSentimentAnalyzer()
    
    # サンプルテキスト
    sample_texts = [
        "この商品は本当に素晴らしいです！品質も良く、配送も早かったです。",
        "期待していた商品でしたが、実際に使ってみると期待外れでした。",
        "まあまあの商品です。特に良い点も悪い点もありません。"
    ]
    
    print("サンプルテキストの感情分析:")
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{i}. {text}")
        result = analyzer.analyze_sentiment(text)
        
        # 感情を日本語で表示
        sentiment_jp = {
            'positive': 'ポジティブ（良い）',
            'negative': 'ネガティブ（悪い）',
            'neutral': 'ニュートラル（普通）'
        }
        sentiment_display = sentiment_jp.get(result['sentiment'], result['sentiment'])
        print(f"   感情: {sentiment_display}")
        
        # 信頼度を日本語で表示
        confidence_percent = result['confidence'] * 100
        if confidence_percent >= 80:
            confidence_level = "非常に高い"
        elif confidence_percent >= 60:
            confidence_level = "高い"
        elif confidence_percent >= 40:
            confidence_level = "中程度"
        else:
            confidence_level = "低い"
        
        print(f"   信頼度: {confidence_percent:.1f}% ({confidence_level})")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 引数がない場合は対話モードを実行
        print("=== 口コミレビュー感情分析システム ===")
        print("BERTを使用してポジティブ/ネガティブを分類します\n")
        
        # 感情分析器の初期化
        print("感情分析器を初期化中...")
        analyzer = JapaneseSentimentAnalyzer()
        
        print("対話モードで感情分析を実行します。")
        print("分析したいテキストを入力してください（終了するには 'quit' と入力）:")
        
        while True:
            text = input("\nテキスト: ").strip()
            
            if text.lower() in ['quit', 'exit', '終了']:
                print("分析を終了します。")
                break
            
            if not text:
                print("テキストを入力してください。")
                continue
            
            # 感情分析を実行
            result = analyzer.analyze_sentiment(text)
            
            print(f"\n分析結果:")
            
            # 感情を日本語で表示
            sentiment_jp = {
                'positive': 'ポジティブ（良い）',
                'negative': 'ネガティブ（悪い）',
                'neutral': 'ニュートラル（普通）'
            }
            sentiment_display = sentiment_jp.get(result['sentiment'], result['sentiment'])
            print(f"  感情: {sentiment_display}")
            
            # 信頼度を日本語で表示
            confidence_percent = result['confidence'] * 100
            if confidence_percent >= 80:
                confidence_level = "非常に高い"
            elif confidence_percent >= 60:
                confidence_level = "高い"
            elif confidence_percent >= 40:
                confidence_level = "中程度"
            else:
                confidence_level = "低い"
            
            print(f"  信頼度: {confidence_percent:.1f}% ({confidence_level})")
            print(f"  詳細スコア: {result['scores']}")
    else:
        main()
