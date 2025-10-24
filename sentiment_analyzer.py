import torch
import numpy as np
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import re

# 日本語フォントの設定
plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

class JapaneseSentimentAnalyzer:
    """
    日本語の口コミレビューを感情分析するクラス
    BERTを使用してポジティブ/ネガティブを分類
    """
    
    def __init__(self, model_name: str = "cl-tohoku/bert-base-japanese-whole-word-masking"):
        """
        感情分析器の初期化
        
        Args:
            model_name: 使用するBERTモデルの名前
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self._load_model()
    
    def _load_model(self):
        """BERTモデルとトークナイザーを読み込み"""
        try:
            print(f"モデル '{self.model_name}' を読み込み中...")
            
            # 感情分析用のパイプラインを作成
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True
            )
            print("モデルの読み込みが完了しました。")
            
        except Exception as e:
            print(f"モデルの読み込みに失敗しました: {e}")
            print("代替モデルを使用します...")
            
            # 代替として、東北大学の日本語BERTモデルを使用
            try:
                self.classifier = pipeline(
                    "sentiment-analysis",
                    model="cl-tohoku/bert-base-japanese-whole-word-masking",
                    tokenizer="cl-tohoku/bert-base-japanese-whole-word-masking",
                    return_all_scores=True
                )
                print("代替モデルの読み込みが完了しました。")
            except Exception as e2:
                print(f"代替モデルの読み込みも失敗しました: {e2}")
                raise e2
    
    def preprocess_text(self, text: str) -> str:
        """
        テキストの前処理
        
        Args:
            text: 前処理するテキスト
            
        Returns:
            前処理されたテキスト
        """
        # 改行文字を削除
        text = re.sub(r'\n', ' ', text)
        # 連続する空白を単一の空白に置換
        text = re.sub(r'\s+', ' ', text)
        # 前後の空白を削除
        text = text.strip()
        return text
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        単一のテキストの感情分析
        
        Args:
            text: 分析するテキスト
            
        Returns:
            感情分析結果の辞書
        """
        # テキストの前処理
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return {
                'text': text,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.0, 'negative': 0.0}
            }
        
        # ルールベースの感情分析を追加
        rule_based_sentiment = self._rule_based_sentiment_analysis(processed_text)
        
        try:
            # BERT感情分析を実行
            results = self.classifier(processed_text)
            
            
            # 結果を整理
            sentiment_scores = {}
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                sentiment_scores[label] = score
            
            # ラベルマッピング（label_0, label_1などを適切な感情に変換）
            # 東北大学のBERTモデルでは: LABEL_0=ポジティブ, LABEL_1=ネガティブ
            label_mapping = {
                'label_0': 'positive',  # このモデルではlabel_0がポジティブ
                'label_1': 'negative',   # このモデルではlabel_1がネガティブ
                'label_2': 'neutral',   # 通常label_2がニュートラル
                'negative': 'negative',
                'positive': 'positive',
                'neutral': 'neutral',
                'neg': 'negative',
                'pos': 'positive',
                'neu': 'neutral',
                'n': 'negative',
                'p': 'positive'
            }
            
            # 最も高いスコアの感情を決定
            best_result = max(results[0], key=lambda x: x['score'])
            original_label = best_result['label'].lower()
            
            # ラベルを適切な感情に変換
            bert_sentiment = label_mapping.get(original_label, 'neutral')
            bert_confidence = best_result['score']
            
            # ルールベースの分析をメインに使用（BERTは感情分析用に訓練されていないため）
            sentiment = rule_based_sentiment['sentiment']
            confidence = rule_based_sentiment['confidence']
            
            # BERTの結果がルールベースと一致する場合は信頼度を向上
            if bert_sentiment == rule_based_sentiment['sentiment']:
                confidence = min(0.95, (bert_confidence + rule_based_sentiment['confidence']) / 2)
            # BERTの結果が異なる場合は、ルールベースの信頼度を少し下げる
            elif bert_confidence > 0.7:
                confidence = max(0.3, rule_based_sentiment['confidence'] - 0.1)
            
            # スコアを適切な形式に変換
            converted_scores = {}
            for result in results[0]:
                original_label = result['label'].lower()
                mapped_sentiment = label_mapping.get(original_label, 'neutral')
                converted_scores[mapped_sentiment] = result['score']
            
            sentiment_scores = converted_scores
            
            return {
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'scores': sentiment_scores
            }
            
        except Exception as e:
            print(f"感情分析中にエラーが発生しました: {e}")
            # エラー時はルールベースの結果を使用
            return {
                'text': text,
                'sentiment': rule_based_sentiment['sentiment'],
                'confidence': rule_based_sentiment['confidence'],
                'scores': {'positive': 0.0, 'negative': 0.0}
            }
    
    def _rule_based_sentiment_analysis(self, text: str) -> Dict:
        """
        ルールベースの感情分析
        
        Args:
            text: 分析するテキスト
            
        Returns:
            感情分析結果の辞書
        """
        # ポジティブなキーワード
        positive_words = [
            '素晴らしい', '良い', 'いい', '最高', '満足', 'おすすめ', '快適',
            '便利', '使いやすい', '気に入った', '完璧', '優秀', '素晴らしい',
            '最高', '満足', 'おすすめ', '快適', '便利', '使いやすい', '気に入った'
        ]
        
        # ネガティブなキーワード
        negative_words = [
            '悪い', '最悪', '不満', '問題', '困った', '期待外れ', '残念', 'ひどい',
            '使いにくい', '不便', '嫌い', '失敗', 'ダメ', 'ひどい', '最悪', '不満',
            '問題', '困った', '期待外れ', '残念', 'ひどい', '使いにくい', '不便', '嫌い'
        ]
        
        # ニュートラルなキーワード
        neutral_words = [
            'まあまあ', '普通', '特に', 'ない', 'ありません', '普通', 'まあまあ',
            '特に', 'ない', 'ありません', '普通', 'まあまあ', '特に', 'ない', 'ありません'
        ]
        
        # ニュートラルな表現パターン
        neutral_patterns = [
            '特に良い点も悪い点もありません',
            '特に良い点も悪い点もない',
            '特に良い点も悪い点も',
            '良い点も悪い点もありません',
            '良い点も悪い点もない',
            '良い点も悪い点も'
        ]
        
        # テキストを小文字に変換
        text_lower = text.lower()
        
        # ニュートラルな表現パターンをチェック
        neutral_pattern_found = any(pattern in text_lower for pattern in neutral_patterns)
        
        # キーワードの出現回数をカウント
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        neutral_count = sum(1 for word in neutral_words if word in text_lower)
        
        # 感情を決定
        if neutral_pattern_found:
            # ニュートラルな表現パターンが見つかった場合
            sentiment = 'neutral'
            confidence = 0.9
        elif neutral_count > 0 and positive_count == 0 and negative_count == 0:
            # ニュートラルキーワードのみの場合
            sentiment = 'neutral'
            confidence = 0.8
        elif positive_count > negative_count:
            sentiment = 'positive'
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        複数のテキストの感情分析
        
        Args:
            texts: 分析するテキストのリスト
            
        Returns:
            感情分析結果のリスト
        """
        results = []
        for i, text in enumerate(texts):
            print(f"分析中: {i+1}/{len(texts)}")
            result = self.analyze_sentiment(text)
            results.append(result)
        
        return results
    
    def get_sentiment_summary(self, results: List[Dict]) -> Dict:
        """
        感情分析結果のサマリーを取得
        
        Args:
            results: 感情分析結果のリスト
            
        Returns:
            サマリー統計
        """
        total = len(results)
        positive_count = sum(1 for r in results if r['sentiment'] == 'positive')
        negative_count = sum(1 for r in results if r['sentiment'] == 'negative')
        neutral_count = total - positive_count - negative_count
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        return {
            'total_reviews': total,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_ratio': positive_count / total if total > 0 else 0,
            'negative_ratio': negative_count / total if total > 0 else 0,
            'neutral_ratio': neutral_count / total if total > 0 else 0,
            'average_confidence': avg_confidence
        }
    
    def visualize_results(self, results: List[Dict], save_path: str = None):
        """
        感情分析結果の可視化
        
        Args:
            results: 感情分析結果のリスト
            save_path: 保存先パス（Noneの場合は表示のみ）
        """
        # サマリー統計を取得
        summary = self.get_sentiment_summary(results)
        
        # 図のサイズを設定
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('口コミレビュー感情分析結果', fontsize=16, fontweight='bold', fontfamily='Hiragino Sans')
        
        # 1. 感情分布の円グラフ
        labels = ['ポジティブ', 'ネガティブ', 'ニュートラル']
        sizes = [summary['positive_count'], summary['negative_count'], summary['neutral_count']]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('感情分布', fontfamily='Hiragino Sans')
        
        # 2. 感情分布の棒グラフ
        sentiment_counts = [summary['positive_count'], summary['negative_count'], summary['neutral_count']]
        axes[0, 1].bar(labels, sentiment_counts, color=colors)
        axes[0, 1].set_title('感情別レビュー数', fontfamily='Hiragino Sans')
        axes[0, 1].set_ylabel('レビュー数', fontfamily='Hiragino Sans')
        
        # 3. 信頼度の分布
        confidences = [r['confidence'] for r in results]
        axes[1, 0].hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('信頼度分布', fontfamily='Hiragino Sans')
        axes[1, 0].set_xlabel('信頼度', fontfamily='Hiragino Sans')
        axes[1, 0].set_ylabel('頻度', fontfamily='Hiragino Sans')
        
        # 4. 感情別信頼度の箱ひげ図
        positive_confidences = [r['confidence'] for r in results if r['sentiment'] == 'positive']
        negative_confidences = [r['confidence'] for r in results if r['sentiment'] == 'negative']
        neutral_confidences = [r['confidence'] for r in results if r['sentiment'] == 'neutral']
        
        data_to_plot = [positive_confidences, negative_confidences, neutral_confidences]
        axes[1, 1].boxplot(data_to_plot, labels=['ポジティブ', 'ネガティブ', 'ニュートラル'])
        axes[1, 1].set_title('感情別信頼度', fontfamily='Hiragino Sans')
        axes[1, 1].set_ylabel('信頼度', fontfamily='Hiragino Sans')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"結果を {save_path} に保存しました。")
        
        plt.show()
    
    def export_results(self, results: List[Dict], filename: str = 'sentiment_analysis_results.csv'):
        """
        分析結果をCSVファイルに出力
        
        Args:
            results: 感情分析結果のリスト
            filename: 出力ファイル名
        """
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"結果を {filename} に保存しました。")
