import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
import json

class ReviewDataProcessor:
    """
    口コミレビューデータの前処理を行うクラス
    """
    
    def __init__(self):
        """データプロセッサの初期化"""
        self.stop_words = self._load_stop_words()
    
    def _load_stop_words(self) -> List[str]:
        """日本語のストップワードを定義"""
        return [
            'です', 'ます', 'だ', 'である', 'です', 'ます', 'でした', 'ました',
            'の', 'に', 'は', 'を', 'が', 'と', 'で', 'から', 'まで', 'より',
            'も', 'か', 'や', 'など', 'とか', 'って', 'という', 'というのは',
            'これ', 'それ', 'あれ', 'この', 'その', 'あの', 'ここ', 'そこ', 'あそこ',
            '私', 'あなた', '彼', '彼女', '私たち', 'あなたたち', '彼ら', '彼女ら'
        ]
    
    def clean_text(self, text: str) -> str:
        """
        テキストのクリーニング
        
        Args:
            text: クリーニングするテキスト
            
        Returns:
            クリーニングされたテキスト
        """
        if not isinstance(text, str):
            return ""
        
        # HTMLタグを削除
        text = re.sub(r'<[^>]+>', '', text)
        
        # URLを削除
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # メールアドレスを削除
        text = re.sub(r'\S+@\S+', '', text)
        
        # 数字を削除（必要に応じてコメントアウト）
        # text = re.sub(r'\d+', '', text)
        
        # 特殊文字を削除
        text = re.sub(r'[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', '', text)
        
        # 連続する空白を単一の空白に置換
        text = re.sub(r'\s+', ' ', text)
        
        # 前後の空白を削除
        text = text.strip()
        
        return text
    
    def remove_stop_words(self, text: str) -> str:
        """
        ストップワードを削除
        
        Args:
            text: 処理するテキスト
            
        Returns:
            ストップワードを削除したテキスト
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def preprocess_review(self, review: str) -> str:
        """
        レビューの前処理
        
        Args:
            review: 前処理するレビュー
            
        Returns:
            前処理されたレビュー
        """
        # テキストのクリーニング
        cleaned_text = self.clean_text(review)
        
        # ストップワードの削除（オプション）
        # processed_text = self.remove_stop_words(cleaned_text)
        
        return cleaned_text
    
    def create_sample_data(self) -> List[Dict]:
        """
        サンプルの口コミデータを作成
        
        Returns:
            サンプルデータのリスト
        """
        sample_reviews = [
            {
                "review": "この商品は本当に素晴らしいです！品質も良く、配送も早かったです。また購入したいと思います。",
                "rating": 5,
                "category": "positive"
            },
            {
                "review": "期待していた商品でしたが、実際に使ってみると期待外れでした。品質が悪く、すぐに壊れてしまいました。",
                "rating": 1,
                "category": "negative"
            },
            {
                "review": "まあまあの商品です。特に良い点も悪い点もありません。普通の商品だと思います。",
                "rating": 3,
                "category": "neutral"
            },
            {
                "review": "とても満足しています！スタッフの対応も親切で、商品の説明も丁寧でした。おすすめです！",
                "rating": 5,
                "category": "positive"
            },
            {
                "review": "注文してから届くまで時間がかかりすぎました。商品自体は悪くないですが、配送に問題があります。",
                "rating": 2,
                "category": "negative"
            },
            {
                "review": "商品は普通ですが、価格が少し高い気がします。もう少し安ければ良いと思います。",
                "rating": 3,
                "category": "neutral"
            },
            {
                "review": "最高の商品です！友達にも勧めました。リピート購入を考えています。",
                "rating": 5,
                "category": "positive"
            },
            {
                "review": "商品が届いた時点で既に壊れていました。返品手続きも面倒で、二度と利用しません。",
                "rating": 1,
                "category": "negative"
            },
            {
                "review": "商品の品質は良いですが、パッケージが破れていました。中身は問題ありませんでした。",
                "rating": 4,
                "category": "positive"
            },
            {
                "review": "特に問題はありませんが、特に良い点もありません。普通の商品です。",
                "rating": 3,
                "category": "neutral"
            },
            {
                "review": "素晴らしい商品でした！デザインも機能も完璧です。家族全員が気に入っています。",
                "rating": 5,
                "category": "positive"
            },
            {
                "review": "商品の説明と実際の商品が違いました。期待していた機能がなく、がっかりしました。",
                "rating": 2,
                "category": "negative"
            },
            {
                "review": "商品は良いですが、配送料が高すぎます。商品自体の価格は妥当だと思います。",
                "rating": 3,
                "category": "neutral"
            },
            {
                "review": "とても良い商品です！使いやすく、価格も手頃です。友達にも勧めたいと思います。",
                "rating": 5,
                "category": "positive"
            },
            {
                "review": "商品が期待していたものと全く違いました。写真と実際の商品が異なり、騙された気分です。",
                "rating": 1,
                "category": "negative"
            }
        ]
        
        return sample_reviews
    
    def load_data_from_csv(self, filepath: str, text_column: str = 'review') -> List[Dict]:
        """
        CSVファイルからデータを読み込み（テキストのみ）
        
        Args:
            filepath: CSVファイルのパス
            text_column: テキストが含まれる列名（デフォルト: 'review'）
            
        Returns:
            データのリスト（review列のみ）
        """
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            
            # テキスト列が存在するかチェック
            if text_column not in df.columns:
                print(f"エラー: '{text_column}'列が見つかりません。利用可能な列: {list(df.columns)}")
                return []
            
            data = []
            for _, row in df.iterrows():
                # テキストのみを抽出（rating列は無視）
                review_text = str(row[text_column]).strip()
                if review_text and review_text != 'nan':  # 空でないテキストのみ
                    item = {
                        'review': review_text
                    }
                    data.append(item)
            
            print(f"CSVファイルから {len(data)} 件のレビューを読み込みました。")
            return data
            
        except Exception as e:
            print(f"CSVファイルの読み込みに失敗しました: {e}")
            return []
    
    def save_data_to_csv(self, data: List[Dict], filepath: str):
        """
        データをCSVファイルに保存
        
        Args:
            data: 保存するデータ
            filepath: 保存先ファイルパス
        """
        try:
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"データを {filepath} に保存しました。")
        except Exception as e:
            print(f"CSVファイルの保存に失敗しました: {e}")
    
    def process_batch(self, reviews: List[str]) -> List[str]:
        """
        複数のレビューを一括処理
        
        Args:
            reviews: 処理するレビューのリスト
            
        Returns:
            処理されたレビューのリスト
        """
        processed_reviews = []
        for review in reviews:
            processed = self.preprocess_review(review)
            processed_reviews.append(processed)
        
        return processed_reviews
