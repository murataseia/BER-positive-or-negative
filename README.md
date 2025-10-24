# 日本語口コミレビュー感情分析システム

BERTを使用した日本語の口コミレビューの感情分析システムです。ポジティブ、ネガティブ、ニュートラルの3つの感情に分類します。

## 特徴

- **BERTベース**: 東北大学の日本語BERTモデルを使用
- **ハイブリッド分析**: ルールベース分析とBERT分析を組み合わせ
- **3分類**: ポジティブ、ネガティブ、ニュートラル
- **可視化機能**: 分析結果をグラフで表示
- **複数の実行モード**: 対話モード、CSV分析、サンプル分析

## インストール

### 1. リポジトリのクローン
```bash
git clone <repository-url>
cd pojinega
```

### 2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

## 使用方法

### 対話モード（デフォルト）
```bash
python main_sentiment_analysis.py
```

### CSVファイルから分析
```bash
python main_sentiment_analysis.py --input your_reviews.csv --visualize
```

### サンプルデータで分析
```bash
python main_sentiment_analysis.py --sample --visualize
```

### 単一テキストの分析
```bash
python main_sentiment_analysis.py --text "この商品は最高です！"
```

## ファイル構成

```
pojinega/
├── main_sentiment_analysis.py    # メイン実行ファイル
├── sentiment_analyzer.py         # 感情分析エンジン
├── data_processor.py             # データ前処理
├── requirements.txt              # 依存関係
├── README.md                     # このファイル
├── .gitignore                   # Git除外ファイル
└── simple_reviews.csv           # サンプルデータ
```

## 感情分析の仕組み

### 1. ルールベース分析（メイン）
- **ポジティブキーワード**: 素晴らしい、最高、満足、おすすめ など
- **ネガティブキーワード**: 悪い、最悪、不満、期待外れ など
- **ニュートラルキーワード**: まあまあ、普通、特に など
- **ニュートラルパターン**: "特に良い点も悪い点もありません" など

### 2. BERT分析（補助）
- 東北大学の日本語BERTモデルを使用
- ルールベースの結果と組み合わせて信頼度を調整

### 3. 感情判断ロジック
1. ニュートラルな表現パターンをチェック
2. キーワードの出現回数をカウント
3. ポジティブ/ネガティブキーワードの数を比較
4. BERTの結果と組み合わせて最終判断

## 出力ファイル

- `sentiment_results.csv`: 詳細な分析結果
- `sentiment_analysis_visualization.png`: 可視化結果

## サンプルデータ

`simple_reviews.csv`には以下のようなサンプルレビューが含まれています：

- ポジティブ: "この商品は本当に素晴らしいです！品質も良く、配送も早かったです。"
- ネガティブ: "期待していた商品でしたが、実際に使ってみると期待外れでした。"
- ニュートラル: "まあまあの商品です。特に良い点も悪い点もありません。"

## 技術仕様

- **Python**: 3.8以上
- **BERTモデル**: cl-tohoku/bert-base-japanese-whole-word-masking
- **主要ライブラリ**: torch, transformers, pandas, matplotlib, seaborn

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

プルリクエストやイシューの報告を歓迎します。

## 更新履歴

- v1.0.0: 初回リリース
  - BERTベースの感情分析機能
  - ルールベース分析との組み合わせ
  - 可視化機能
  - 複数の実行モード