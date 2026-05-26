# UGC分析によるインバウンド向け投稿戦略デモ

広島 × 韓国Z世代 × Instagram を想定した、UGC分析・戦略提案のプロトタイプです。

## 目的

Instagram上の観光UGCを想定し、単なる「いいね数」ではなく、来訪意図・国内外反応ギャップ・直近トレンドを組み合わせて、インバウンド向けに優先すべき投稿テーマと見せ方を提案します。

## 提出物の構成
## 提出物の構成

```text
.
├── app.py
├── requirements.txt
├── data/
│   └── hiroshima_ugc_raw_dummy.csv
├── KIYONO_UGC_proposal.pdf
└── README.md

## 実行方法

```bash
pip install -r requirements.txt
streamlit run app.py
```

ブラウザでStreamlitが開いたら、以下を順に確認できます。

1. 高PriorityScore投稿
2. 観光要素別の優先度
3. 国内外反応ギャップの時系列
4. いいね数とVisitIntentのズレ
5. 重みの感度分析
6. データ連動のプロモーション戦略

## スコア設計

CSVには `VisitIntent` や `PriorityScore` などのスコア列を持たせていません。`app.py` 内で、生データから以下の指標を計算します。

- `VisitIntent`: 実際に行きたいと思わせる投稿か
- `CreatorBiasCorrection`: インフルエンサー性・顔出しによる過大評価の補正
- `InboundGap`: 国内反応より海外言語圏反応が相対的に高いか
- `TrendGrowth`: 直近30日で海外反応ギャップが伸びているか
- `PriorityScore`: 施策として優先する総合スコア

## データについて

本プロトタイプでは、Instagram API等の取得制約を考慮し、広島観光UGCを模したダミーデータを利用しています。目的は実データから確定的な結論を出すことではなく、UGCをどのように評価し、インバウンド向け情報発信戦略へ変換するかという分析フレームワークを示すことです。

実運用では、Instagram Graph API、SNS分析ツール、公式アカウントのインサイト、広告管理画面、公式サイト遷移、予約ページ遷移などと接続して検証します。
