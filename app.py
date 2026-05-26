from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "hiroshima_ugc_raw_dummy.csv"


# -----------------------------
# Normalization helpers
# -----------------------------
def log_normalize(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    max_value = series.max()
    if max_value <= 0:
        return pd.Series(0.0, index=series.index)
    return np.log1p(series) / np.log1p(max_value)


def max_normalize(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    max_value = series.max()
    if max_value <= 0:
        return pd.Series(0.0, index=series.index)
    return series / max_value


def minmax_normalize(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return pd.Series(0.5, index=series.index)
    return (series - min_value) / (max_value - min_value)


# -----------------------------
# Data / score computation
# -----------------------------
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def compute_visit_intent(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute VisitIntent from raw post-level variables.

    The input CSV intentionally does NOT include VisitIntent. This function
    computes it from raw behavioral / content / creator variables.
    """
    df = raw_df.copy()
    df["post_date"] = pd.to_datetime(df["post_date"])

    # Information density: Can a viewer realistically act on this post?
    df["InfoDensity"] = (
        df["has_access_info"]
        + df["has_time_info"]
        + df["has_cost_info"]
        + df["has_plan"]
    ) / 4

    # SaveProxy: direct saves are hard to obtain from public data, so we proxy
    # "worth saving for later" using planning/actionable components.
    df["SaveProxy"] = (
        df["has_plan"]
        + df["has_access_info"]
        + df["has_time_info"]
        + df["has_route_map"]
        + df["has_booking_hint"]
    ) / 5

    # CommentIntent: explicit "I want to go" type comments, log-normalized.
    df["CommentIntent"] = log_normalize(df["comment_intent_count"])

    # AdjustedEngagement: likes are kept, but normalized by follower count.
    df["engagement_raw"] = df["likes"] / df["follower_count"].replace(0, np.nan)
    df["engagement_raw"] = df["engagement_raw"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["AdjustedEngagement"] = max_normalize(df["engagement_raw"])

    # VisitIntent score: hypothesis-based weights. The dashboard later checks
    # sensitivity to these weights so the model is not presented as absolute.
    df["BaseScore"] = (
        0.35 * df["InfoDensity"]
        + 0.30 * df["CommentIntent"]
        + 0.20 * df["SaveProxy"]
        + 0.15 * df["AdjustedEngagement"]
    )

    # Creator bias correction: reduce over-crediting posts driven by creator fame
    # or face-centric appeal rather than tourism content.
    df["CreatorBiasCorrection"] = (
        1 - 0.15 * df["is_influencer"] - 0.10 * df["has_face"]
    ).clip(lower=0.60, upper=1.00)

    df["VisitIntent"] = df["BaseScore"] * df["CreatorBiasCorrection"]
    return df


def compute_inbound_gap_and_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Compute InboundGap and TrendGrowth.

    TrendGrowth follows the proposal: latest 30 days vs the previous 90 days.
    This avoids using a vague latest-month-vs-all-past proxy.
    """
    df = df.copy()

    df["JapaneseReaction"] = log_normalize(df["japanese_reaction_count"])
    df["ForeignReaction"] = log_normalize(df["foreign_reaction_count"])
    df["InboundGapRaw"] = df["ForeignReaction"] - df["JapaneseReaction"]
    df["InboundGap"] = minmax_normalize(df["InboundGapRaw"])

    max_date = df["post_date"].max()
    recent_start = max_date - pd.Timedelta(days=30)
    past_start = max_date - pd.Timedelta(days=120)

    df["trend_window"] = np.where(
        df["post_date"] >= recent_start,
        "recent_30d",
        np.where(df["post_date"] >= past_start, "previous_90d", "older"),
    )

    recent = (
        df[df["trend_window"] == "recent_30d"]
        .groupby("element_tag")["InboundGapRaw"]
        .mean()
        .rename("recent_gap")
    )
    past = (
        df[df["trend_window"] == "previous_90d"]
        .groupby("element_tag")["InboundGapRaw"]
        .mean()
        .rename("past_gap")
    )
    trend = pd.concat([recent, past], axis=1)

    # If one side is missing in the dummy data, use the available overall mean
    # for the element to avoid producing artificial extremes.
    element_mean = df.groupby("element_tag")["InboundGapRaw"].mean().rename("element_mean")
    trend = trend.join(element_mean, how="outer")
    trend["recent_gap"] = trend["recent_gap"].fillna(trend["element_mean"])
    trend["past_gap"] = trend["past_gap"].fillna(trend["element_mean"])
    trend = trend.fillna(0)
    trend["TrendGrowthRaw"] = trend["recent_gap"] - trend["past_gap"]
    trend["TrendGrowth"] = minmax_normalize(trend["TrendGrowthRaw"])

    df = df.merge(
        trend[["recent_gap", "past_gap", "TrendGrowthRaw", "TrendGrowth"]],
        left_on="element_tag",
        right_index=True,
        how="left",
    )
    df[["recent_gap", "past_gap", "TrendGrowthRaw", "TrendGrowth"]] = df[
        ["recent_gap", "past_gap", "TrendGrowthRaw", "TrendGrowth"]
    ].fillna(0)
    return df


def compute_scores(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Raw UGC-like data -> scores used by the dashboard.

    Important: the CSV intentionally does not include VisitIntent, InboundGap,
    TrendGrowth, or PriorityScore. They are computed here so the demo behaves
    as an analysis system rather than a spreadsheet viewer.
    """
    df = compute_visit_intent(raw_df)
    df = compute_inbound_gap_and_trend(df)

    # Strategic prioritization. This is not an objective truth; sensitivity
    # analysis below checks whether top elements change when weights move.
    df["PriorityScore"] = (
        0.50 * df["VisitIntent"]
        + 0.30 * df["InboundGap"]
        + 0.20 * df["TrendGrowth"]
    )

    return df


# -----------------------------
# Summary / strategy generation
# -----------------------------
def summarize_elements(scored_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scored_df.groupby("element_tag")
        .agg(
            posts=("post_id", "count"),
            avg_priority=("PriorityScore", "mean"),
            avg_visit_intent=("VisitIntent", "mean"),
            avg_inbound_gap=("InboundGapRaw", "mean"),
            trend_growth=("TrendGrowthRaw", "mean"),
            access_rate=("has_access_info", "mean"),
            plan_rate=("has_plan", "mean"),
            route_rate=("has_route_map", "mean"),
        )
        .reset_index()
        .sort_values("avg_priority", ascending=False)
    )
    return summary


def compute_sensitivity(scored_df: pd.DataFrame, top_n: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check whether strategic top elements remain stable when weights change."""
    schemes = {
        "baseline_50_30_20": (0.50, 0.30, 0.20),
        "visit_heavy_70_20_10": (0.70, 0.20, 0.10),
        "inbound_heavy_40_45_15": (0.40, 0.45, 0.15),
        "trend_heavy_40_25_35": (0.40, 0.25, 0.35),
    }

    rows = []
    scored = scored_df.copy()
    for scheme, (wv, wi, wt) in schemes.items():
        scored["scenario_score"] = (
            wv * scored["VisitIntent"]
            + wi * scored["InboundGap"]
            + wt * scored["TrendGrowth"]
        )
        summary = (
            scored.groupby("element_tag")["scenario_score"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
        )
        for rank, (element, score) in enumerate(summary.items(), start=1):
            rows.append({"scheme": scheme, "rank": rank, "element_tag": element, "score": score})

    ranking_df = pd.DataFrame(rows)
    stability_df = (
        ranking_df.groupby("element_tag")
        .agg(
            top3_count=("scheme", "nunique"),
            best_rank=("rank", "min"),
            avg_rank=("rank", "mean"),
        )
        .reset_index()
        .sort_values(["top3_count", "best_rank"], ascending=[False, True])
    )
    return ranking_df, stability_df


def find_like_vs_intent_examples(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Find posts that are liked a lot but have lower VisitIntent, for explanation."""
    df = scored_df.copy()
    df["likes_rank_pct"] = df["likes"].rank(pct=True)
    df["visit_rank_pct"] = df["VisitIntent"].rank(pct=True)
    df["like_intent_gap"] = df["likes_rank_pct"] - df["visit_rank_pct"]
    return df.sort_values("like_intent_gap", ascending=False)[
        [
            "post_id", "spot", "element_tag", "content_type", "likes", "follower_count",
            "InfoDensity", "CommentIntent", "VisitIntent", "like_intent_gap"
        ]
    ].head(8)


def make_strategy_text(scored_df: pd.DataFrame, n: int = 3) -> list[str]:
    summary = summarize_elements(scored_df).head(n)
    strategies: list[str] = []

    element_advice = {
        "路面電車": "広島らしい移動体験として、乗り方・所要時間・降車後の回遊先までセットで見せる",
        "島旅": "島単体ではなく、船/電車/徒歩の移動手順と半日プランを組み合わせる",
        "海沿いサイクリング": "レンタル方法・走行時間・初心者向け区間を明示し、体験投稿として設計する",
        "お好み焼き": "注文方法・焼き方・店の選び方を入れ、食文化体験として見せる",
        "カフェ巡り": "徒歩ルート、近接スポット、滞在時間を含む保存型投稿にする",
        "平和学習": "単体の写真ではなく、背景説明・所要時間・周辺回遊と合わせて文脈化する",
        "夜景": "撮影地点・アクセス・帰り方を添え、景色単体の消費で終わらせない",
        "商店街": "買い歩き・小物・ローカルフードを組み合わせ、短時間で回れる導線にする",
        "商店街・日常風景": "買い歩き・自動販売機・路地・ローカルフードなど、外国人にとって新鮮に見える日常要素を回遊導線に組み込む",
        "歴史・平和文脈": "写真単体ではなく、背景説明・見学所要時間・周辺回遊を添えて文脈化する",
        "ローカルフード": "食べ方・注文方法・周辺観光を合わせ、食体験を旅程の起点にする",
    }
    content_advice = {
        "plan": "半日/1日プランとして、移動時間・費用・回り方をセットで見せる",
        "food": "店名・注文方法・周辺スポットまで含めて、食体験を旅行導線に接続する",
        "cafe": "徒歩ルートや周辺散策と合わせて、保存しやすいカフェ巡り投稿にする",
        "landscape": "景色単体ではなく、撮影地点・行き方・前後の回遊先を添える",
        "experience": "体験の手順・所要時間・予約/料金情報を含める",
    }

    for _, row in summary.iterrows():
        element = row["element_tag"]
        subset = scored_df[scored_df["element_tag"] == element].copy()
        spot = subset.groupby("spot")["PriorityScore"].mean().sort_values(ascending=False).index[0]
        ctype = subset.groupby("content_type")["PriorityScore"].mean().sort_values(ascending=False).index[0]

        reason_parts = []
        if row["avg_visit_intent"] >= scored_df["VisitIntent"].mean():
            reason_parts.append("来訪意図スコアが平均以上")
        if row["avg_inbound_gap"] > 0:
            reason_parts.append("国内反応より海外言語圏の反応が相対的に高い")
        if row["trend_growth"] > 0:
            reason_parts.append("直近30日で海外反応ギャップが拡大している")
        if not reason_parts:
            reason_parts.append("総合優先度が相対的に高い")

        execution_parts = []
        execution_parts.append("アクセス情報は維持" if row["access_rate"] >= 0.5 else "アクセス情報を追加")
        execution_parts.append("プラン形式を活用" if row["plan_rate"] >= 0.35 else "回遊プランへ組み込む")
        execution_parts.append("地図/導線を明示" if row["route_rate"] >= 0.35 else "地図・導線で補強")

        advice = element_advice.get(element, content_advice.get(ctype, "情報量の高い投稿にする"))
        strategies.append(
            f"{element}: {spot}を中心に、{advice}。"
            f"理由は{ '、'.join(reason_parts) }ため。"
            f"実行時は{ '、'.join(execution_parts) }。"
        )

    return strategies


# -----------------------------
# Streamlit UI
# -----------------------------
def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="UGC Inbound Strategy Demo", layout="wide")
    st.title("UGC分析によるインバウンド向け投稿戦略デモ")
    st.caption("広島 × 韓国Z世代 × Instagramを想定したプロトタイプ。CSVは生データのみで、スコアはapp.py内で計算します。")

    raw_df = load_raw_data()
    scored_df = compute_scores(raw_df)

    with st.expander("スコア計算ロジック", expanded=True):
        st.markdown(
            """
            **VisitIntent** = BaseScore × CreatorBiasCorrection  
            **BaseScore** = 0.35×InfoDensity + 0.30×CommentIntent + 0.20×SaveProxy + 0.15×AdjustedEngagement  
            **InboundGap** = ForeignReaction - JapaneseReaction を正規化  
            **TrendGrowth** = 直近30日平均InboundGap - 過去90日平均InboundGap  
            **PriorityScore** = 0.50×VisitIntent + 0.30×InboundGap + 0.20×TrendGrowth
            """
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("投稿数", len(scored_df))
    col2.metric("平均VisitIntent", f"{scored_df['VisitIntent'].mean():.3f}")
    col3.metric("平均PriorityScore", f"{scored_df['PriorityScore'].mean():.3f}")
    col4.metric("対象期間", f"{scored_df['post_date'].min().date()} - {scored_df['post_date'].max().date()}")

    st.header("1. 高PriorityScore投稿")
    st.dataframe(
        scored_df.sort_values("PriorityScore", ascending=False)[[
            "post_id", "post_date", "spot", "element_tag", "content_type",
            "VisitIntent", "InboundGapRaw", "TrendGrowthRaw", "PriorityScore"
        ]].head(15),
        use_container_width=True,
    )

    st.header("2. 観光要素別の優先度")
    element_summary = summarize_elements(scored_df)
    st.bar_chart(element_summary.set_index("element_tag")[["avg_priority"]])
    st.dataframe(element_summary, use_container_width=True)

    st.header("3. 国内外反応ギャップの時系列")
    scored_df["month"] = scored_df["post_date"].dt.to_period("M").astype(str)
    monthly_gap = (
        scored_df.groupby(["month", "element_tag"])["InboundGapRaw"]
        .mean()
        .reset_index()
        .pivot(index="month", columns="element_tag", values="InboundGapRaw")
    )
    st.line_chart(monthly_gap)

    st.header("4. いいね数とVisitIntentのズレ")
    st.caption("いいね数が多くても、情報量や行きたいコメントが弱い投稿はVisitIntentが低くなることを確認します。")
    st.scatter_chart(scored_df, x="likes", y="VisitIntent", color="content_type")
    st.dataframe(find_like_vs_intent_examples(scored_df), use_container_width=True)

    st.header("5. 重みの感度分析")
    st.caption("PriorityScoreの重みを変えても上位候補が大きく崩れないかを確認します。")
    ranking_df, stability_df = compute_sensitivity(scored_df)
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("シナリオ別Top3")
        st.dataframe(ranking_df, use_container_width=True)
    with col_b:
        st.subheader("Top3安定性")
        st.bar_chart(stability_df.set_index("element_tag")[["top3_count"]])
        st.dataframe(stability_df, use_container_width=True)

    st.header("6. データ連動のプロモーション戦略")
    for i, text in enumerate(make_strategy_text(scored_df), start=1):
        st.markdown(f"**提案{i}.** {text}")

    with st.expander("生データと算出後データを確認"):
        st.subheader("CSVの生データ（スコア列なし）")
        st.dataframe(raw_df.head(20), use_container_width=True)
        st.subheader("app.pyで算出したスコア付きデータ")
        st.dataframe(scored_df.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
