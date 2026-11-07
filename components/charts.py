"""Reusable Plotly chart builders using the GymMind template."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.theme import AMBER, GREEN, MUTED, PALETTE, PURPLE, RED, SEQ, TURQUOISE
from utils.formatting import CURRENCY


def show(fig: go.Figure, height: int = 340, key: str | None = None) -> None:
    fig.update_layout(height=height)
    st.plotly_chart(fig, theme=None, key=key, config={"displayModeBar": False, "displaylogo": False})


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def area_line(df: pd.DataFrame, x: str, y: str, title: str = "", color: str = TURQUOISE,
              name: str | None = None, money: bool = False, fill: bool = True) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[x], y=df[y], mode="lines+markers", name=name or y,
                               line=dict(color=color, width=2.6, shape="spline", smoothing=0.6),
                               marker=dict(size=5, color=color),
                               fill="tozeroy" if fill else None, fillcolor=_rgba(color, 0.12),
                               hovertemplate=(f"%{{x|%b %Y}}<br>{CURRENCY}%{{y:,.0f}}<extra></extra>" if money
                                              else "%{x|%b %Y}<br>%{y:,.2f}<extra></extra>")))
    fig.update_layout(title=title, showlegend=False)
    if money:
        fig.update_yaxes(tickprefix=CURRENCY, tickformat=",.0f")
    return fig


def multi_line(df: pd.DataFrame, x: str, ys: list[str], names: list[str] | None = None, title: str = "",
               colors: list[str] | None = None, money: bool = False, dash: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    for i, y in enumerate(ys):
        c = (colors or PALETTE)[i % len(colors or PALETTE)]
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", name=(names or ys)[i],
                                 line=dict(color=c, width=2.4, dash=(dash or ["solid"] * len(ys))[i])))
    fig.update_layout(title=title, hovermode="x unified")
    if money:
        fig.update_yaxes(tickprefix=CURRENCY, tickformat=",.0f")
    return fig


def stacked_bar(df: pd.DataFrame, title: str = "", money: bool = False, horizontal: bool = False,
                colors: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    for i, c in enumerate(df.columns):
        col = (colors or PALETTE)[i % len(colors or PALETTE)]
        if horizontal:
            fig.add_trace(go.Bar(y=df.index, x=df[c], name=str(c), orientation="h", marker_color=col))
        else:
            fig.add_trace(go.Bar(x=df.index, y=df[c], name=str(c), marker_color=col,
                                 marker_line_width=0))
    fig.update_layout(barmode="stack", title=title, hovermode="x unified")
    if money:
        (fig.update_xaxes if horizontal else fig.update_yaxes)(tickprefix=CURRENCY, tickformat=",.0f")
    return fig


def bar(x, y, title: str = "", color=PURPLE, horizontal: bool = False, money: bool = False,
        text_fmt: str | None = None, colors=None) -> go.Figure:
    kw = dict(marker=dict(color=colors if colors is not None else color, line=dict(width=0)))
    if text_fmt:
        kw.update(text=[text_fmt.format(v) for v in y], textposition="outside",
                  textfont=dict(color="#C9CEE6", size=11), cliponaxis=False)
    fig = go.Figure(go.Bar(x=y if horizontal else x, y=x if horizontal else y,
                           orientation="h" if horizontal else "v", **kw))
    fig.update_layout(title=title, showlegend=False)
    if horizontal:
        fig.update_yaxes(autorange="reversed")
    if money:
        (fig.update_xaxes if horizontal else fig.update_yaxes)(tickprefix=CURRENCY, tickformat=",.0f")
    return fig


def donut(labels, values, title: str = "", center: str = "", colors=None) -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.68, sort=False,
                           marker=dict(colors=colors or PALETTE, line=dict(color="#0B1020", width=3)),
                           textinfo="percent", textfont=dict(color="#fff", size=11),
                           hovertemplate="%{label}<br>%{value:,.0f} (%{percent})<extra></extra>"))
    fig.update_layout(title=title, legend=dict(orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle"),
                      annotations=[dict(text=center, showarrow=False, font=dict(size=16, color="#fff"))])
    return fig


def heatmap(z: pd.DataFrame, title: str = "", fmt: str = ".0f", colorscale=None, zmin=None, zmax=None,
            x_title: str = "", y_title: str = "", show_text: bool = True, hover_suffix: str = "") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z.to_numpy(), x=[str(c) for c in z.columns], y=[str(i) for i in z.index],
        colorscale=colorscale or SEQ, zmin=zmin, zmax=zmax, xgap=2, ygap=2,
        texttemplate=f"%{{z:{fmt}}}" if show_text else None, textfont=dict(size=10, color="#fff"),
        hovertemplate=f"%{{y}} · %{{x}}<br>%{{z:{fmt}}}{hover_suffix}<extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=MUTED))))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, type="category")
    return fig


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, test: pd.DataFrame | None = None,
                   title: str = "", money: bool = False, show_fitted: bool = False,
                   history_tail: int | None = None) -> go.Figure:
    """Actual data = solid turquoise, model predictions = dashed purple with a shaded band."""
    h = history.tail(history_tail) if history_tail else history
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["date"], y=h["actual"], name="Actual (historical)", mode="lines",
                             line=dict(color=TURQUOISE, width=2.4)))
    if show_fitted:
        fig.add_trace(go.Scatter(x=h["date"], y=h["fitted"], name="Model fit", mode="lines",
                                 line=dict(color=MUTED, width=1.2, dash="dot")))
    if test is not None and len(test):
        fig.add_trace(go.Scatter(x=test["date"], y=test["predicted"], name="Back-test prediction",
                                 mode="lines+markers", line=dict(color=AMBER, width=2, dash="dot"),
                                 marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=list(forecast["date"]) + list(forecast["date"][::-1]),
                             y=list(forecast["upper"]) + list(forecast["lower"][::-1]),
                             fill="toself", fillcolor=_rgba(PURPLE, 0.16), line=dict(width=0),
                             name="95% prediction interval", hoverinfo="skip"))
    # connect last actual to first prediction
    xs = [h["date"].iloc[-1]] + list(forecast["date"])
    ys = [h["actual"].iloc[-1]] + list(forecast["predicted"])
    fig.add_trace(go.Scatter(x=xs, y=ys, name="Forecast (predicted)", mode="lines+markers",
                             line=dict(color=PURPLE, width=2.6, dash="dash"), marker=dict(size=5)))
    x0 = h["date"].iloc[-1]
    fig.add_vline(x=x0, line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"))
    fig.add_annotation(x=x0, y=1.0, yref="paper", text="  FORECAST →", showarrow=False, xanchor="left",
                       font=dict(color=PURPLE, size=11, family="Inter"))
    fig.update_layout(title=title, hovermode="x unified")
    if money:
        fig.update_yaxes(tickprefix=CURRENCY, tickformat=",.0f")
    return fig


def gauge(value: float, title: str = "", suffix: str = "%", max_value: float = 100) -> go.Figure:
    color = GREEN if value >= 70 else (AMBER if value >= 50 else RED)
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, number=dict(suffix=suffix, font=dict(size=28, color="#fff")),
                                 gauge=dict(axis=dict(range=[0, max_value], tickcolor=MUTED),
                                            bar=dict(color=color, thickness=0.28),
                                            bgcolor="rgba(255,255,255,0.04)", borderwidth=0,
                                            steps=[dict(range=[0, max_value], color="rgba(255,255,255,0.03)")]),
                                 title=dict(text=title, font=dict(size=13, color=MUTED))))
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=10))
    return fig
