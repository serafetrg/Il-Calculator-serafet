import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from math import sqrt
from dataclasses import dataclass
from typing import List, Tuple

# =========================
# DATA CLASSES & LOGIC
# =========================

@dataclass
class LPInputs:
    deposit_value_usd: float
    sol_price: float
    lower_percent: float
    upper_percent: float

@dataclass
class LPState:
    lower_price: float
    upper_price: float
    liquidity_L: float
    amount_sol_init: float
    amount_usdc_init: float

@dataclass
class CompositionPoint:
    price: float
    side: str
    sol_delta: float
    sol_amount: float
    usdc_amount: float
    sol_pct: float
    usdc_pct: float
    lp_value: float
    hodl_value: float

def position_amounts_at_price(L: float, Pa: float, Pb: float, P: float) -> Tuple[float, float]:
    sqrtP = sqrt(P)
    sqrtPa = sqrt(Pa)
    sqrtPb = sqrt(Pb)

    if P <= Pa:
        amount_sol = L * (sqrtPb - sqrtPa) / (sqrtPa * sqrtPb)
        amount_usdc = 0.0
    elif P >= Pb:
        amount_sol = 0.0
        amount_usdc = L * (sqrtPb - sqrtPa)
    else:
        amount_sol = L * (sqrtPb - sqrtP) / (sqrtP * sqrtPb)
        amount_usdc = L * (sqrtP - sqrtPa)
    return amount_sol, amount_usdc

def init_lp_state(inputs: LPInputs) -> LPState:
    V = inputs.deposit_value_usd
    P = inputs.sol_price
    Pa = P * (1 - inputs.lower_percent / 100.0)
    Pb = P * (1 + inputs.upper_percent / 100.0)

    if Pa <= 0 or Pb <= 0: return None
    if not (Pa < P < Pb): return None

    sqrtP = sqrt(P)
    sqrtPa = sqrt(Pa)
    sqrtPb = sqrt(Pb)

    term = ((sqrtPb - sqrtP) / (sqrtP * sqrtPb)) * P + (sqrtP - sqrtPa)
    if term <= 0: return None

    L = V / term
    amount_sol_init = L * (sqrtPb - sqrtP) / (sqrtP * sqrtPb)
    amount_usdc_init = L * (sqrtP - sqrtPa)

    return LPState(Pa, Pb, L, amount_sol_init, amount_usdc_init)

def _compute_composition(lp_state: LPState, price: float, side_label: str) -> CompositionPoint:
    Pa, Pb, L = lp_state.lower_price, lp_state.upper_price, lp_state.liquidity_L
    sol_amt, usdc_amt = position_amounts_at_price(L, Pa, Pb, price)
    
    sol_value = sol_amt * price
    lp_val = sol_value + usdc_amt
    
    # Calculate HODL value at this specific price for comparison
    hodl_val = (lp_state.amount_sol_init * price) + lp_state.amount_usdc_init
    
    if lp_val == 0.0:
        sol_pct = usdc_pct = 0.0
    else:
        sol_pct = (sol_value / lp_val) * 100.0
        usdc_pct = (usdc_amt / lp_val) * 100.0

    return CompositionPoint(price, side_label, 0.0, sol_amt, usdc_amt, sol_pct, usdc_pct, lp_val, hodl_val)

def generate_data(lines: int, deposit: float, anchor: float, lower: float, upper: float):
    inputs = LPInputs(deposit, anchor, lower, upper)
    lp_state = init_lp_state(inputs)
    
    if not lp_state:
        return [], None

    Pa = lp_state.lower_price
    P = anchor
    sqrtPa, sqrtP = sqrt(Pa), sqrt(P)
    
    # Calculate interior points
    interior_count = lines - 2
    short_prices = []
    for i in range(interior_count):
        frac = (i + 1) / (interior_count + 1)
        sqrt_level = sqrtP - frac * (sqrtP - sqrtPa)
        price_level = sqrt_level ** 2
        if Pa < price_level < P:
            short_prices.append(price_level)

    # Combine and Dedupe
    raw_prices = [P] + short_prices + [Pa]
    uniq_prices = sorted(list(set([round(p, 4) for p in raw_prices])), reverse=True)
    
    points = []
    for price in uniq_prices:
        rp = round(price, 4)
        if rp == round(P, 4): side = "Anchor"
        elif rp == round(Pa, 4): side = "LowerBound"
        else: side = "Short"
        points.append(_compute_composition(lp_state, price, side))

    # Calculate Deltas
    prev_sol = None
    for i, pt in enumerate(points):
        if i == 0:
            pt.sol_delta = pt.sol_amount
        else:
            pt.sol_delta = pt.sol_amount - (prev_sol if prev_sol is not None else 0.0)
        prev_sol = pt.sol_amount
        
    return points, lp_state

def calculate_metrics(points, lp_state, deposit):
    if not points or not lp_state: return None
    
    lower_row = next((p for p in points if p.side == "LowerBound"), None)
    if not lower_row: return None
    
    Pa = lp_state.lower_price
    # Metric snapshot at Lower Bound
    hodl_value = lp_state.amount_sol_init * Pa + lp_state.amount_usdc_init
    lp_value = lower_row.lp_value
    il_usd = hodl_value - lp_value
    
    pnl_shorts = sum([pt.sol_delta * (pt.price - Pa) for pt in points])
    total_val = lp_value + pnl_shorts
    eff_pct = (total_val / deposit) * 100.0 if deposit else 0
    
    return {
        "hodl": hodl_value,
        "lp_val": lp_value,
        "il": il_usd,
        "short_pnl": pnl_shorts,
        "total": total_val,
        "eff": eff_pct
    }

def find_max_valid(deposit, anchor, lower, upper, min_notional, cap=100):
    max_valid = 0
    for lines in range(2, cap + 1):
        pts, _ = generate_data(lines, deposit, anchor, lower, upper)
        if pts:
            if all(abs(pt.sol_delta) * pt.price >= min_notional for pt in pts):
                max_valid = lines
    return max_valid

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="LP Hedging Calculator", layout="wide")

st.title("🛡️ LP Impermanent Loss & Hedge Calculator")
st.markdown("Optimal short distribution to hedge an Anchor -> Lower Bound LP position.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Configuration")
    
    with st.form(key='lp_config_form'):
        DEPOSIT = st.number_input("Deposit ($)", value=1000.0, step=100.0)
        ANCHOR = st.number_input("Anchor Price ($)", value=166.0, step=0.1, format="%.4f")
        
        col1, col2 = st.columns(2)
        with col1:
            LOWER_PCT = st.number_input("Lower Range (%)", value=10.0, step=0.5)
        with col2:
            UPPER_PCT = st.number_input("Upper Range (%)", value=10.0, step=0.5)
            
        LINES = st.number_input("Checkpoint Lines", value=30, min_value=2, step=1)
        
        st.divider()
        st.subheader("Exchange Constraints")
        MIN_NOTIONAL = st.number_input("Min Order ($)", value=10.0, step=1.0)
        
        # CONFIRM BUTTON
        submit_button = st.form_submit_button(label='Run Calculation', type="primary")

if 'first_load' not in st.session_state:
    st.session_state.first_load = True

if submit_button or st.session_state.first_load:
    st.session_state.first_load = False 

    # --- Calculation ---
    points, lp_state = generate_data(LINES, DEPOSIT, ANCHOR, LOWER_PCT, UPPER_PCT)

    if points and lp_state:
        metrics = calculate_metrics(points, lp_state, DEPOSIT)
        max_valid = find_max_valid(DEPOSIT, ANCHOR, LOWER_PCT, UPPER_PCT, MIN_NOTIONAL)

        # --- Validation Banner ---
        if LINES <= max_valid:
            st.success(f"✅ **Valid Configuration**: {LINES} lines is within the max limit of {max_valid} (based on ${MIN_NOTIONAL} min order).")
        else:
            st.warning(f"⚠️ **Order Size Warning**: Current lines ({LINES}) exceeds max valid ({max_valid}). Some orders are < ${MIN_NOTIONAL}.")

        # --- Key Metrics ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("HODL Value @ Lower", f"${metrics['hodl']:,.2f}")
        m2.metric("LP Value @ Lower", f"${metrics['lp_val']:,.2f}", delta=f"-${metrics['il']:,.2f} IL", delta_color="inverse")
        m3.metric("Short Hedge PnL", f"${metrics['short_pnl']:,.2f}")
        m4.metric("Total PnL (Hedge+LP)", f"${metrics['total']:,.2f}", f"{metrics['eff']:.2f}% Eff")

        # --- Data Prep for UI ---
        df_data = []
        for pt in points:
            df_data.append({
                "Side": pt.side,
                "Price": pt.price,
                "SOL Δ": pt.sol_delta,
                "SOL Amount": pt.sol_amount,
                "USDC Amount": pt.usdc_amount,
                "SOL %": pt.sol_pct,
                "USDC %": pt.usdc_pct,
                "LP Value ($)": pt.lp_value,
                # Extra hidden columns for charts/logic
                "Order Value": pt.sol_delta * pt.price,
                "HODL Value": pt.hodl_value,
                "USDC Value": pt.usdc_amount,
                "SOL Value": pt.sol_amount * pt.price
            })
        df = pd.DataFrame(df_data)

        # --- Tabs ---
        tab_grid, tab_charts = st.tabs(["📊 Data Grid", "📈 Visual Analysis"])

        with tab_grid:
            st.subheader("Execution Order Grid")
            
            # 1. Define exact columns from original tool
            cols_to_show = ["Side", "Price", "SOL Δ", "SOL Amount", "USDC Amount", "SOL %", "USDC %", "LP Value ($)"]

            # 2. Logic to highlight rows (re-calculating value on the fly since we aren't showing the column)
            def highlight_small_orders(row):
                val = row['SOL Δ'] * row['Price']
                color = '#ff4b4b20' if val < MIN_NOTIONAL else ''
                return [f'background-color: {color}' for _ in row]

            # 3. Format specifiers matching your python script
            st.dataframe(
                df[cols_to_show].style.apply(highlight_small_orders, axis=1).format({
                    "Price": "{:.4f}",
                    "SOL Δ": "{:.8f}",
                    "SOL Amount": "{:.8f}",
                    "USDC Amount": "{:.8f}",
                    "SOL %": "{:.2f}%",
                    "USDC %": "{:.2f}%",
                    "LP Value ($)": "{:.2f}",
                }),
                use_container_width=True,
                height=600
            )

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "lp_hedge_grid.csv", "text/csv")

        with tab_charts:
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.subheader("Asset Composition")
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=df['Price'], y=df['USDC Value'], mode='lines', stackgroup='one', name='USDC',
                    line=dict(width=0.5, color='#26a69a')
                ))
                fig_comp.add_trace(go.Scatter(
                    x=df['Price'], y=df['SOL Value'], mode='lines', stackgroup='one', name='SOL',
                    line=dict(width=0.5, color='#9c27b0')
                ))
                fig_comp.update_layout(
                    xaxis_title="Price (Descending)", 
                    yaxis_title="Value ($)", 
                    title="LP Value Breakdown (USDC vs SOL)",
                    xaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_comp, use_container_width=True)

            with col_c2:
                st.subheader("Hedge Order Sizing")
                fig_bar = px.bar(
                    df, x="Price", y="Order Value", 
                    color="Order Value",
                    title="Order Size per Checkpoint",
                    color_continuous_scale="Viridis"
                )
                fig_bar.update_layout(xaxis=dict(autorange="reversed"))
                fig_bar.add_hline(y=MIN_NOTIONAL, line_dash="dot", line_color="red", annotation_text="Min Limit")
                st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("Impermanent Loss Trajectory")
            fig_il = go.Figure()
            fig_il.add_trace(go.Scatter(x=df['Price'], y=df['HODL Value'], mode='lines', name='HODL', line=dict(dash='dash', color='gray')))
            fig_il.add_trace(go.Scatter(x=df['Price'], y=df['LP Value ($)'], mode='lines', name='LP Value', line=dict(color='orange')))
            fig_il.update_layout(
                xaxis_title="Price", 
                yaxis_title="Total Value ($)", 
                xaxis=dict(autorange="reversed"),
                hovermode="x unified"
            )
            st.plotly_chart(fig_il, use_container_width=True)

    else:
        st.error("Invalid Configuration. Please check price ranges.")
else:
    st.info("👈 Adjust settings in the sidebar and click **Run Calculation** to start.")
