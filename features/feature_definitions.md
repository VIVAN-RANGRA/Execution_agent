# Feature Definitions

## Feature 1: Spread-to-Volatility Ratio (Toxicity Proxy)
- Formula: `(ask_price - bid_price) / rolling_realized_vol_60s`
- Intuition: High value = liquidity providers pricing in adverse selection.
- Expected range: [0, ∞); typically 0–10
- Normalization: z-score

## Feature 2: Order Flow Imbalance (OFI)
- Formula: `(buyer_initiated_vol - seller_initiated_vol) / total_vol` over last 10s
- Intuition: Positive = buy pressure; negative = sell pressure.
- Expected range: [-1, 1]
- Normalization: none

## Feature 3: Volume Participation Rate
- Formula: `volume_in_last_30s / expected_volume_in_30s`
- Intuition: High = unusually active market; cheaper to hide in volume.
- Normalization: z-score

## Feature 4: Short-Term Price Momentum
- Formula: `(mid_price_now - mid_price_60s_ago) / mid_price_60s_ago * 10000` (bps)
- Intuition: Positive momentum for buy = price moving against you; be aggressive.
- Normalization: z-score

## Feature 5: Inventory Fraction Remaining
- Formula: `inventory_remaining / total_order_quantity`
- Expected range: [0, 1]
- Normalization: none

## Feature 6: Time Fraction Remaining
- Formula: `time_steps_remaining / total_time_steps`
- Expected range: [0, 1]
- Normalization: none

## Feature 7: Urgency Ratio (Inventory / Time)
- Formula: `inventory_fraction_remaining / (time_fraction_remaining + 1e-8)`
- Intuition: >1 = behind schedule; <1 = ahead of schedule.
- Normalization: clip to [0, 5], then z-score

## Feature 8: Realized Volatility Short (60s)
- Formula: `std(1s returns over last 60s) * sqrt(60)`
- Intuition: High = more timing risk; be aggressive.
- Normalization: z-score

## Feature 9: Realized Volatility Long (300s)
- Formula: `std(1s returns over last 300s) * sqrt(300)`
- Intuition: Baseline volatility regime for the session.
- Normalization: z-score

## Feature 10: Volatility Ratio (Short / Long)
- Formula: `vol_short_60s / (vol_long_300s + 1e-10)`
- Intuition: >1 = recent volatility spike; regime has changed.
- Normalization: z-score

## Feature 11: Best Bid Quantity (Normalized)
- Formula: `bid_qty / rolling_mean_bid_qty_60s`
- Intuition: Thick book = can execute passively without significant impact.
- Normalization: z-score

## Feature 12: Best Ask Quantity (Normalized)
- Formula: `ask_qty / rolling_mean_ask_qty_60s`
- Intuition: Same as above for the sell side.
- Normalization: z-score
