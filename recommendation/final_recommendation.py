

import logging
import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Any, List
from dataclasses import dataclass
from enum import Enum

from config.config import MODEL_NAME, GROQ_API_KEY, TEMPERATURE, TOP_N_RECOMMENDATIONS, DEBUG_RECOMMENDATION_LOGGING
from data.models import State
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

_llm = None
try:
    if GROQ_API_KEY and GROQ_API_KEY != "demo":
        from langchain_groq import ChatGroq
        _llm = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY, temperature=TEMPERATURE)
    else:
        logger.info("LLM not configured, using rule-based decisions")
except Exception as e:
    logger.warning(f"LLM initialization failed: {e}")
    _llm = None


class FactorType(Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    RISK = "risk"
    MACRO = "macro"
    MONTE_CARLO = "monte_carlo"
    BACKTEST = "backtest"


@dataclass
class FactorAnalysis:
    
    factor_type: FactorType
    strength: float  # -1 to 1, where 1 is strong buy signal, -1 is strong sell
    confidence: float  # 0 to 1, confidence in the signal
    weight: float  # Dynamic weight for this factor
    data: Dict[str, Any]  # Raw data used for analysis
    reasoning: str  # Brief explanation


@dataclass
class MarketConditions:
    
    volatility_regime: str  # "low", "medium", "high"
    trend_strength: str  # "weak", "moderate", "strong"
    market_sentiment: str  # "bearish", "neutral", "bullish"
    risk_environment: str  # "low_risk", "moderate_risk", "high_risk"


class EnhancedRecommendationEngine:
    

    def __init__(self):
        self.base_weights = {
            FactorType.TECHNICAL: 0.3,
            FactorType.FUNDAMENTAL: 0.15,
            FactorType.SENTIMENT: 0.20,
            FactorType.RISK: 0.15,
            FactorType.MACRO: 0.10,
            FactorType.MONTE_CARLO: 0.05,
            FactorType.BACKTEST: 0.04
        }

    def analyze_factors(self, symbol: str, state: State) -> List[FactorAnalysis]:
        
        factors = []

        # Get stock data for technical (df needed for enhancements)
        stock_data = state.get("stock_data", {})
        df = stock_data.get(symbol, pd.DataFrame()) if symbol in stock_data else pd.DataFrame()

        # Technical Analysis (enhanced with df)
        technical = state.get("technical_signals", {}).get(symbol, {})
        technical_factor = self._analyze_technical_factor(technical, df)
        factors.append(technical_factor)
        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"[{symbol}] Technical factor: strength={technical_factor.strength:.3f}, confidence={technical_factor.confidence:.3f}, weight={technical_factor.weight:.3f}")

        # Fundamental Analysis
        fundamental = state.get("fundamental_analysis", {}).get(symbol, {})
        fundamental_factor = self._analyze_fundamental_factor(fundamental)
        factors.append(fundamental_factor)
        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"[{symbol}] Fundamental factor: strength={fundamental_factor.strength:.3f}, confidence={fundamental_factor.confidence:.3f}, weight={fundamental_factor.weight:.3f}")

        # Sentiment Analysis
        sentiment = state.get("sentiment_scores", {}).get(symbol, {})
        sentiment_factor = self._analyze_sentiment_factor(sentiment)
        factors.append(sentiment_factor)
        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"[{symbol}] Sentiment factor: strength={sentiment_factor.strength:.3f}, confidence={sentiment_factor.confidence:.3f}, weight={sentiment_factor.weight:.3f}")

        # Risk Assessment
        risk = state.get("risk_metrics", {}).get(symbol, {})
        risk_factor = self._analyze_risk_factor(risk)
        factors.append(risk_factor)
        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"[{symbol}] Risk factor: strength={risk_factor.strength:.3f}, confidence={risk_factor.confidence:.3f}, weight={risk_factor.weight:.3f}")

        # Macro Analysis
        macro_scores = state.get("macro_scores", {})
        macro = macro_scores.get("composite", 0.0) if "error" not in macro_scores else 0.0
        macro_factor = self._analyze_macro_factor(macro, macro_scores)
        factors.append(macro_factor)
        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"[{symbol}] Macro factor: strength={macro_factor.strength:.3f}, confidence={macro_factor.confidence:.3f}, weight={macro_factor.weight:.3f}")

        # Monte Carlo Simulation
        simulation_results = state.get("simulation_results", {})
        monte_carlo_factor = self._analyze_monte_carlo_factor(simulation_results)
        factors.append(monte_carlo_factor)
        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"[{symbol}] Monte Carlo factor: strength={monte_carlo_factor.strength:.3f}, confidence={monte_carlo_factor.confidence:.3f}, weight={monte_carlo_factor.weight:.3f}")

        # Backtest Results
        backtest_results = state.get("backtest_results", {})
        backtest_factor = self._analyze_backtest_factor(backtest_results)
        factors.append(backtest_factor)
        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"[{symbol}] Backtest factor: strength={backtest_factor.strength:.3f}, confidence={backtest_factor.confidence:.3f}, weight={backtest_factor.weight:.3f}")

        return factors

    def assess_market_conditions(self, factors: List[FactorAnalysis]) -> MarketConditions:
        
        # Determine volatility regime
        risk_factors = [f for f in factors if f.factor_type == FactorType.RISK]
        if risk_factors:
            risk_strength = risk_factors[0].strength
            if abs(risk_strength) > 0.7:
                volatility_regime = "high"
            elif abs(risk_strength) > 0.3:
                volatility_regime = "medium"
            else:
                volatility_regime = "low"
        else:
            volatility_regime = "medium"

        # Determine trend strength
        technical_factors = [f for f in factors if f.factor_type == FactorType.TECHNICAL]
        if technical_factors:
            tech_confidence = technical_factors[0].confidence
            if tech_confidence > 0.8:
                trend_strength = "strong"
            elif tech_confidence > 0.5:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"
        else:
            trend_strength = "moderate"

        # Determine market sentiment
        sentiment_factors = [f for f in factors if f.factor_type == FactorType.SENTIMENT]
        macro_factors = [f for f in factors if f.factor_type == FactorType.MACRO]
        combined_sentiment = 0.0
        count = 0
        for factor in sentiment_factors + macro_factors:
            combined_sentiment += factor.strength
            count += 1
        if count > 0:
            avg_sentiment = combined_sentiment / count
            if avg_sentiment > 0.3:
                market_sentiment = "bullish"
            elif avg_sentiment < -0.3:
                market_sentiment = "bearish"
            else:
                market_sentiment = "neutral"
        else:
            market_sentiment = "neutral"

        # Determine risk environment
        risk_factors = [f for f in factors if f.factor_type in [FactorType.RISK, FactorType.MONTE_CARLO]]
        if risk_factors:
            avg_risk = sum(f.strength for f in risk_factors) / len(risk_factors)
            if avg_risk > 0.5:
                risk_environment = "low_risk"
            elif avg_risk > -0.5:
                risk_environment = "moderate_risk"
            else:
                risk_environment = "high_risk"
        else:
            risk_environment = "moderate_risk"

        return MarketConditions(
            volatility_regime=volatility_regime,
            trend_strength=trend_strength,
            market_sentiment=market_sentiment,
            risk_environment=risk_environment
        )

    def calculate_dynamic_weights(self, factors: List[FactorAnalysis], market_conditions: MarketConditions) -> Dict[FactorType, float]:
        
        weights = self.base_weights.copy()

        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"Base weights: {', '.join([f'{k.value}: {v:.3f}' for k, v in weights.items()])}")
            logger.debug(f"Market conditions: volatility={market_conditions.volatility_regime}, trend={market_conditions.trend_strength}, sentiment={market_conditions.market_sentiment}, risk={market_conditions.risk_environment}")

        # Adjust weights based on volatility regime
        if market_conditions.volatility_regime == "high":
            # In high volatility, reduce weight of momentum indicators, increase risk weight
            weights[FactorType.TECHNICAL] *= 0.8
            weights[FactorType.RISK] *= 1.3
            weights[FactorType.MONTE_CARLO] *= 1.5
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug("High volatility adjustments: Technical -20%, Risk +30%, Monte Carlo +50%")
        elif market_conditions.volatility_regime == "low":
            # In low volatility, increase technical weight
            weights[FactorType.TECHNICAL] *= 1.2
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug("Low volatility adjustment: Technical +20%")

        # Adjust based on trend strength
        if market_conditions.trend_strength == "strong":
            # Strong trends favor technical analysis
            weights[FactorType.TECHNICAL] *= 1.1
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug("Strong trend adjustment: Technical +10%")
        elif market_conditions.trend_strength == "weak":
            # Weak trends favor fundamental and sentiment
            weights[FactorType.FUNDAMENTAL] *= 1.2
            weights[FactorType.SENTIMENT] *= 1.1
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug("Weak trend adjustments: Fundamental +20%, Sentiment +10%")

        # Adjust based on market sentiment
        if market_conditions.market_sentiment in ["bullish", "bearish"]:
            # Strong sentiment favors momentum factors
            weights[FactorType.SENTIMENT] *= 1.1
            weights[FactorType.MACRO] *= 1.1
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug("Strong sentiment adjustments: Sentiment +10%, Macro +10%")
        else:
            # Neutral sentiment favors fundamental analysis
            weights[FactorType.FUNDAMENTAL] *= 1.1
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug("Neutral sentiment adjustment: Fundamental +10%")

        # Adjust based on risk environment
        if market_conditions.risk_environment == "high_risk":
            weights[FactorType.RISK] *= 1.4
            weights[FactorType.BACKTEST] *= 1.3
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug("High risk adjustments: Risk +40%, Backtest +30%")
        elif market_conditions.risk_environment == "low_risk":
            weights[FactorType.TECHNICAL] *= 1.1
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug("Low risk adjustment: Technical +10%")

        # Factor alignment adjustment - reduce weight of conflicting factors
        factor_strengths = {f.factor_type: f.strength for f in factors}
        consensus_score = self._calculate_factor_consensus(factor_strengths)
        if consensus_score > 0.7:  # High consensus
            # Increase weight of technical and sentiment factors
            weights[FactorType.TECHNICAL] *= 1.2
            weights[FactorType.SENTIMENT] *= 1.2
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug("High consensus adjustment: Technical +20%, Sentiment +20%")
        elif consensus_score < 0.5:  # Low consensus
            # Increase weight of high-confidence factors
            for factor in factors:
                if factor.confidence > 0.7:
                    weights[factor.factor_type] *= 1.1
                    if DEBUG_RECOMMENDATION_LOGGING:
                        logger.debug(f"Low consensus adjustment: {factor.factor_type.value} +10% (confidence: {factor.confidence:.3f})")

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"Final normalized weights: {', '.join([f'{k.value}: {v:.3f}' for k, v in weights.items()])}")

        return weights

    def synthesize_decision(self, factors: List[FactorAnalysis], weights: Dict[FactorType, float]) -> Dict[str, Any]:
        
        # Calculate weighted composite score
        composite_score = 0.0
        total_weight = 0.0
        factor_contributions = {}

        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug("Calculating composite score from factor contributions:")

        for factor in factors:
            weight = weights.get(factor.factor_type, 0.0)
            contribution = factor.strength * weight
            composite_score += contribution
            total_weight += weight
            factor_contributions[factor.factor_type.value] = {
                'strength': factor.strength,
                'weight': weight,
                'contribution': contribution,
                'confidence': factor.confidence
            }
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"  {factor.factor_type.value}: strength={factor.strength:.3f} * weight={weight:.3f} = contribution={contribution:.3f}")

        if total_weight == 0:
            composite_score = 0.0
        else:
            composite_score = composite_score / total_weight

        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"Composite score calculation: total_contribution={composite_score * total_weight:.3f} / total_weight={total_weight:.3f} = {composite_score:.3f}")

        # Apply aggressive BUY boosts
        original_score = composite_score

        # Strong signal boost: +0.1 if any factor > 0.8
        strong_signals = [f for f in factors if f.strength > 0.8]
        if strong_signals:
            composite_score += 0.1
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Strong signal boost applied: +0.1 (factors: {[f.factor_type.value for f in strong_signals]})")

        # Momentum boost: +0.05 if technical and sentiment are both positive
        technical_factor = next((f for f in factors if f.factor_type == FactorType.TECHNICAL), None)
        sentiment_factor = next((f for f in factors if f.factor_type == FactorType.SENTIMENT), None)
        if technical_factor and sentiment_factor and technical_factor.strength > 0 and sentiment_factor.strength > 0:
            composite_score += 0.05
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug("Momentum boost applied: +0.05 (technical and sentiment both positive)")

        # Count positive factors for BUY if positive logic
        positive_factors = sum(1 for f in factors if f.strength > 0)

        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"Aggressive boosts applied: original={original_score:.3f}, boosted={composite_score:.3f}, positive_factors={positive_factors}")

        # Determine action with updated thresholds to reduce HOLD bias
        if composite_score > 0.1 or (composite_score > 0.05 and positive_factors >= 3):
            action = "BUY"
        elif composite_score < -0.1:
            action = "SELL"
        else:
            action = "HOLD"

        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"Decision threshold comparison: score={composite_score:.3f}, BUY_threshold=0.4, SELL_threshold=-0.6 -> action={action}")

        # Calculate confidence based on factor consensus and Monte Carlo validation
        confidence = self._calculate_confidence(factors, composite_score, weights)

        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"Final decision: action={action}, composite_score={composite_score:.3f}, confidence={confidence:.3f}")

        return {
            'action': action,
            'composite_score': composite_score,
            'confidence': confidence,
            'factor_contributions': factor_contributions,
            'decision_reasoning': self._generate_decision_reasoning(action, composite_score, factors, weights)
        }

    def _calculate_volatility_and_trend(self, df: pd.DataFrame) -> Dict[str, float]:
        
        if len(df) < 20:
            return {'volatility': 0.02, 'trend_strength': 0.5}  # Defaults
        
        # ATR for volatility
        high = df['High']
        low = df['Low']
        close = df['Close']
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        volatility = atr / close.iloc[-1] if close.iloc[-1] > 0 else 0.02
        
        # Simple trend strength (EMA slope + volume trend)
        ema20 = close.ewm(span=20).mean()
        slope = (ema20.iloc[-1] - ema20.iloc[-2]) / ema20.iloc[-2] if len(ema20) >= 2 else 0
        vol_trend = df['Volume'].rolling(20).mean().pct_change().iloc[-1]
        trend_strength = abs(slope) + abs(vol_trend)  # Normalized later
        trend_strength = min(1.0, trend_strength * 10)  # Scale
        
        return {'volatility': volatility, 'trend_strength': trend_strength}
    
    def _get_indicator_performance_weights(self, indicator: str) -> float:
        
        performance_weights = {
            'RSI': 1.0, 'MACD': 1.1, 'SMA': 0.9, 'EMA': 0.95,
            'Bollinger': 0.9, 'Stochastic': 1.0, 'WilliamsR': 1.05,
            'CCI': 1.05, 'OBV': 1.0, 'VWAP': 1.1, 'PivotPoints': 1.05,
            'Ichimoku': 1.1, 'Fibonacci': 0.95, 'SupportResistance': 1.0,
            'MLSignal': 1.2, 'VPVR': 1.0, 'HeikinAshi': 0.95,
            'Harmonic': 1.15, 'HMM': 1.1, 'LSTM': 1.25, 'Patterns': 1.1
        }
        return performance_weights.get(indicator, 1.0)
    
    def calculate_dynamic_indicator_weights(self, indicators: List[str], volatility: float, trend_strength: float, is_trending: bool = True) -> Dict[str, float]:
        
        weights = {}
        base_vol_adj = 0.8 if volatility > 0.03 else 1.2 if volatility < 0.01 else 1.0  # Reduce in high vol, boost low
        base_trend_adj = 1.2 if trend_strength > 0.7 else 0.8 if trend_strength < 0.3 else 1.0
        
        for ind in indicators:
            perf_weight = self._get_indicator_performance_weights(ind)
            context_adj = 1.2 if (is_trending and ind in ['MACD', 'EMA', 'SMA']) else 1.2 if (not is_trending and ind in ['RSI', 'Stochastic', 'WilliamsR', 'CCI', 'Bollinger']) else 1.0
            weights[ind] = perf_weight * base_vol_adj * base_trend_adj * context_adj
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"Dynamic weights: vol_adj={base_vol_adj:.2f}, trend_adj={base_trend_adj:.2f}, is_trending={is_trending}, weights={weights}")
        
        return weights
    
    def assess_signal_quality(self, technical: Dict[str, Any], recency_threshold: int = 5) -> Dict[str, Any]:
        
        quality_signals = {}
        conflicting = set()
        recent_signals = {k: v for k, v in technical.items() if isinstance(v, str) and k.endswith(('_daily', '_4h', '_1h'))}  # Assume recent TFs
        
        # Identify conflicts (opposite signals from same indicator family)
        for ind in ['RSI', 'MACD', 'Stochastic']:
            buy_keys = [k for k, v in technical.items() if ind.lower() in k.lower() and v == 'buy']
            sell_keys = [k for k, v in technical.items() if ind.lower() in k.lower() and v == 'sell']
            if buy_keys and sell_keys:
                conflicting.update(buy_keys + sell_keys)
                if DEBUG_RECOMMENDATION_LOGGING:
                    logger.debug(f"Conflicting {ind} signals filtered: {buy_keys + sell_keys}")
        
        # Filter low-strength (e.g., neutral or weak) and conflicts
        for key, value in technical.items():
            if isinstance(value, str) and value.lower() in ['buy', 'sell', 'neutral', 'hold']:
                if key in conflicting or value == 'neutral':
                    quality_signals[key] = 'filtered'  # Mark as filtered
                else:
                    # Recency boost if recent TF
                    recency_weight = 1.2 if key in recent_signals else 1.0
                    quality_signals[key] = {'signal': value, 'recency_weight': recency_weight}
            elif isinstance(value, dict) and 'strength' in value:  # For aggregated
                if abs(value['strength']) < 0.3:  # Low strength threshold
                    quality_signals[key] = 'filtered'
                else:
                    quality_signals[key] = value
        
        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"Signal quality: filtered {len([k for k,v in quality_signals.items() if v == 'filtered'])}, enhanced {len(quality_signals) - len([k for k,v in quality_signals.items() if v == 'filtered'])}")
        
        return quality_signals
    
    def calculate_consensus_confidence(self, quality_signals: Dict[str, Any], multi_tf_consistency: float = 0.0, pattern_reliability: float = 1.0, market_vol: float = 0.02) -> float:
        
        buy_count = sum(1 for v in quality_signals.values() if isinstance(v, dict) and v.get('signal') == 'buy')
        sell_count = sum(1 for v in quality_signals.values() if isinstance(v, dict) and v.get('signal') == 'sell')
        total = buy_count + sell_count
        consensus = abs(buy_count - sell_count) / total if total > 0 else 0.5
        
        # Multi-TF alignment boost
        tf_boost = 1 + 0.3 * multi_tf_consistency if multi_tf_consistency > 0 else 1.0
        
        # Pattern reliability adjustment (if patterns present)
        pattern_adj = pattern_reliability if pattern_reliability < 1.0 else 1.0
        
        # Market condition modifiers (vol: reduce in high vol)
        vol_modifier = 0.8 if market_vol > 0.03 else 1.2 if market_vol < 0.01 else 1.0
        
        # Recency from quality_signals
        recency_factor = np.mean([v.get('recency_weight', 1.0) for v in quality_signals.values() if isinstance(v, dict)]) if any(isinstance(v, dict) for v in quality_signals.values()) else 1.0
        
        confidence = consensus * tf_boost * pattern_adj * vol_modifier * recency_factor
        confidence = min(1.0, max(0.0, confidence))
        
        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"Consensus confidence: consensus={consensus:.2f}, tf_boost={tf_boost:.2f}, pattern_adj={pattern_adj:.2f}, vol_mod={vol_modifier:.2f}, recency={recency_factor:.2f} -> {confidence:.2f}")
        
        return confidence
    
    def ensemble_technical_signals(self, quality_signals: Dict[str, Any], dynamic_weights: Dict[str, float], market_vol: float, trend_strength: float) -> Dict[str, Any]:
        
        score = 0.0
        total_weight = 0.0
        votes = {'buy': 0, 'sell': 0}
        
        for key, val in quality_signals.items():
            if isinstance(val, dict) and 'signal' in val:
                signal = val['signal'].lower()
                weight = dynamic_weights.get(key.split('_')[0] if '_' in key else key, 0.0)  # Base key
                weight *= val.get('recency_weight', 1.0)
                if signal == 'buy':
                    score += weight * 1.0
                    votes['buy'] += 1
                elif signal == 'sell':
                    score += weight * -1.0
                    votes['sell'] += 1
                total_weight += weight
        
        # Robustness: if extreme (high vol + strong trend opposite), adjust
        extreme_adj = 0.0
        if market_vol > 0.04 and trend_strength > 0.8:
            if score > 0.5 and votes['sell'] > votes['buy']:  # Contrarian extreme
                extreme_adj = -0.2
            elif score < -0.5 and votes['buy'] > votes['sell']:
                extreme_adj = 0.2
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Extreme condition adjustment: {extreme_adj:.2f} (vol={market_vol:.3f}, trend={trend_strength:.2f})")
        
        strength = max(-1.0, min(1.0, (score / total_weight if total_weight > 0 else 0) + extreme_adj))
        dominant = 'buy' if votes['buy'] > votes['sell'] else 'sell' if votes['sell'] > votes['buy'] else 'neutral'
        
        return {'strength': strength, 'dominant_signal': dominant, 'votes': votes, 'total_weight': total_weight}
    
    def _aggregate_multi_timeframe_signals(self, technical: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        
        timeframes = ['monthly', 'weekly', 'daily']
        timeframe_weights = {'monthly': 0.4, 'weekly': 0.35, 'daily': 0.25}
        
        # Group signals by base indicator
        indicator_groups = {}
        for key, value in technical.items():
            if isinstance(value, str) and value.lower() in ['buy', 'sell', 'neutral', 'hold']:
                # Extract base indicator and timeframe
                base_key = key
                tf = 'daily'  # Default
                for t in timeframes:
                    if key.endswith(f'_{t}'):
                        base_key = key[:-len(f'_{t}')]
                        tf = t
                        break
                
                if base_key not in indicator_groups:
                    indicator_groups[base_key] = {}
                indicator_groups[base_key][tf] = value.lower()
        
        # Calculate market metrics from df (assuming df available; pass if needed)
        market_metrics = self._calculate_volatility_and_trend(df)
        volatility = market_metrics['volatility']
        trend_strength = market_metrics['trend_strength']
        is_trending = trend_strength > 0.5
        
        # Get all unique indicators for weighting
        all_indicators = list(indicator_groups.keys())
        if 'Patterns' in technical:  # Assume patterns if present
            all_indicators.append('Patterns')
        dynamic_weights = self.calculate_dynamic_indicator_weights(all_indicators, volatility, trend_strength, is_trending)
        
        aggregated = {}
        total_consistency = 0
        num_indicators = 0
        quality_signals = self.assess_signal_quality(technical)
        
        for indicator, tf_signals in indicator_groups.items():
            available_tfs = list(tf_signals.keys())
            if len(available_tfs) == 0:
                continue
            
            # Signal values: buy=1, sell=-1, neutral=0
            signal_values = {'buy': 1, 'sell': -1, 'neutral': 0, 'hold': 0}
            weighted_strength = 0
            agreement_count = 0
            dominant_signal = None
            signal_counts = {'buy': 0, 'sell': 0, 'neutral': 0}
            tf_qualities = []
            
            for tf in available_tfs:
                signal = tf_signals[tf]
                signal_counts[signal] += 1
                tf_key = f"{indicator}_{tf}"
                quality = quality_signals.get(tf_key, {'signal': signal, 'recency_weight': 1.0})
                if quality != 'filtered':
                    adj_weight = timeframe_weights.get(tf, 0.25) * quality.get('recency_weight', 1.0)
                    weighted_strength += signal_values.get(signal, 0) * adj_weight
                    tf_qualities.append(quality)
                else:
                    weighted_strength += 0  # Filtered
            
            # Ensemble per indicator
            ind_ensemble = self.ensemble_technical_signals({tf: q for tf, q in zip(available_tfs, tf_qualities)}, {indicator: dynamic_weights.get(indicator, 0.1)}, volatility, trend_strength)
            strength = ind_ensemble['strength']
            
            # Determine dominant signal
            max_count = max(signal_counts.values())
            for sig, count in signal_counts.items():
                if count == max_count:
                    dominant_signal = sig
                    break
            
            # Enhanced consistency with quality
            valid_tfs = len([q for q in tf_qualities if q != 'filtered'])
            consistency = (signal_counts.get(dominant_signal, 0) / len(available_tfs)) if len(available_tfs) > 0 else 0
            if valid_tfs > 0:
                consistency *= (valid_tfs / len(available_tfs))  # Quality factor
            
            aggregated[indicator] = {
                'strength': strength,
                'dominant_signal': dominant_signal,
                'consistency': consistency,
                'available_tfs': available_tfs,
                'ensemble_votes': ind_ensemble['votes']
            }
            
            total_consistency += consistency
            num_indicators += 1
        
        # Overall ensemble
        overall_quality = self.ensemble_technical_signals(quality_signals, dynamic_weights, volatility, trend_strength)
        overall_consistency = total_consistency / num_indicators if num_indicators > 0 else 0
        overall_strength = overall_quality['strength']
        
        # Enhanced overall confidence
        pattern_rel = technical.get('Patterns', {}).get('confidence', 1.0) if 'Patterns' in technical else 1.0
        overall_conf = self.calculate_consensus_confidence(quality_signals, overall_consistency, pattern_rel, volatility)
        
        return {
            'aggregated_signals': aggregated,
            'overall_strength': max(-1.0, min(1.0, overall_strength)),
            'overall_consistency': overall_consistency,
            'num_indicators': num_indicators,
            'overall_confidence': overall_conf,
            'quality_signals': quality_signals,
            'dynamic_weights': dynamic_weights
        }

    def _analyze_technical_factor(self, technical: Dict[str, Any], df: pd.DataFrame) -> FactorAnalysis:
        
        if 'error' in technical and isinstance(technical['error'], str):
            return FactorAnalysis(
                factor_type=FactorType.TECHNICAL,
                strength=0.0,
                confidence=0.0,
                weight=self.base_weights[FactorType.TECHNICAL],
                data=technical,
                reasoning="Technical analysis unavailable"
            )

        # Check for multi-timeframe signals
        has_multi_tf = any(key.endswith('_daily') or key.endswith('_weekly') or key.endswith('_monthly') for key in technical.keys())
        
        if has_multi_tf:
            # Use multi-timeframe aggregation
            agg_results = self._aggregate_multi_timeframe_signals(technical, df)
            strength = agg_results['overall_strength']
            consistency = agg_results['overall_consistency']
            num_indicators = agg_results['num_indicators']
            
            # Confidence: base from number of indicators + consistency bonus
            base_confidence = min(1.0, num_indicators / 15)  # Adjusted for more indicators
            confidence = base_confidence * (1 + 0.3 * consistency)  # Up to 30% boost for high consistency
            confidence = min(1.0, confidence)
            
            reasoning = f"Multi-TF Technical: {num_indicators} indicators, consistency {consistency:.2f}, strength {strength:.2f}"
        else:
            # Backward compatibility: original single-timeframe logic
            buy_signals = 0
            sell_signals = 0
            total_signals = 0
            advanced_indicators = ['Ichimoku', 'Fibonacci', 'MLSignal', 'VWAP', 'PivotPoints', 'WilliamsR', 'CCI', 'OBV']

            for key, value in technical.items():
                if isinstance(value, str) and value.lower() in ['buy', 'sell', 'hold']:
                    total_signals += 1
                    if value.lower() == 'buy':
                        buy_signals += 1
                    elif value.lower() == 'sell':
                        sell_signals += 1

            # Advanced indicators get higher weight
            advanced_buy = sum(1 for ind in advanced_indicators if technical.get(ind, '').lower() == 'buy')
            advanced_sell = sum(1 for ind in advanced_indicators if technical.get(ind, '').lower() == 'sell')

            # Calculate strength
            if total_signals > 0:
                net_signals = (buy_signals + advanced_buy * 2) - (sell_signals + advanced_sell * 2)
                strength = net_signals / (total_signals + len(advanced_indicators) * 2)
                strength = max(-1.0, min(1.0, strength))
            else:
                strength = 0.0

            # Calculate confidence based on signal agreement
            confidence = min(1.0, total_signals / 15)  # Adjusted denominator for more indicators

            reasoning = f"Technical: {buy_signals} buy, {sell_signals} sell signals. Advanced: {advanced_buy} buy, {advanced_sell} sell."

        return FactorAnalysis(
            factor_type=FactorType.TECHNICAL,
            strength=strength,
            confidence=confidence,
            weight=self.base_weights[FactorType.TECHNICAL],
            data=technical,
            reasoning=reasoning
        )

    def _analyze_fundamental_factor(self, fundamental: Dict[str, Any]) -> FactorAnalysis:
        
        if 'error' in fundamental and isinstance(fundamental['error'], str):
            return FactorAnalysis(
                factor_type=FactorType.FUNDAMENTAL,
                strength=0.0,
                confidence=0.0,
                weight=self.base_weights[FactorType.FUNDAMENTAL],
                data=fundamental,
                reasoning="Fundamental analysis unavailable"
            )

        valuations = fundamental.get('valuations', '')
        if valuations == 'undervalued':
            strength = 1.0
            confidence = 0.8
            reasoning = "Fundamentals indicate undervaluation"
        elif valuations == 'overvalued':
            strength = -1.0
            confidence = 0.8
            reasoning = "Fundamentals indicate overvaluation"
        else:
            strength = 0.0
            confidence = 0.3
            reasoning = "Fundamental analysis neutral or inconclusive"

        return FactorAnalysis(
            factor_type=FactorType.FUNDAMENTAL,
            strength=strength,
            confidence=confidence,
            weight=self.base_weights[FactorType.FUNDAMENTAL],
            data=fundamental,
            reasoning=reasoning
        )

    def _analyze_sentiment_factor(self, sentiment: Dict[str, Any]) -> FactorAnalysis:
        
        if 'error' in sentiment and isinstance(sentiment['error'], str):
            return FactorAnalysis(
                factor_type=FactorType.SENTIMENT,
                strength=0.0,
                confidence=0.0,
                weight=self.base_weights[FactorType.SENTIMENT],
                data=sentiment,
                reasoning="Sentiment analysis unavailable"
            )

        compound = sentiment.get('compound', 0)
        if compound > 0.1:
            strength = min(1.0, compound * 2)
            confidence = min(1.0, abs(compound) * 2)
            reasoning = f"Positive sentiment (compound: {compound:.2f})"
        elif compound < -0.1:
            strength = max(-1.0, compound * 2)
            confidence = min(1.0, abs(compound) * 2)
            reasoning = f"Negative sentiment (compound: {compound:.2f})"
        else:
            strength = 0.0
            confidence = 0.5
            reasoning = f"Neutral sentiment (compound: {compound:.2f})"

        return FactorAnalysis(
            factor_type=FactorType.SENTIMENT,
            strength=strength,
            confidence=confidence,
            weight=self.base_weights[FactorType.SENTIMENT],
            data=sentiment,
            reasoning=reasoning
        )

    def _analyze_risk_factor(self, risk: Dict[str, Any]) -> FactorAnalysis:
        
        if 'error' in risk and isinstance(risk['error'], str):
            return FactorAnalysis(
                factor_type=FactorType.RISK,
                strength=0.0,
                confidence=0.0,
                weight=self.base_weights[FactorType.RISK],
                data=risk,
                reasoning="Risk analysis unavailable"
            )

        risk_ok = risk.get('risk_ok', True)
        volatility = risk.get('volatility', 0)
        sharpe = risk.get('sharpe_ratio', 0)

        # Calculate risk score
        risk_score = 0.0
        if not risk_ok:
            risk_score = -1.0
        elif volatility > 0.4:
            risk_score = -0.5
        elif volatility > 0.2:
            risk_score = -0.3
        elif sharpe > 1.0:
            risk_score = 0.5
        elif sharpe > 0.5:
            risk_score = 0.3

        confidence = 0.8 if 'volatility' in risk and 'sharpe_ratio' in risk else 0.5
        reasoning = f"Risk: {'OK' if risk_ok else 'High'}, Vol: {volatility:.1f}, Sharpe: {sharpe:.2f}"

        return FactorAnalysis(
            factor_type=FactorType.RISK,
            strength=risk_score,
            confidence=confidence,
            weight=self.base_weights[FactorType.RISK],
            data=risk,
            reasoning=reasoning
        )

    def _analyze_macro_factor(self, macro: float, macro_scores: Dict[str, Any]) -> FactorAnalysis:
        
        if "error" in macro_scores:
            return FactorAnalysis(
                factor_type=FactorType.MACRO,
                strength=0.0,
                confidence=0.0,
                weight=self.base_weights[FactorType.MACRO],
                data=macro_scores,
                reasoning="Macro analysis unavailable"
            )

        strength = macro
        confidence = 0.7  # Macro analysis typically has moderate confidence
        reasoning = f"Macro economic score: {macro:.2f}"

        return FactorAnalysis(
            factor_type=FactorType.MACRO,
            strength=strength,
            confidence=confidence,
            weight=self.base_weights[FactorType.MACRO],
            data=macro_scores,
            reasoning=reasoning
        )

    def _analyze_monte_carlo_factor(self, simulation_results: Dict[str, Any]) -> FactorAnalysis:
        
        if not simulation_results or "error" in simulation_results:
            return FactorAnalysis(
                factor_type=FactorType.MONTE_CARLO,
                strength=0.0,
                confidence=0.0,
                weight=self.base_weights[FactorType.MONTE_CARLO],
                data=simulation_results,
                reasoning="Monte Carlo simulation unavailable"
            )

        # Extract key metrics
        expected_return = simulation_results.get('expected_return', 0)
        var_95 = simulation_results.get('var_95', 0)
        max_drawdown = simulation_results.get('max_drawdown', 0)

        # Calculate strength based on risk-adjusted returns
        if expected_return > 0.02 and abs(var_95) < 0.1 and max_drawdown < 0.15:
            strength = 0.8
        elif expected_return > 0.01 and abs(var_95) < 0.15:
            strength = 0.4
        elif expected_return < -0.02 or abs(var_95) > 0.2 or max_drawdown > 0.25:
            strength = -0.8
        elif expected_return < 0:
            strength = -0.4
        else:
            strength = 0.0

        confidence = 0.6  # Monte Carlo provides probabilistic insights
        reasoning = f"MC: ExpRet {expected_return:.1%}, VaR95 {var_95:.1%}, MaxDD {max_drawdown:.1%}"

        return FactorAnalysis(
            factor_type=FactorType.MONTE_CARLO,
            strength=strength,
            confidence=confidence,
            weight=self.base_weights[FactorType.MONTE_CARLO],
            data=simulation_results,
            reasoning=reasoning
        )

    def _analyze_backtest_factor(self, backtest_results: Dict[str, Any]) -> FactorAnalysis:
        
        if not backtest_results or "error" in backtest_results:
            return FactorAnalysis(
                factor_type=FactorType.BACKTEST,
                strength=0.0,
                confidence=0.0,
                weight=self.base_weights[FactorType.BACKTEST],
                data=backtest_results,
                reasoning="Backtest results unavailable"
            )

        sharpe = backtest_results.get('sharpe_ratio', backtest_results.get('averaged_sharpe_ratio', 0))
        win_rate = backtest_results.get('win_rate', backtest_results.get('averaged_win_rate', 0))
        max_drawdown = backtest_results.get('max_drawdown', backtest_results.get('averaged_max_drawdown', 0))

        # Calculate strength based on backtest performance
        if sharpe > 1.5 and win_rate > 0.6 and max_drawdown < 0.15:
            strength = 0.9
        elif sharpe > 1.0 and win_rate > 0.55:
            strength = 0.6
        elif sharpe < 0.5 or win_rate < 0.45 or max_drawdown > 0.25:
            strength = -0.9
        elif sharpe < 0.8 or win_rate < 0.5:
            strength = -0.6
        else:
            strength = 0.0

        confidence = 0.7  # Backtest provides historical validation
        reasoning = f"Backtest: Sharpe {sharpe:.2f}, WinRate {win_rate:.1%}, MaxDD {max_drawdown:.1%}"

        return FactorAnalysis(
            factor_type=FactorType.BACKTEST,
            strength=strength,
            confidence=confidence,
            weight=self.base_weights[FactorType.BACKTEST],
            data=backtest_results,
            reasoning=reasoning
        )

    def _calculate_factor_consensus(self, factor_strengths: Dict[FactorType, float]) -> float:
        
        if not factor_strengths:
            return 0.0

        strengths = list(factor_strengths.values())
        if len(strengths) < 2:
            return 1.0

        # Calculate agreement ratio
        mean_strength = sum(strengths) / len(strengths)
        agreement_count = sum(1 for s in strengths if abs(s - mean_strength) < 0.3)
        consensus = agreement_count / len(strengths)

        return consensus

    def _calculate_confidence(self, factors: List[FactorAnalysis], composite_score: float, weights: Dict[FactorType, float]) -> float:
        
        # Base confidence from factor consensus
        factor_strengths = {f.factor_type: f.strength for f in factors}
        consensus = self._calculate_factor_consensus(factor_strengths)

        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"Confidence calculation - Factor consensus: {consensus:.3f}")

        # Weight confidence by factor importance
        weighted_confidence = 0.0
        total_weight = 0.0
        for factor in factors:
            weight = weights.get(factor.factor_type, 0.0)
            weighted_confidence += factor.confidence * weight
            total_weight += weight
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"  {factor.factor_type.value}: confidence={factor.confidence:.3f} * weight={weight:.3f} = {factor.confidence * weight:.3f}")

        if total_weight > 0:
            avg_factor_confidence = weighted_confidence / total_weight
        else:
            avg_factor_confidence = 0.5

        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"Weighted average factor confidence: {avg_factor_confidence:.3f}")

        # Boost confidence for strong consensus and extreme scores
        confidence_multiplier = 1.0
        if consensus > 0.7:
            confidence_multiplier += 0.2
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug("Consensus boost: +0.2 (consensus > 0.7)")
        if abs(composite_score) > 0.7:
            confidence_multiplier += 0.1
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Extreme score boost: +0.1 (abs(score) > 0.7, score={composite_score:.3f})")

        # Include Monte Carlo validation if available
        mc_factors = [f for f in factors if f.factor_type == FactorType.MONTE_CARLO and f.confidence > 0]
        if mc_factors:
            mc_alignment = 1.0 if (composite_score > 0 and mc_factors[0].strength > 0) or (composite_score < 0 and mc_factors[0].strength < 0) else 0.0
            confidence_multiplier += mc_alignment * 0.1
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Monte Carlo alignment boost: +{mc_alignment * 0.1:.1f} (alignment: {mc_alignment:.1f})")

        final_confidence = min(1.0, avg_factor_confidence * consensus * confidence_multiplier)

        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"Final confidence calculation: {avg_factor_confidence:.3f} * {consensus:.3f} * {confidence_multiplier:.3f} = {final_confidence:.3f}")

        return final_confidence

    def _generate_decision_reasoning(self, action: str, composite_score: float, factors: List[FactorAnalysis], weights: Dict[FactorType, float]) -> str:
        
        reasoning_parts = [f"Decision: {action} (Composite Score: {composite_score:.2f})"]

        # Add factor contributions
        for factor in factors:
            weight = weights.get(factor.factor_type, 0.0)
            contribution = factor.strength * weight
            reasoning_parts.append(f"{factor.factor_type.value.title()}: {factor.reasoning} (Strength: {factor.strength:.2f}, Weight: {weight:.2f}, Contribution: {contribution:.3f})")

        return " | ".join(reasoning_parts)


_engine = EnhancedRecommendationEngine()


def _rank_buy_candidates(final_recommendations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    
    # Filter for BUY actions with confidence > 40%
    buy_candidates = {
        symbol: rec for symbol, rec in final_recommendations.items()
        if rec.get("action") == "BUY" and rec.get("confidence", 0) > 0.4
    }

    if not buy_candidates:
        return {
            "top_buy_candidate": None,
            "buy_ranking": [],
            "ranking_reasoning": "No BUY recommendations with confidence > 40% found"
        }

    # Sort by confidence descending
    ranked_buys = sorted(
        buy_candidates.items(),
        key=lambda x: x[1].get("confidence", 0),
        reverse=True
    )

    # Top candidate
    top_symbol, top_rec = ranked_buys[0]

    # Enhanced reasoning for top candidate
    original_reasoning = top_rec.get("reasoning", "")
    ranking_details = f"Top BUY candidate among {len(ranked_buys)} qualified stocks. Ranking: " + \
                     ", ".join([f"{symbol} ({rec.get('confidence', 0):.1%})" for symbol, rec in ranked_buys])

    enhanced_reasoning = f"{original_reasoning} | {ranking_details}"

    # Update top candidate reasoning
    top_rec["reasoning"] = enhanced_reasoning

    return {
        "top_buy_candidate": {top_symbol: top_rec},
        "buy_ranking": [{"symbol": symbol, "confidence": rec.get("confidence", 0), "composite_score": rec.get("composite_score", 0)}
                       for symbol, rec in ranked_buys],
        "ranking_reasoning": f"Selected {top_symbol} as top BUY with {top_rec.get('confidence', 0):.1%} confidence from {len(ranked_buys)} candidates"
    }


def final_recommendation_agent(state: State) -> State:
    
    logging.info("Starting final recommendation agent")

    stock_data = state.get("stock_data", {})
    final_recommendations = {}

    for symbol in stock_data.keys():
        try:
            # Collect analysis results
            technical = state.get("technical_signals", {}).get(symbol, {})
            fundamental = state.get("fundamental_analysis", {}).get(symbol, {})
            sentiment = state.get("sentiment_scores", {}).get(symbol, {})
            risk = state.get("risk_metrics", {}).get(symbol, {})
            macro_scores = state.get("macro_scores", {})
            macro = macro_scores.get("composite", 0.0) if "error" not in macro_scores else 0.0

            # Generate recommendation using enhanced engine
            recommendation = _generate_recommendation(symbol, technical, fundamental, sentiment, risk, macro, state)

            final_recommendations[symbol] = recommendation
            logger.info(f"Generated recommendation for {symbol}")

        except Exception as e:
            logger.error(f"Error generating recommendation for {symbol}: {e}")
            final_recommendations[symbol] = {
                "action": "HOLD",
                "reasoning": "Error occurred during analysis",
                "confidence": 0.0
            }

    # Check if multiple stocks for ranking logic
    if len(stock_data) > 1:
        # Apply BUY ranking system for multiple stocks
        ranking_result = _rank_buy_candidates(final_recommendations)

        # Set top BUY candidate as primary recommendation if available
        if ranking_result["top_buy_candidate"]:
            state["final_recommendation"] = ranking_result["top_buy_candidate"]
            state["top_buy_candidate"] = ranking_result["top_buy_candidate"]
        else:
            # Fallback to original ranking if no BUY candidates
            ranked_recommendations = sorted(
                final_recommendations.items(),
                key=lambda x: x[1].get("confidence", 0),
                reverse=True
            )[:TOP_N_RECOMMENDATIONS]
            top_recommendations = {symbol: rec for symbol, rec in ranked_recommendations}
            state["final_recommendation"] = top_recommendations

        # Add ranking information
        state["buy_ranking"] = ranking_result["buy_ranking"]
        state["ranking_reasoning"] = ranking_result["ranking_reasoning"]

    else:
        # Single stock analysis - maintain existing behavior
        state["final_recommendation"] = final_recommendations

    # Common state updates
    state["llm_reasoning"] = {symbol: rec.get("llm_reasoning") for symbol, rec in state["final_recommendation"].items()}
    state["all_recommendations"] = final_recommendations  # Keep all for reference
    return state


def _generate_recommendation(
    symbol: str,
    technical: Dict[str, Union[str, float]],
    fundamental: Dict[str, Union[str, float]],
    sentiment: Dict[str, Union[str, float]],
    risk: Dict[str, Union[str, float]],
    macro: float,
    state: State
) -> Dict[str, Union[str, float]]:
    
    try:
        # Step 1: Comprehensive factor analysis
        factors = _engine.analyze_factors(symbol, state)

        # Step 2: Assess market conditions
        market_conditions = _engine.assess_market_conditions(factors)

        # Step 3: Calculate dynamic weights
        dynamic_weights = _engine.calculate_dynamic_weights(factors, market_conditions)

        # Step 4: Synthesize decision
        decision = _engine.synthesize_decision(factors, dynamic_weights)

        # Step 5: Generate enhanced LLM reasoning if available
        llm_reasoning = _get_enhanced_llm_reasoning(symbol, decision, factors, market_conditions, dynamic_weights) if _llm else None

        # Prepare final output with backward compatibility
        return {
            "action": decision['action'],
            "reasoning": llm_reasoning if llm_reasoning else decision['decision_reasoning'],
            "confidence": decision['confidence'],
            "composite_score": decision['composite_score'],
            "llm_reasoning": llm_reasoning,
            "factor_contributions": decision['factor_contributions'],
            "market_conditions": {
                "volatility_regime": market_conditions.volatility_regime,
                "trend_strength": market_conditions.trend_strength,
                "market_sentiment": market_conditions.market_sentiment,
                "risk_environment": market_conditions.risk_environment
            }
        }

    except Exception as e:
        logger.error(f"Error in enhanced recommendation generation: {e}")
        # Fallback to original logic
        return _generate_recommendation_fallback(symbol, technical, fundamental, sentiment, risk, macro)


def _generate_recommendation_fallback(
    symbol: str,
    technical: Dict[str, Union[str, float]],
    fundamental: Dict[str, Union[str, float]],
    sentiment: Dict[str, Union[str, float]],
    risk: Dict[str, Union[str, float]],
    macro: float
) -> Dict[str, Union[str, float]]:
    
    try:
        # Calculate weighted composite score for swing trading
        scores = _calculate_scores(technical, fundamental, sentiment, risk, macro)

        # Define base weights (prioritize technicals for swing trading, add macro)
        weights = {
            "technical": 0.45,
            "sentiment": 0.25,
            "risk": 0.15,
            "fundamental": 0.05,
            "macro": 0.10
        }

        # Check for missing analyses
        analysis_data = {
            "technical": technical,
            "fundamental": fundamental,
            "sentiment": sentiment,
            "risk": risk,
            "macro": {"composite": macro}
        }
        missing_analyses = [
            analysis for analysis, data in analysis_data.items()
            if 'error' in data and isinstance(data['error'], str)
        ]

        # Adjust weights for missing analyses
        for analysis in missing_analyses:
            weights[analysis] = 0.0

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] /= total_weight

        # Compute composite score
        composite_score = sum(scores[analysis] * weights[analysis] for analysis in scores)

        # Determine action based on composite score
        action, confidence = _determine_action(composite_score)

        llm_reasoning = _get_llm_reasoning(symbol, action, technical, fundamental, sentiment, risk, macro) if _llm else None

        # Create comprehensive reasoning
        reasoning_parts = []

        reasoning_parts.append(f"Composite score: {composite_score:.2f}")

        if missing_analyses:
            reasoning_parts.append(f"Missing: {', '.join(missing_analyses)} (weights adjusted)")

        # Technical reasoning
        technical_buys = sum(1 for v in technical.values() if isinstance(v, str) and v.lower() == "buy")
        technical_sells = sum(1 for v in technical.values() if isinstance(v, str) and v.lower() == "sell")
        technical_holds = sum(1 for v in technical.values() if isinstance(v, str) and v.lower() == "hold")
        total_tech_signals = technical_buys + technical_sells + technical_holds
        if total_tech_signals > 0:
            reasoning_parts.append(f"Technical: {technical_buys} BUY, {technical_sells} SELL, {technical_holds} HOLD (score: {scores['technical']:.2f})")
        else:
            reasoning_parts.append(f"Technical: No signals (score: {scores['technical']:.2f})")

        # Fundamental reasoning
        if "fundamental" in missing_analyses:
            reasoning_parts.append("Fundamental: Unavailable (excluded)")
        else:
            if scores["fundamental"] != 0:
                valuation = "undervalued" if scores["fundamental"] > 0 else "overvalued"
                reasoning_parts.append(f"Fundamental: {valuation} (score: {scores['fundamental']:.2f})")
            else:
                reasoning_parts.append(f"Fundamental: Neutral (score: {scores['fundamental']:.2f})")

        # Sentiment reasoning
        compound = sentiment.get('compound', 0)
        sentiment_dir = "positive" if scores["sentiment"] > 0 else "negative" if scores["sentiment"] < 0 else "neutral"
        reasoning_parts.append(f"Sentiment: {sentiment_dir} (score: {scores['sentiment']:.2f}, compound: {compound:.2f})")

        # Risk reasoning
        volatility = risk.get('volatility', 0)
        if scores["risk"] > 0:
            reasoning_parts.append(f"Risk: Favorable (score: {scores['risk']:.2f})")
        elif scores["risk"] < 0:
            reasoning_parts.append(f"Risk: Elevated (score: {scores['risk']:.2f}, vol: {volatility:.1f})")
        else:
            reasoning_parts.append(f"Risk: Neutral (score: {scores['risk']:.2f}, vol: {volatility:.1f})")

        # Macro reasoning
        macro_dir = "positive" if scores["macro"] > 0 else "negative" if scores["macro"] < 0 else "neutral"
        reasoning_parts.append(f"Macro: {macro_dir} (score: {scores['macro']:.2f})")

        # Combine reasoning
        if llm_reasoning:
            reasoning = llm_reasoning
        else:
            reasoning = " | ".join(reasoning_parts) if reasoning_parts else "Rule-based analysis"

        return {
            "action": action,
            "reasoning": reasoning,
            "confidence": confidence,
            "composite_score": composite_score,
            "llm_reasoning": llm_reasoning
        }

    except Exception as e:
        logger.error(f"Error in recommendation generation: {e}")
        return {
            "action": "HOLD",
            "reasoning": "Error in analysis",
            "confidence": 0.0
        }


def _calculate_scores(technical, fundamental, sentiment, risk, macro) -> Dict[str, float]:
    
    scores = {}

    # Technical score, normalized
    if 'error' in technical and isinstance(technical['error'], str):
        scores["technical"] = 0.0
    else:
        technical_score = 0
        technical_signals = 0
        for value in technical.values():
            if isinstance(value, str) and value.lower() in ["buy", "sell", "hold"]:
                technical_signals += 1
                if value.lower() == "buy":
                    technical_score += 1
                elif value.lower() == "sell":
                    technical_score -= 1
        if technical_signals > 0:
            scores["technical"] = technical_score / technical_signals
        else:
            scores["technical"] = 0.0

    # Fundamental score
    if 'error' in fundamental and isinstance(fundamental['error'], str):
        scores["fundamental"] = 0.0
    else:
        valuations = fundamental.get('valuations', '')
        if valuations == 'undervalued':
            scores["fundamental"] = 1.0
        elif valuations == 'overvalued':
            scores["fundamental"] = -1.0
        else:
            scores["fundamental"] = 0.0

    # Sentiment score
    if 'error' in sentiment and isinstance(sentiment['error'], str):
        scores["sentiment"] = 0.0
    else:
        compound = sentiment.get('compound', 0)
        if compound > 0.05:
            scores["sentiment"] = 1.0
        elif compound < -0.05:
            scores["sentiment"] = -1.0
        else:
            scores["sentiment"] = 0.0

    # Risk score
    if 'error' in risk and isinstance(risk['error'], str):
        scores["risk"] = 0.0
    else:
        risk_ok = risk.get('risk_ok', True)
        volatility = risk.get('volatility', 0)
        sharpe = risk.get('sharpe_ratio', 0)

        risk_score = 0.0
        if not risk_ok:
            risk_score = -1.0
        elif volatility > 0.4:
            risk_score = -0.5
        elif sharpe > 1.0:
            risk_score = 0.5

        scores["risk"] = risk_score

    # Macro score
    scores["macro"] = macro if abs(macro) > 0 else 0.0

    return scores


def _determine_action(composite_score: float) -> tuple[str, int]:
    
    if composite_score > 0.4:
        action = "BUY"
        confidence = min(100, int(50 + composite_score * 50))
    elif composite_score < -0.6:
        action = "SELL"
        confidence = min(100, int(50 + abs(composite_score) * 50))
    else:
        action = "HOLD"
        confidence = int(abs(composite_score) * 100)
    return action, confidence


def _get_llm_reasoning(
    symbol: str,
    action: str,
    technical: Dict,
    fundamental: Dict,
    sentiment: Dict,
    risk: Dict,
    macro: float
) -> Optional[str]:
    
    if not _llm:
        return None

    try:
        prompt_template = PromptTemplate(
            input_variables=["action", "technicals", "fundamentals", "sentiment", "risk", "macro"],
            template=
        )

        chain = prompt_template | _llm

        result = chain.invoke({
            "action": action,
            "technicals": technical,
            "fundamentals": fundamental,
            "sentiment": sentiment,
            "risk": risk,
            "macro": macro
        })
        return result.content.strip()

    except Exception as e:
        logger.warning(f"LLM reasoning failed: {e}")
        return None


def _get_enhanced_llm_reasoning(
    symbol: str,
    decision: Dict[str, Any],
    factors: List[FactorAnalysis],
    market_conditions: MarketConditions,
    weights: Dict[FactorType, float]
) -> Optional[str]:
    
    if not _llm:
        return None

    try:
        # Prepare comprehensive context
        factor_summary = []
        for factor in factors:
            weight = weights.get(factor.factor_type, 0.0)
            factor_summary.append(f"{factor.factor_type.value}: {factor.reasoning} (weight: {weight:.2f})")

        market_context = f"Market Conditions - Volatility: {market_conditions.volatility_regime}, Trend: {market_conditions.trend_strength}, Sentiment: {market_conditions.market_sentiment}, Risk: {market_conditions.risk_environment}"

        prompt_template = PromptTemplate(
            input_variables=["symbol", "action", "confidence", "composite_score", "factors", "market_conditions"],
            template=
        )

        chain = prompt_template | _llm

        result = chain.invoke({
            "symbol": symbol,
            "action": decision['action'],
            "confidence": decision['confidence'],
            "composite_score": decision['composite_score'],
            "factors": "\n".join(factor_summary),
            "market_conditions": market_context
        })
        return result.content.strip()

    except Exception as e:
        logger.warning(f"Enhanced LLM reasoning failed: {e}")
        return None


def _get_backtest_interpretation(backtest: Dict[str, Any]) -> str:
    if not _llm:
        return "LLM not available for backtest interpretation."
    sharpe = backtest.get('sharpe_ratio', backtest.get('averaged_sharpe_ratio', 0))
    win_rate = backtest.get('win_rate', backtest.get('averaged_win_rate', 0))
    drawdown = backtest.get('max_drawdown', backtest.get('averaged_max_drawdown', 0))
    prompt = PromptTemplate(
        input_variables=["sharpe", "win_rate", "drawdown"],
        template="Based on the backtest results for NSE swings: Sharpe ratio {sharpe}, win rate {win_rate}, max drawdown {drawdown}. Provide recommendation, risks, and benefits for trading."
    )
    chain = prompt | _llm
    result = chain.invoke({
        "sharpe": sharpe,
        "win_rate": win_rate,
        "drawdown": drawdown
    })
    return result.content.strip()


def backtest_interpretation_agent(state: State) -> State:
    if 'backtest_results' in state and 'error' not in state['backtest_results']:
        interp = _get_backtest_interpretation(state['backtest_results'])
        state['backtest_interpretation'] = interp
        logger.info(f"Backtest interpretation: {interp}")
    return state