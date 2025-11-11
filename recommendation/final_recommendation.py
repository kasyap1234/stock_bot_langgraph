

import logging
import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Any, List
from dataclasses import dataclass
from enum import Enum

from config.api_config import MODEL_NAME, GROQ_API_KEY, TEMPERATURE
from config.trading_config import TOP_N_RECOMMENDATIONS
from config.constants import DEBUG_RECOMMENDATION_LOGGING
from data.models import State
from langchain_core.prompts import PromptTemplate
from .intelligent_ensemble import IntelligentEnsembleEngine, Signal, SignalType, MarketContext

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
    ML = "ml"
    NEURAL = "neural"
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
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MarketConditions:
    
    volatility_regime: str  # "low", "medium", "high"
    trend_strength: str  # "weak", "moderate", "strong"
    market_sentiment: str  # "bearish", "neutral", "bullish"
    risk_environment: str  # "low_risk", "moderate_risk", "high_risk"


class EnhancedRecommendationEngine:
    

    def __init__(self):
        # IMPROVED: Increased backtest weight - historical performance is most predictive
        # Increased ML/Neural weights - these models have proven accuracy
        # Reduced macro weight - less predictive for individual stocks
        self.base_weights = {
            FactorType.TECHNICAL: 0.22,
            FactorType.FUNDAMENTAL: 0.14,
            FactorType.SENTIMENT: 0.18,
            FactorType.RISK: 0.14,
            FactorType.MACRO: 0.06,
            FactorType.ML: 0.08,
            FactorType.NEURAL: 0.08,
            FactorType.MONTE_CARLO: 0.04,
            FactorType.BACKTEST: 0.10  # DOUBLED from 0.04 to 0.10 - backtest is most reliable
        }
        self.current_market_conditions = None
        # Track prediction accuracy for adaptive weighting
        self.prediction_history = {}

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

        # ML Analysis
        ml_predictions = state.get("ml_predictions", {}).get(symbol, {})
        ml_factor = self._analyze_ml_factor(ml_predictions)
        factors.append(ml_factor)
        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"[{symbol}] ML factor: strength={ml_factor.strength:.3f}, confidence={ml_factor.confidence:.3f}, weight={ml_factor.weight:.3f}")

        # Neural Network Analysis
        nn_predictions = state.get("nn_predictions", {}).get(symbol, {})
        neural_factor = self._analyze_neural_factor(nn_predictions)
        factors.append(neural_factor)
        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"[{symbol}] Neural factor: strength={neural_factor.strength:.3f}, confidence={neural_factor.confidence:.3f}, weight={neural_factor.weight:.3f}")

        # Monte Carlo Simulation
        simulation_results = state.get("simulation_results", {})
        monte_carlo_factor = self._analyze_monte_carlo_factor(simulation_results)
        factors.append(monte_carlo_factor)
        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"[{symbol}] Monte Carlo factor: strength={monte_carlo_factor.strength:.3f}, confidence={monte_carlo_factor.confidence:.3f}, weight={monte_carlo_factor.weight:.3f}")

        # Backtest Results
        backtest_results = state.get("backtest_results", {})
        backtest_factor = self._analyze_backtest_factor(backtest_results, symbol, df)
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

        # Apply tiered signal strength system
        composite_score = self._apply_signal_strength_tiers(composite_score, factors)

        # Apply market regime adaptation
        composite_score = self._apply_market_regime_adaptation(composite_score)

        # Apply signal decay system
        composite_score = self._apply_signal_decay(composite_score, factors)

        # Legacy boosts (keep for compatibility)
        # Strong signal boost: +0.1 if any factor > 0.8
        strong_signals = [f for f in factors if f.strength > 0.8]
        if strong_signals:
            composite_score += 0.1
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Strong signal boost applied: +0.1 (factors: {[f.factor_type.value for f in strong_signals]})")

        # Backtest profitability boost
        backtest_factor = next((f for f in factors if f.factor_type == FactorType.BACKTEST), None)
        if backtest_factor and backtest_factor.strength > 0.6:
            composite_score += 0.3
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Backtest profitability boost: +0.3 (strength: {backtest_factor.strength:.3f})")

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
            logger.debug(f"All boosts applied: original={original_score:.3f}, final={composite_score:.3f}, positive_factors={positive_factors}")

        # IMPROVED: Dynamic thresholds based on market conditions and signal quality
        backtest_factor = next((f for f in factors if f.factor_type == FactorType.BACKTEST), None)
        ml_factor = next((f for f in factors if f.factor_type == FactorType.ML), None)
        neural_factor = next((f for f in factors if f.factor_type == FactorType.NEURAL), None)
        technical_factor = next((f for f in factors if f.factor_type == FactorType.TECHNICAL), None)
        
        # Calculate signal confirmation score (how many models agree)
        buy_signals = sum(1 for f in factors if f.strength > 0.3)
        sell_signals = sum(1 for f in factors if f.strength < -0.3)
        strong_buy_signals = sum(1 for f in factors if f.strength > 0.6)
        strong_sell_signals = sum(1 for f in factors if f.strength < -0.6)
        
        # Adaptive thresholds based on volatility
        vol_multiplier = 1.0
        if self.current_market_conditions:
            if self.current_market_conditions.volatility_regime == "high":
                vol_multiplier = 1.3  # Require stronger signals in volatile markets
            elif self.current_market_conditions.volatility_regime == "low":
                vol_multiplier = 0.8  # Can be more aggressive in calm markets
        
        buy_threshold = 0.05 * vol_multiplier  # IMPROVED: Lower from 0.08, but adaptive
        sell_threshold = -0.05 * vol_multiplier
        
        # Strong backtest override - if backtest is excellent, trust it more
        if backtest_factor and backtest_factor.strength > 0.7 and composite_score > -0.02:
            action = "BUY"
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"BUY: Strong backtest override (strength={backtest_factor.strength:.3f})")
        # Strong ML/Neural consensus - models agree strongly
        elif ml_factor and neural_factor and ml_factor.strength > 0.6 and neural_factor.strength > 0.6:
            action = "BUY"
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"BUY: ML/Neural consensus (ML={ml_factor.strength:.3f}, NN={neural_factor.strength:.3f})")
        # Multiple strong buy signals with positive score
        elif strong_buy_signals >= 2 and composite_score > 0:
            action = "BUY"
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"BUY: Multiple strong signals ({strong_buy_signals} signals)")
        # Standard buy threshold with confirmation
        elif composite_score > buy_threshold and (buy_signals >= 4 or strong_buy_signals >= 1):
            action = "BUY"
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"BUY: Standard threshold (score={composite_score:.3f} > {buy_threshold:.3f})")
        # Strong sell signals
        elif strong_sell_signals >= 2 and composite_score < 0:
            action = "SELL"
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"SELL: Multiple strong signals ({strong_sell_signals} signals)")
        # Standard sell threshold
        elif composite_score < sell_threshold and (sell_signals >= 4 or strong_sell_signals >= 1):
            action = "SELL"
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"SELL: Standard threshold (score={composite_score:.3f} < {sell_threshold:.3f})")
        # Poor backtest - consider selling even with neutral score
        elif backtest_factor and backtest_factor.strength < -0.5 and composite_score < 0.1:
            action = "SELL"
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"SELL: Poor backtest (strength={backtest_factor.strength:.3f})")
        else:
            action = "HOLD"
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"HOLD: No clear signal (score={composite_score:.3f}, buy_signals={buy_signals}, sell_signals={sell_signals})")

        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"Decision: action={action}, score={composite_score:.3f}, threshold=±{buy_threshold:.3f}, strong_buy={strong_buy_signals}, strong_sell={strong_sell_signals}")

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

    def _apply_signal_strength_tiers(self, composite_score: float, factors: List[FactorAnalysis]) -> float:
        """Apply tiered signal strength system to enhance decisive signals"""
        
        # Count strong signals (>0.8 or <-0.8)
        strong_buy_signals = sum(1 for f in factors if f.strength > 0.8)
        strong_sell_signals = sum(1 for f in factors if f.strength < -0.8)
        
        # Count moderate signals (0.3-0.8 or -0.3 to -0.8)
        moderate_buy_signals = sum(1 for f in factors if 0.3 < f.strength <= 0.8)
        moderate_sell_signals = sum(1 for f in factors if -0.8 <= f.strength < -0.3)
        
        # Apply tiered adjustments
        if strong_buy_signals >= 2 and composite_score > 0.2:
            # Tier 1: Multiple strong buy signals - boost significantly
            composite_score += 0.15
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Tier 1 BUY boost: +0.15 (strong_buy={strong_buy_signals}, moderate_buy={moderate_buy_signals})")
        elif strong_buy_signals >= 1 and moderate_buy_signals >= 2 and composite_score > 0.1:
            # Tier 2: Strong + moderate buy signals - moderate boost
            composite_score += 0.08
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Tier 2 BUY boost: +0.08 (strong_buy={strong_buy_signals}, moderate_buy={moderate_buy_signals})")
        elif strong_sell_signals >= 2 and composite_score < -0.2:
            # Tier 1: Multiple strong sell signals - boost significantly
            composite_score -= 0.15
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Tier 1 SELL boost: -0.15 (strong_sell={strong_sell_signals}, moderate_sell={moderate_sell_signals})")
        elif strong_sell_signals >= 1 and moderate_sell_signals >= 2 and composite_score < -0.1:
            # Tier 2: Strong + moderate sell signals - moderate boost
            composite_score -= 0.08
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Tier 2 SELL boost: -0.08 (strong_sell={strong_sell_signals}, moderate_sell={moderate_sell_signals})")
        
        return composite_score

    def _apply_market_regime_adaptation(self, composite_score: float) -> float:
        """Apply market regime-based adaptations to composite score"""
        
        # Get current market conditions
        market_conditions = self.current_market_conditions
        if not market_conditions:
            return composite_score
        
        # Apply regime-specific adjustments
        if market_conditions.volatility_regime == "high":
            # In high volatility, reduce signal strength to avoid false signals
            composite_score *= 0.85
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"High volatility regime: composite_score reduced by 15% to {composite_score:.3f}")
        elif market_conditions.volatility_regime == "low":
            # In low volatility, increase signal strength for better responsiveness
            composite_score *= 1.1
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Low volatility regime: composite_score increased by 10% to {composite_score:.3f}")
        
        # Trend regime adjustments
        if market_conditions.trend_strength == "strong":
            # In strong trends, amplify existing signals
            if composite_score > 0:
                composite_score *= 1.05
            elif composite_score < 0:
                composite_score *= 1.05
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Strong trend regime: composite_score amplified by 5% to {composite_score:.3f}")
        
        # Sentiment regime adjustments
        if market_conditions.market_sentiment == "bullish" and composite_score > 0:
            composite_score *= 1.03
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Bullish sentiment regime: BUY signals boosted by 3% to {composite_score:.3f}")
        elif market_conditions.market_sentiment == "bearish" and composite_score < 0:
            composite_score *= 1.03
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Bearish sentiment regime: SELL signals boosted by 3% to {composite_score:.3f}")
        
        return composite_score
    
    def _apply_signal_decay(self, composite_score: float, factors: List[FactorAnalysis]) -> float:
        """Apply signal decay based on factor freshness and market conditions"""
        
        # Get current market conditions for decay adjustment
        market_conditions = self.current_market_conditions
        
        # Calculate average factor age (assuming factors have timestamp info)
        current_time = pd.Timestamp.now()
        factor_ages = []
        
        for factor in factors:
            # Estimate factor age based on data freshness (simplified)
            # In real implementation, factors should have timestamp metadata
            if hasattr(factor, 'timestamp'):
                age_hours = (current_time - factor.timestamp).total_seconds() / 3600
            else:
                # Default age based on factor type
                age_hours = {
                    FactorType.TECHNICAL: 1,      # Technical indicators are recent
                    FactorType.FUNDAMENTAL: 24,    # Fundamental data is daily
                    FactorType.SENTIMENT: 2,       # Sentiment data is recent
                    FactorType.RISK: 1,            # Risk data is very recent
                    FactorType.MACRO: 24           # Macro data is daily
                }.get(factor.factor_type, 6)
            
            factor_ages.append(age_hours)
        
        # Calculate decay factor based on average age
        avg_age_hours = sum(factor_ages) / len(factor_ages) if factor_ages else 6
        
        # Base decay: 5% per hour, capped at 50% decay
        base_decay = min(0.05 * avg_age_hours, 0.5)
        
        # Adjust decay based on market conditions
        decay_multiplier = 1.0
        if market_conditions:
            if market_conditions.volatility_regime == "high":
                # Faster decay in high volatility markets
                decay_multiplier = 1.3
            elif market_conditions.volatility_regime == "low":
                # Slower decay in stable markets
                decay_multiplier = 0.8
            
            if market_conditions.trend_strength == "strong":
                # Slower decay in strong trends
                decay_multiplier *= 0.9
        
        # Apply decay
        total_decay = base_decay * decay_multiplier
        decay_factor = 1.0 - total_decay
        
        # Apply decay to composite score (reduce magnitude)
        if composite_score > 0:
            composite_score *= decay_factor
        elif composite_score < 0:
            composite_score *= decay_factor
        
        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"Signal decay applied: avg_age={avg_age_hours:.1f}h, decay={total_decay:.2f}, factor={decay_factor:.2f}, new_score={composite_score:.3f}")
        
        return composite_score
    
    def _calculate_volatility_and_trend(self, df: pd.DataFrame) -> Dict[str, float]:
        
        if len(df) < 20:
            return {'volatility': 0.02, 'trend_strength': 0.5}  # Defaults
        
        df = df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
        
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
                weight=0.25,
                data=technical,
                reasoning="Technical analysis unavailable"
            )

        ensemble = technical.get('technical_ensemble', {})
        if ensemble:
            signal = ensemble.get('signal', 'HOLD')
            strength = 1 if signal == 'BUY' else -1 if signal == 'SELL' else 0
            strength *= ensemble.get('confidence', 1.0)
            confidence = ensemble.get('confidence', 0.5)
            reasoning = f"Technical ensemble: {signal} (conf: {confidence:.2f})"
        else:
            # Fallback to original logic
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
            weight=0.25,
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

        signal = fundamental.get('fundamental_signal')
        score = fundamental.get('fundamental_score')
        confidence = float(fundamental.get('fundamental_confidence', 0.4))

        if signal:
            if score is None:
                score = 0.0
            strength = float(np.clip(score, -1.0, 1.0))
            if strength == 0.0:
                if signal == 'BUY':
                    strength = 0.4
                elif signal == 'SELL':
                    strength = -0.4
            reasoning = (
                f"Signal {signal} with valuation score {fundamental.get('valuation_score', 0):.2f} "
                f"and financial score {fundamental.get('financial_health_score', 0):.2f}"
            )
        else:
            valuations = fundamental.get('general_valuation', 'unknown')
            if isinstance(valuations, str) and 'undervalued' in valuations:
                strength = 0.6
                confidence = max(confidence, 0.6)
                reasoning = f"General valuation {valuations}"
            elif isinstance(valuations, str) and 'overvalued' in valuations:
                strength = -0.6
                confidence = max(confidence, 0.6)
                reasoning = f"General valuation {valuations}"
            else:
                strength = 0.0
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

    def _analyze_backtest_factor(self, backtest_results: Dict[str, Any], symbol: str, df: pd.DataFrame) -> FactorAnalysis:
        # Extract metrics from backtest_results if available, else compute fallback
        if backtest_results and "error" not in backtest_results:
            sharpe = backtest_results.get('sharpe_ratio', backtest_results.get('averaged_sharpe_ratio', 0))
            win_rate = backtest_results.get('win_rate', backtest_results.get('averaged_win_rate', 0))
            max_drawdown = backtest_results.get('max_drawdown', backtest_results.get('averaged_max_drawdown', 0))
            total_return = backtest_results.get('total_return', 0)
            source = "backtest_results"
            reasoning = f"Backtest results analysis: Sharpe {sharpe:.2f}, WinRate {win_rate:.1%}, MaxDD {max_drawdown:.1%}"
        else:
            df = df.rename(columns={'close': 'Close'})
            if df.empty or len(df) < 100:
                sharpe = 0.0
                win_rate = 0.0
                max_drawdown = 0.0
                total_return = 0.0
                source = "insufficient_data"
                reasoning = "Insufficient historical data for backtest"
            else:
                returns = df['Close'].pct_change().dropna()
                if len(returns) < 50:
                    sharpe = 0.0
                    win_rate = 0.0
                    max_drawdown = 0.0
                    total_return = 0.0
                    source = "insufficient_returns"
                    reasoning = "Insufficient returns data for backtest"
                else:
                    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1)
                    num_years = len(returns) / 252
                    annual_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else total_return
                    vol = returns.std() * np.sqrt(252)
                    sharpe = annual_return / vol if vol > 0 else 0
                    win_rate = (returns > 0).mean()
                    cum_max = df['Close'].expanding().max()
                    drawdown = (cum_max - df['Close']) / cum_max
                    max_drawdown = drawdown.max()
                    source = "historical_fallback"
                    reasoning = f"Historical B&H Backtest: Sharpe {sharpe:.2f}, Win Rate {win_rate:.1%}, Total Return {total_return:.1%}, Max DD {max_drawdown:.1%}"
                    backtest_results = {'sharpe_ratio': sharpe, 'win_rate': win_rate, 'max_drawdown': max_drawdown, 'total_return': total_return}

        if source in ["insufficient_data", "insufficient_returns"]:
            strength = 0.0
        else:
            # Calculate strength based on backtest performance
            if sharpe > 1.5 and win_rate > 0.6 and max_drawdown < 0.15:
                strength = 0.9
            elif sharpe > 1.0 and win_rate > 0.55:
                strength = 0.8
            elif total_return > 0:
                strength = 0.8
            elif sharpe < 0.5 or win_rate < 0.45 or max_drawdown > 0.25:
                strength = -0.9
            elif sharpe < 0.8 or win_rate < 0.5:
                strength = -0.6
            else:
                strength = 0.0

        confidence = 0.7  # Backtest provides historical validation

        return FactorAnalysis(
            factor_type=FactorType.BACKTEST,
            strength=strength,
            confidence=confidence,
            weight=self.base_weights[FactorType.BACKTEST],
            data=backtest_results,
            reasoning=reasoning
        )

    def _analyze_ml_factor(self, ml_predictions: Dict[str, Any]) -> FactorAnalysis:
        
        if not ml_predictions or 'error' in ml_predictions:
            return FactorAnalysis(
                factor_type=FactorType.ML,
                strength=0.0,
                confidence=0.0,
                weight=self.base_weights[FactorType.ML],
                data=ml_predictions,
                reasoning="ML analysis unavailable"
            )

        # Extract from nested structure: ml_predictions['latest_prediction']
        latest_pred = ml_predictions.get('latest_prediction', {})
        
        # Get ensemble probability [prob_class_0, prob_class_1]
        ensemble_proba = latest_pred.get('ensemble_probability')
        confidence_score = latest_pred.get('confidence_score', 0.7)
        
        if ensemble_proba is not None and len(ensemble_proba) == 2:
            # Convert probability to strength: -1 (sell) to 1 (buy)
            # If prob_class_1 > 0.5, it's a buy signal
            prob_buy = float(ensemble_proba[1]) if hasattr(ensemble_proba[1], '__float__') else ensemble_proba[1]
            strength = (prob_buy - 0.5) * 2  # Maps 0.5->0, 1.0->1, 0.0->-1
            strength = np.clip(strength, -1.0, 1.0)
            confidence = min(0.9, confidence_score)
            reasoning = f"ML ensemble: {prob_buy*100:.1f}% buy probability, confidence {confidence:.2f}"
        else:
            # Fallback: use simple prediction if available
            ensemble_pred = latest_pred.get('ensemble_prediction')
            if ensemble_pred is not None:
                strength = 0.6 if ensemble_pred > 0.5 else -0.6
                confidence = 0.6
                reasoning = f"ML prediction: {ensemble_pred}"
            else:
                strength = 0.0
                confidence = 0.3
                reasoning = "ML prediction inconclusive"

        return FactorAnalysis(
            factor_type=FactorType.ML,
            strength=strength,
            confidence=confidence,
            weight=self.base_weights[FactorType.ML],
            data=ml_predictions,
            reasoning=reasoning
        )

    def _analyze_neural_factor(self, nn_predictions: Dict[str, Any]) -> FactorAnalysis:
        
        if not nn_predictions or 'error' in nn_predictions:
            return FactorAnalysis(
                factor_type=FactorType.NEURAL,
                strength=0.0,
                confidence=0.0,
                weight=self.base_weights[FactorType.NEURAL],
                data=nn_predictions,
                reasoning="Neural analysis unavailable"
            )

        # Extract from nested structure: nn_predictions['predictions']
        predictions = nn_predictions.get('predictions', {})
        
        # Get ensemble prediction and confidence
        ensemble_pred = predictions.get('ensemble_prediction')
        nn_confidence = predictions.get('confidence', 0.75)
        ensemble_std = predictions.get('ensemble_std', 0.2)
        
        if ensemble_pred is not None:
            # Convert NN prediction to strength
            # Assuming ensemble_pred is a probability or binary (0/1)
            if isinstance(ensemble_pred, (list, np.ndarray)):
                ensemble_pred = float(ensemble_pred[0]) if len(ensemble_pred) > 0 else 0.5
            else:
                ensemble_pred = float(ensemble_pred)
            
            # Map to -1 to 1 range
            strength = (ensemble_pred - 0.5) * 2
            strength = np.clip(strength, -1.0, 1.0)
            
            # Adjust confidence based on ensemble uncertainty
            confidence = min(0.9, nn_confidence * (1 - min(0.3, ensemble_std)))
            reasoning = f"Neural ensemble: prediction {ensemble_pred:.3f}, std {ensemble_std:.3f}, confidence {confidence:.2f}"
        else:
            strength = 0.0
            confidence = 0.3
            reasoning = "Neural prediction inconclusive"

        return FactorAnalysis(
            factor_type=FactorType.NEURAL,
            strength=strength,
            confidence=confidence,
            weight=self.base_weights[FactorType.NEURAL],
            data=nn_predictions,
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

        # IMPROVED: Better confidence calibration with stronger boosts
        confidence_multiplier = 1.0
        
        # Strong consensus boost
        if consensus > 0.8:
            confidence_multiplier += 0.35  # IMPROVED: from 0.2
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug("Strong consensus boost: +0.35 (consensus > 0.8)")
        elif consensus > 0.7:
            confidence_multiplier += 0.25  # IMPROVED: from 0.2
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug("Good consensus boost: +0.25 (consensus > 0.7)")
        
        # Extreme score boost - strong signals should have higher confidence
        if abs(composite_score) > 0.5:
            confidence_multiplier += 0.25  # IMPROVED: from 0.1
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Strong signal boost: +0.25 (abs(score) > 0.5, score={composite_score:.3f})")
        elif abs(composite_score) > 0.3:
            confidence_multiplier += 0.15
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Moderate signal boost: +0.15 (abs(score) > 0.3)")

        # Backtest confidence boost - historical performance matters most
        backtest_factors = [f for f in factors if f.factor_type == FactorType.BACKTEST]
        if backtest_factors and abs(backtest_factors[0].strength) > 0.5:
            backtest_boost = min(0.3, abs(backtest_factors[0].strength) * 0.3)
            confidence_multiplier += backtest_boost
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Backtest boost: +{backtest_boost:.2f} (strength={backtest_factors[0].strength:.3f})")
        
        # ML/Neural consensus boost
        ml_factors = [f for f in factors if f.factor_type in [FactorType.ML, FactorType.NEURAL]]
        if len(ml_factors) >= 2:
            ml_agreement = sum(1 for f in ml_factors if (f.strength > 0.3) == (composite_score > 0)) / len(ml_factors)
            if ml_agreement > 0.7:
                confidence_multiplier += 0.2
                if DEBUG_RECOMMENDATION_LOGGING:
                    logger.debug(f"ML/Neural agreement boost: +0.2 (agreement={ml_agreement:.2f})")

        # Include Monte Carlo validation if available
        mc_factors = [f for f in factors if f.factor_type == FactorType.MONTE_CARLO and f.confidence > 0]
        if mc_factors:
            mc_alignment = 1.0 if (composite_score > 0 and mc_factors[0].strength > 0) or (composite_score < 0 and mc_factors[0].strength < 0) else 0.0
            confidence_multiplier += mc_alignment * 0.15  # IMPROVED: from 0.1
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Monte Carlo alignment boost: +{mc_alignment * 0.15:.2f} (alignment: {mc_alignment:.1f})")

        # Momentum-based confidence adjustments
        momentum_boost = self._calculate_momentum_boost(factors, composite_score)
        confidence_multiplier += momentum_boost
        if DEBUG_RECOMMENDATION_LOGGING and momentum_boost != 0:
            logger.debug(f"Momentum boost: {momentum_boost:+.2f}")

        # IMPROVED: Better baseline and calibration
        # Increase base confidence from avg_factor_confidence to make it less conservative
        base_confidence = min(0.85, avg_factor_confidence * 1.15)  # Boost by 15%
        
        final_confidence = min(1.0, base_confidence * (0.7 + 0.3 * consensus) * confidence_multiplier)

        if DEBUG_RECOMMENDATION_LOGGING:
            logger.debug(f"Final confidence calculation: {base_confidence:.3f} * {(0.7 + 0.3 * consensus):.3f} * {confidence_multiplier:.3f} = {final_confidence:.3f}")

        return final_confidence

    def _calculate_momentum_boost(self, factors: List[FactorAnalysis], composite_score: float) -> float:
        """Calculate momentum-based confidence boost based on factor alignment and strength."""
        
        momentum_boost = 0.0
        
        # Calculate momentum factors from technical analysis
        momentum_factors = [f for f in factors if f.factor_type == FactorType.TECHNICAL and f.metadata and 'momentum' in f.metadata]
        if momentum_factors:
            # Average momentum strength
            avg_momentum = sum(f.strength for f in momentum_factors) / len(momentum_factors)
            
            # Boost confidence if momentum aligns with composite score
            if (composite_score > 0 and avg_momentum > 0.3) or (composite_score < 0 and avg_momentum < -0.3):
                momentum_boost += 0.15
                if DEBUG_RECOMMENDATION_LOGGING:
                    logger.debug(f"Momentum alignment boost: +0.15 (momentum: {avg_momentum:.3f})")
            elif abs(avg_momentum) > 0.5:  # Strong momentum regardless of direction
                momentum_boost += 0.08
                if DEBUG_RECOMMENDATION_LOGGING:
                    logger.debug(f"Strong momentum boost: +0.08 (momentum: {avg_momentum:.3f})")
        
        # Volume-based momentum confirmation
        volume_factors = [f for f in factors if f.factor_type == FactorType.TECHNICAL and f.metadata and 'volume' in f.metadata]
        if volume_factors:
            avg_volume_strength = sum(f.strength for f in volume_factors) / len(volume_factors)
            
            # Volume confirmation boost
            if abs(avg_volume_strength) > 0.4:
                momentum_boost += 0.05
                if DEBUG_RECOMMENDATION_LOGGING:
                    logger.debug(f"Volume confirmation boost: +0.05 (volume: {avg_volume_strength:.3f})")
        
        # Trend strength momentum
        trend_factors = [f for f in factors if f.factor_type == FactorType.TECHNICAL and f.metadata and 'trend' in f.metadata]
        if trend_factors:
            avg_trend_strength = sum(f.strength for f in trend_factors) / len(trend_factors)
            
            # Trend momentum boost
            if abs(avg_trend_strength) > 0.6:
                momentum_boost += 0.10
                if DEBUG_RECOMMENDATION_LOGGING:
                    logger.debug(f"Trend momentum boost: +0.10 (trend: {avg_trend_strength:.3f})")
        
        # Multi-timeframe momentum convergence
        timeframe_momentum = {}
        for factor in factors:
            if factor.factor_type == FactorType.TECHNICAL and factor.metadata and 'timeframe' in factor.metadata:
                timeframe = factor.metadata['timeframe']
                if timeframe not in timeframe_momentum:
                    timeframe_momentum[timeframe] = []
                timeframe_momentum[timeframe].append(factor.strength)
        
        if len(timeframe_momentum) >= 3:  # At least 3 timeframes
            # Calculate momentum alignment across timeframes
            timeframe_consensus = 0
            for timeframe, strengths in timeframe_momentum.items():
                avg_strength = sum(strengths) / len(strengths) if strengths else 0
                if (composite_score > 0 and avg_strength > 0.2) or (composite_score < 0 and avg_strength < -0.2):
                    timeframe_consensus += 1
            
            consensus_ratio = timeframe_consensus / len(timeframe_momentum)
            if consensus_ratio > 0.7:  # Strong consensus across timeframes
                momentum_boost += 0.12
                if DEBUG_RECOMMENDATION_LOGGING:
                    logger.debug(f"Multi-timeframe consensus boost: +0.12 (consensus: {consensus_ratio:.2f})")
        
        # Cap momentum boost to avoid overconfidence
        momentum_boost = min(momentum_boost, 0.25)
        
        return momentum_boost

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
    """EnsembleActor: Generate final recommendations."""
    logger.info("EnsembleActor started")
    
    stock_data = state.get("stock_data", {})
    final_recommendations = {}

    # Initialize the intelligent ensemble engine
    intelligent_engine = IntelligentEnsembleEngine()

    for symbol in stock_data.keys():
        try:
            # Collect analysis results
            technical = state.get("technical_signals", {}).get(symbol, {})
            fundamental = state.get("fundamental_analysis", {}).get(symbol, {})
            sentiment = state.get("sentiment_scores", {}).get(symbol, {})
            risk = state.get("risk_metrics", {}).get(symbol, {})
            macro_scores = state.get("macro_scores", {})
            macro = macro_scores.get("composite", 0.0) if "error" not in macro_scores else 0.0

            # NEW: Create signals for the intelligent ensemble engine
            signals = _create_signals_from_analysis(technical, fundamental, sentiment, risk, macro, symbol)
            
            # NEW: Determine market context from state or defaults
            market_context = _create_market_context_from_state(state, symbol)

            # NEW: Generate recommendation using the intelligent ensemble engine
            recommendation_obj = intelligent_engine.generate_ensemble_recommendation(signals, market_context)
            
            # Convert the recommendation object to the expected dictionary format
            recommendation = {
                "action": recommendation_obj.action,
                "reasoning": recommendation_obj.reasoning,
                "confidence": recommendation_obj.confidence,
                "composite_score": recommendation_obj.strength,  # Use strength as composite score
                "llm_reasoning": getattr(recommendation_obj, 'llm_reasoning', None),
                "factor_contributions": recommendation_obj.signal_contributions,
                "market_conditions": {
                    "volatility_regime": market_context.volatility_regime,
                    "trend_regime": market_context.trend_regime,
                    "market_sentiment": market_context.market_sentiment,
                    "correlation_regime": market_context.correlation_regime,
                    "volume_regime": market_context.volume_regime
                }
            }

            final_recommendations[symbol] = recommendation
            logger.info(f"EnsembleActor generated recommendation for {symbol}: {recommendation.get('action', 'HOLD')} (confidence: {recommendation.get('confidence', 0):.2f})")

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
    state["all_recommendations"] = final_recommendations # Keep all for reference
    symbols = list(stock_data.keys())
    logger.info(f"EnsembleActor completed for {len(symbols)} symbols: {symbols}")
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
        
        # Store current market conditions for regime adaptation
        _engine.current_market_conditions = market_conditions

        # Step 3: Calculate dynamic weights
        dynamic_weights = _engine.calculate_dynamic_weights(factors, market_conditions)

        # Step 4: Synthesize decision
        decision = _engine.synthesize_decision(factors, dynamic_weights)

        # Post-processing for profitability bias
        backtest_factor = next((f for f in factors if f.factor_type == FactorType.BACKTEST), None)
        profitability_note = ""
        if backtest_factor:
            backtest_data = backtest_factor.data
            total_return = backtest_data.get('total_return', 0) if isinstance(backtest_data, dict) else 0
            sharpe = backtest_data.get('sharpe_ratio', backtest_factor.strength) if isinstance(backtest_data, dict) else backtest_factor.strength
            win_rate = backtest_data.get('win_rate', 0) if isinstance(backtest_data, dict) else 0
            if sharpe > 0.5 or win_rate > 0.5 or total_return > -0.05:
                if decision['action'] != 'BUY':
                    decision['action'] = 'BUY'
                profitability_note = f" Based on backtest Sharpe {sharpe:.1f} and win rate {win_rate:.0%}, recommend BUY for expected profit."
            else:
                if decision['action'] == 'BUY':
                    decision['action'] = 'HOLD'
                profitability_note = f" Caution: Backtest shows negative returns (Sharpe {sharpe:.1f}, win rate {win_rate:.0%}), adjusting to HOLD/SELL."
        else:
            profitability_note = ""

        # Step 5: Generate enhanced LLM reasoning if available
        llm_reasoning = _get_enhanced_llm_reasoning(symbol, decision, factors, market_conditions, dynamic_weights) if _llm else None

        if llm_reasoning:
            llm_reasoning += profitability_note
        else:
            decision['decision_reasoning'] += profitability_note

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
    
    # Optimized thresholds for more decisive trading
    if composite_score > 0.25:  # Reduced from 0.4
        action = "BUY"
        confidence = min(100, int(50 + composite_score * 50))
    elif composite_score < -0.35:  # Reduced from -0.6
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
            template="""Based on the following analysis for swing trading:
Action: {action}
Technical Analysis: {technicals}
Fundamental Analysis: {fundamentals}
Sentiment Analysis: {sentiment}
Risk Assessment: {risk}
Macro Environment: {macro}

Provide a detailed reasoning for the recommended action, highlighting key factors and potential risks."""
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
            template="""You are a profitable trading AI assistant. Your goal is to generate recommendations that maximize profitability while managing risk.

For {symbol}:
Current Rule-based Action: {action} (Confidence: {confidence:.1f}%, Composite Score: {composite_score:.2f})

Factor Analysis (pay special attention to Backtest factor for historical profitability):
{factors}

Market Conditions:
{market_conditions}

Provide comprehensive reasoning prioritizing profitability and risk-adjusted returns. Analyze the backtest performance: if Sharpe >1.0 or win rate >60% or historical return >0%, strongly reinforce BUY and highlight profitable potential. If backtest shows losses (Sharpe <0.5 or win rate <50%), suggest adjusting to HOLD or SELL to avoid drawdowns. Weigh all factors but bias towards historically profitable strategies.

End your response with exactly: "Final Recommendation: BUY" or "Final Recommendation: SELL" or "Final Recommendation: HOLD"."""
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


def _create_signals_from_analysis(technical: Dict, fundamental: Dict, sentiment: Dict,
                                 risk: Dict, macro: float, symbol: str) -> List[Signal]:
    """Convert analysis results into Signal objects for the intelligent ensemble engine."""
    signals = []
    
    # Convert technical signals
    if 'error' not in technical:
        # Example: convert RSI signal
        rsi_signal = technical.get('RSI_daily')
        if rsi_signal and isinstance(rsi_signal, str):
            strength = 0.5 if rsi_signal.lower() == 'buy' else -0.5 if rsi_signal.lower() == 'sell' else 0.0
            signals.append(Signal(
                signal_type=SignalType.TECHNICAL,
                strength=strength,
                confidence=0.7,  # Default confidence
                timestamp=pd.Timestamp.now(),
                source='RSI',
                metadata={'indicator': 'RSI', 'value': technical.get('rsi_value', 'N/A')}
            ))
        
        # Example: convert MACD signal
        macd_signal = technical.get('MACD_daily')
        if macd_signal and isinstance(macd_signal, str):
            strength = 0.5 if macd_signal.lower() == 'buy' else -0.5 if macd_signal.lower() == 'sell' else 0.0
            signals.append(Signal(
                signal_type=SignalType.TECHNICAL,
                strength=strength,
                confidence=0.7,
                timestamp=pd.Timestamp.now(),
                source='MACD',
                metadata={'indicator': 'MACD', 'value': technical.get('macd_value', 'N/A')}
            ))
        
        # Add more technical indicators as needed
    
    # Convert fundamental signal
    if 'error' not in fundamental:
        fundamental_signal = fundamental.get('fundamental_signal')
        if fundamental_signal:
            strength = 0.6 if fundamental_signal == 'BUY' else -0.6 if fundamental_signal == 'SELL' else 0.0
            signals.append(Signal(
                signal_type=SignalType.FUNDAMENTAL,
                strength=strength,
                confidence=0.6,
                timestamp=pd.Timestamp.now(),
                source='Fundamental',
                metadata={'valuation': fundamental.get('valuations', 'N/A')}
            ))
    
    # Convert sentiment signal
    if 'error' not in sentiment:
        compound = sentiment.get('compound', 0)
        if compound > 0.1:
            strength = min(0.8, compound * 2)  # Scale to max 0.8
            confidence = min(1.0, abs(compound) * 2)
        elif compound < -0.1:
            strength = max(-0.8, compound * 2)  # Scale to min -0.8
            confidence = min(1.0, abs(compound) * 2)
        else:
            strength = 0.0
            confidence = 0.5
            
        signals.append(Signal(
            signal_type=SignalType.SENTIMENT,
            strength=strength,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            source='Sentiment',
            metadata={'compound': compound}
        ))
    
    # Convert risk signal
    if 'error' not in risk:
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
            
        signals.append(Signal(
            signal_type=SignalType.RISK,
            strength=risk_score,
            confidence=0.8 if 'volatility' in risk and 'sharpe_ratio' in risk else 0.5,
            timestamp=pd.Timestamp.now(),
            source='Risk',
            metadata={'volatility': volatility, 'sharpe': sharpe}
        ))
    
    # Convert macro signal
    if macro != 0:
        signals.append(Signal(
            signal_type=SignalType.MACRO,
            strength=macro,
            confidence=0.6,  # Macro typically has moderate confidence
            timestamp=pd.Timestamp.now(),
            source='Macro',
            metadata={'composite_score': macro}
        ))
    
    return signals


def _create_market_context_from_state(state: State, symbol: str) -> MarketContext:
    """Create a MarketContext object from the current state."""
    # Get market regime data from state
    # This is a simplified example - in a full implementation, this data would be calculated or fetched
    volatility_regime = state.get("volatility_regime", {}).get(symbol, "medium")
    trend_regime = state.get("trend_regime", {}).get(symbol, "transitional")
    market_sentiment = state.get("market_sentiment", {}).get(symbol, "neutral")
    correlation_regime = state.get("correlation_regime", {}).get(symbol, "medium")
    volume_regime = state.get("volume_regime", {}).get(symbol, "normal")
    
    return MarketContext(
        volatility_regime=volatility_regime,
        trend_regime=trend_regime,
        market_sentiment=market_sentiment,
        correlation_regime=correlation_regime,
        volume_regime=volume_regime
    )


def _parse_suggested_action(reasoning: str) -> Optional[str]:
    reasoning_lower = reasoning.lower()
    if "final recommendation: buy" in reasoning_lower:
        return "BUY"
    elif "final recommendation: sell" in reasoning_lower:
        return "SELL"
    elif "final recommendation: hold" in reasoning_lower:
        return "HOLD"
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