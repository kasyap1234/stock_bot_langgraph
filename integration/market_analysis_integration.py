"""
Enhanced Market Analysis Integration Module

This module provides real-time market analysis integration with adaptive decision-making
capabilities for the stock trading bot.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from agents.market_risk_assessment import MarketRiskAssessor, MarketRiskMetrics, MarketRegime, VolatilityRegime
from data.real_time_data import RealTimeDataManager
from recommendation.final_recommendation import EnhancedRecommendationEngine, MarketConditions
from config.config import DEBUG_RECOMMENDATION_LOGGING

logger = logging.getLogger(__name__)


class SignalConfidence(Enum):
    """Signal confidence levels for adaptive decision-making"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class AdaptiveSignal:
    """Enhanced signal with adaptive parameters"""
    symbol: str
    action: str
    confidence: float
    original_confidence: float
    adaptive_multiplier: float
    market_regime: str
    volatility_regime: str
    signal_strength: str
    timestamp: datetime
    reasoning: str
    factors: Dict[str, float]


class AdaptiveMarketAnalyzer:
    """Enhanced market analyzer with real-time adaptation capabilities"""
    
    def __init__(self):
        self.risk_assessor = MarketRiskAssessor()
        self.realtime_manager = RealTimeDataManager()
        self.recommendation_engine = EnhancedRecommendationEngine()
        self.signal_history: Dict[str, List[AdaptiveSignal]] = {}
        self.performance_tracker: Dict[str, List[float]] = {}
        self.adaptive_parameters = self._initialize_adaptive_params()
        self.callbacks: List[Callable] = []
        
    def _initialize_adaptive_params(self) -> Dict[str, Any]:
        """Initialize adaptive parameters based on market conditions"""
        return {
            'volatility_thresholds': {
                'low': 0.15,
                'high': 0.30,
                'extreme': 0.45
            },
            'confidence_multipliers': {
                'bull_market': 1.1,
                'bear_market': 0.8,
                'sideways_market': 0.95,
                'high_volatility': 0.7,
                'low_volatility': 1.2
            },
            'signal_decay_rates': {
                'technical': 0.05,    # 5% per hour
                'fundamental': 0.02,   # 2% per hour
                'sentiment': 0.08,     # 8% per hour
                'risk': 0.03,          # 3% per hour
                'macro': 0.01          # 1% per hour
            },
            'performance_windows': {
                'short': 5,    # 5 signals
                'medium': 20,  # 20 signals
                'long': 50     # 50 signals
            }
        }
    
    async def start_adaptive_analysis(self, symbols: List[str]) -> None:
        """Start adaptive market analysis for given symbols"""
        try:
            # Add callback for real-time data processing
            self.realtime_manager.add_callback(self._process_realtime_data)
            
            # Start real-time streaming
            await self.realtime_manager.start_streaming(symbols)
            
            # Initialize signal history for each symbol
            for symbol in symbols:
                if symbol not in self.signal_history:
                    self.signal_history[symbol] = []
                if symbol not in self.performance_tracker:
                    self.performance_tracker[symbol] = []
            
            logger.info(f"Started adaptive market analysis for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error starting adaptive analysis: {e}")
            raise
    
    async def stop_adaptive_analysis(self) -> None:
        """Stop adaptive market analysis"""
        try:
            await self.realtime_manager.stop_streaming()
            self.realtime_manager.remove_callback(self._process_realtime_data)
            logger.info("Stopped adaptive market analysis")
        except Exception as e:
            logger.error(f"Error stopping adaptive analysis: {e}")
    
    def _process_realtime_data(self, data: Dict[str, Any]) -> None:
        """Process real-time market data and update adaptive parameters"""
        try:
            symbol = data.get('symbol')
            if not symbol:
                return
            
            # Update market risk assessment
            market_data = {symbol: pd.DataFrame([data])}
            risk_metrics = self.risk_assessor.assess_market_risk(market_data)
            
            # Update recommendation engine with current market conditions
            market_conditions = self._convert_risk_to_market_conditions(risk_metrics)
            self.recommendation_engine.current_market_conditions = market_conditions
            
            # Generate adaptive signals if we have enough data
            if len(self.signal_history.get(symbol, [])) > 0:
                adaptive_signal = self._generate_adaptive_signal(symbol, data, risk_metrics)
                if adaptive_signal:
                    self._store_signal(symbol, adaptive_signal)
                    self._notify_callbacks(adaptive_signal)
            
            # Update adaptive parameters based on recent performance
            self._update_adaptive_parameters(symbol)
            
        except Exception as e:
            logger.error(f"Error processing real-time data: {e}")
    
    def _convert_risk_to_market_conditions(self, risk_metrics: MarketRiskMetrics) -> MarketConditions:
        """Convert risk metrics to market conditions format"""
        return MarketConditions(
            volatility_regime=risk_metrics.volatility_regime.value,
            trend_strength=self._assess_trend_strength(risk_metrics.market_trend),
            market_sentiment=self._assess_market_sentiment(risk_metrics),
            risk_environment=self._assess_risk_environment(risk_metrics),
            correlation_level=self._assess_correlation_level(risk_metrics.correlation_risk)
        )
    
    def _assess_trend_strength(self, market_trend: str) -> str:
        """Assess trend strength from market trend"""
        trend_mapping = {
            'strong_bull': 'strong',
            'bull': 'moderate',
            'neutral': 'weak',
            'bear': 'moderate',
            'strong_bear': 'strong'
        }
        return trend_mapping.get(market_trend, 'weak')
    
    def _assess_market_sentiment(self, risk_metrics: MarketRiskMetrics) -> str:
        """Assess market sentiment from risk metrics"""
        if risk_metrics.regime == MarketRegime.BULL:
            return 'bullish'
        elif risk_metrics.regime == MarketRegime.BEAR:
            return 'bearish'
        else:
            return 'neutral'
    
    def _assess_risk_environment(self, risk_metrics: MarketRiskMetrics) -> str:
        """Assess overall risk environment"""
        risk_score = 0
        
        # Volatility risk
        if risk_metrics.volatility_regime == VolatilityRegime.EXTREME:
            risk_score += 3
        elif risk_metrics.volatility_regime == VolatilityRegime.HIGH:
            risk_score += 2
        
        # Gap risk
        if risk_metrics.gap_risk > 0.05:
            risk_score += 2
        elif risk_metrics.gap_risk > 0.02:
            risk_score += 1
        
        # Liquidity risk
        if risk_metrics.liquidity_risk > 0.1:
            risk_score += 2
        
        # Correlation risk
        if risk_metrics.correlation_risk > 0.7:
            risk_score += 1
        
        if risk_score >= 5:
            return 'high_risk'
        elif risk_score >= 3:
            return 'moderate_risk'
        else:
            return 'low_risk'
    
    def _assess_correlation_level(self, correlation_risk: float) -> str:
        """Assess correlation level"""
        if correlation_risk > 0.8:
            return 'high'
        elif correlation_risk > 0.6:
            return 'moderate'
        else:
            return 'low'
    
    def _generate_adaptive_signal(self, symbol: str, data: Dict[str, Any], 
                                 risk_metrics: MarketRiskMetrics) -> Optional[AdaptiveSignal]:
        """Generate adaptive trading signal based on current market conditions"""
        try:
            # Get base recommendation from enhanced engine
            base_recommendation = self.recommendation_engine.synthesize_decision(
                symbol=symbol,
                technical=data.get('technical', {}),
                fundamental=data.get('fundamental', {}),
                sentiment=data.get('sentiment', {}),
                risk=data.get('risk', {}),
                macro=data.get('macro', 0.0),
                state=data.get('state')
            )
            
            if not base_recommendation:
                return None
            
            # Calculate adaptive multiplier based on market conditions
            adaptive_multiplier = self._calculate_adaptive_multiplier(
                risk_metrics, base_recommendation
            )
            
            # Apply signal decay based on factor freshness
            decayed_confidence = self._apply_signal_decay(
                base_recommendation['confidence'], 
                self.signal_history.get(symbol, [])
            )
            
            # Calculate final adaptive confidence
            final_confidence = decayed_confidence * adaptive_multiplier
            final_confidence = max(0.0, min(1.0, final_confidence))  # Bound between 0 and 1
            
            # Determine signal strength
            signal_strength = self._determine_signal_strength(final_confidence)
            
            return AdaptiveSignal(
                symbol=symbol,
                action=base_recommendation['action'],
                confidence=final_confidence,
                original_confidence=base_recommendation['confidence'],
                adaptive_multiplier=adaptive_multiplier,
                market_regime=risk_metrics.regime.value,
                volatility_regime=risk_metrics.volatility_regime.value,
                signal_strength=signal_strength,
                timestamp=datetime.now(),
                reasoning=base_recommendation['reasoning'],
                factors=base_recommendation.get('factors', {})
            )
            
        except Exception as e:
            logger.error(f"Error generating adaptive signal for {symbol}: {e}")
            return None
    
    def _calculate_adaptive_multiplier(self, risk_metrics: MarketRiskMetrics, 
                                     base_recommendation: Dict[str, Any]) -> float:
        """Calculate adaptive multiplier based on market conditions and signal history"""
        multiplier = 1.0
        
        # Market regime adjustments
        regime_multipliers = self.adaptive_parameters['confidence_multipliers']
        
        if risk_metrics.regime == MarketRegime.BULL:
            multiplier *= regime_multipliers['bull_market']
        elif risk_metrics.regime == MarketRegime.BEAR:
            multiplier *= regime_multipliers['bear_market']
        else:
            multiplier *= regime_multipliers['sideways_market']
        
        # Volatility adjustments
        if risk_metrics.volatility_regime == VolatilityRegime.HIGH:
            multiplier *= regime_multipliers['high_volatility']
        elif risk_metrics.volatility_regime == VolatilityRegime.LOW:
            multiplier *= regime_multipliers['low_volatility']
        elif risk_metrics.volatility_regime == VolatilityRegime.EXTREME:
            multiplier *= 0.5  # Severely reduce confidence in extreme volatility
        
        # Trend strength adjustments
        if hasattr(risk_metrics, 'trend_strength'):
            if risk_metrics.trend_strength == 'strong':
                multiplier *= 1.05
            elif risk_metrics.trend_strength == 'weak':
                multiplier *= 0.95
        
        return multiplier
    
    def _apply_signal_decay(self, confidence: float, signal_history: List[AdaptiveSignal]) -> float:
        """Apply signal decay based on time and market conditions"""
        if not signal_history:
            return confidence
        
        # Calculate weighted average age of recent signals
        current_time = datetime.now()
        total_weight = 0
        weighted_age = 0
        
        for i, signal in enumerate(reversed(signal_history[-10:])):  # Last 10 signals
            age_hours = (current_time - signal.timestamp).total_seconds() / 3600
            weight = 1.0 / (i + 1)  # More recent signals have higher weight
            weighted_age += age_hours * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_age = weighted_age / total_weight
        else:
            avg_age = 0
        
        # Apply decay based on average age (5% per hour default)
        decay_rate = 0.05
        decay_factor = max(0.5, 1.0 - (decay_rate * avg_age))  # Minimum 50% confidence retained
        
        return confidence * decay_factor
    
    def _determine_signal_strength(self, confidence: float) -> str:
        """Determine signal strength based on confidence level"""
        if confidence >= 0.8:
            return "strong"
        elif confidence >= 0.6:
            return "moderate"
        elif confidence >= 0.4:
            return "weak"
        else:
            return "uncertain"
    
    def _store_signal(self, symbol: str, signal: AdaptiveSignal) -> None:
        """Store adaptive signal in history"""
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        
        self.signal_history[symbol].append(signal)
        
        # Keep only recent signals (last 100)
        if len(self.signal_history[symbol]) > 100:
            self.signal_history[symbol] = self.signal_history[symbol][-100:]
    
    def _update_adaptive_parameters(self, symbol: str) -> None:
        """Update adaptive parameters based on recent performance"""
        try:
            if symbol not in self.signal_history or len(self.signal_history[symbol]) < 5:
                return
            
            recent_signals = self.signal_history[symbol][-20:]  # Last 20 signals
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(recent_signals)
            
            # Adjust parameters based on performance
            if performance['win_rate'] < 0.4:  # Poor performance
                # Reduce confidence multipliers
                for key in self.adaptive_parameters['confidence_multipliers']:
                    self.adaptive_parameters['confidence_multipliers'][key] *= 0.95
                
                # Increase decay rates to reduce stale signal impact
                for key in self.adaptive_parameters['signal_decay_rates']:
                    self.adaptive_parameters['signal_decay_rates'][key] *= 1.1
                    
            elif performance['win_rate'] > 0.6:  # Good performance
                # Increase confidence multipliers
                for key in self.adaptive_parameters['confidence_multipliers']:
                    self.adaptive_parameters['confidence_multipliers'][key] *= 1.05
                
                # Decay rates can remain the same or slightly decrease
                for key in self.adaptive_parameters['signal_decay_rates']:
                    self.adaptive_parameters['signal_decay_rates'][key] *= 0.95
            
            # Log parameter updates
            if DEBUG_RECOMMENDATION_LOGGING:
                logger.debug(f"Updated adaptive parameters for {symbol}: win_rate={performance['win_rate']:.2f}")
                
        except Exception as e:
            logger.error(f"Error updating adaptive parameters for {symbol}: {e}")
    
    def _calculate_performance_metrics(self, signals: List[AdaptiveSignal]) -> Dict[str, float]:
        """Calculate performance metrics from signal history"""
        if not signals:
            return {'win_rate': 0.5, 'avg_confidence': 0.5}
        
        # Simple win rate calculation based on signal confidence and market direction
        # This is a simplified version - in practice, you'd track actual trade outcomes
        wins = 0
        total_signals = 0
        total_confidence = 0
        
        for signal in signals:
            total_signals += 1
            total_confidence += signal.confidence
            
            # Simplified win determination based on confidence and signal strength
            if signal.signal_strength in ['strong', 'moderate'] and signal.confidence > 0.6:
                wins += 1
        
        win_rate = wins / total_signals if total_signals > 0 else 0.5
        avg_confidence = total_confidence / total_signals if total_signals > 0 else 0.5
        
        return {
            'win_rate': win_rate,
            'avg_confidence': avg_confidence
        }
    
    def add_callback(self, callback: Callable) -> None:
        """Add callback for adaptive signals"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _notify_callbacks(self, signal: AdaptiveSignal) -> None:
        """Notify all callbacks of new adaptive signal"""
        for callback in self.callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_adaptive_parameters(self) -> Dict[str, Any]:
        """Get current adaptive parameters"""
        return self.adaptive_parameters.copy()
    
    def set_adaptive_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set adaptive parameters"""
        self.adaptive_parameters.update(parameters)
    
    def get_signal_history(self, symbol: str) -> List[AdaptiveSignal]:
        """Get signal history for a symbol"""
        return self.signal_history.get(symbol, []).copy()
    
    def get_performance_summary(self, symbol: str) -> Dict[str, Any]:
        """Get performance summary for a symbol"""
        signals = self.signal_history.get(symbol, [])
        if not signals:
            return {'error': 'No signals available'}
        
        performance = self._calculate_performance_metrics(signals)
        
        return {
            'total_signals': len(signals),
            'win_rate': performance['win_rate'],
            'avg_confidence': performance['avg_confidence'],
            'recent_signals': len(signals[-10:]),
            'adaptive_parameters': self.get_adaptive_parameters()
        }