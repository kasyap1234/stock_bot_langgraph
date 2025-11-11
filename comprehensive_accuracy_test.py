#!/usr/bin/env python3
"""
Comprehensive Accuracy Test for Stock Recommendation Bot
Tests Reliance, TCS, and Infosys with detailed metrics and confidence analysis
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.models import State, HistoricalData
from data.apis import get_stock_history
from data.ingest import clean_stock_data
from agents.technical_analysis import TechnicalAnalysisPipeline
from agents.fundamental_analysis import fundamental_analysis_agent
from agents.sentiment_analysis import sentiment_analysis_agent
from agents.risk_assessment import risk_assessment_agent
from agents.macro_analysis import macro_analysis_agent
from recommendation.final_recommendation import final_recommendation_agent, EnhancedRecommendationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'accuracy_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AccuracyTestReport:
    """Comprehensive test report generator"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'stocks_tested': [],
            'overall_metrics': {},
            'individual_results': {},
            'issues_found': [],
            'confidence_analysis': {},
            'feature_functionality': {}
        }
    
    def add_stock_result(self, symbol: str, result: Dict[str, Any]):
        """Add results for a specific stock"""
        self.results['individual_results'][symbol] = result
        self.results['stocks_tested'].append(symbol)
    
    def add_issue(self, category: str, issue: str, severity: str = "medium"):
        """Add an issue found during testing"""
        self.results['issues_found'].append({
            'category': category,
            'issue': issue,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("=" * 80)
        report.append("STOCK RECOMMENDATION BOT - ACCURACY TEST REPORT")
        report.append("=" * 80)
        report.append(f"Test Date: {self.results['timestamp']}")
        report.append(f"Stocks Tested: {', '.join(self.results['stocks_tested'])}")
        report.append("")
        
        # Overall Metrics
        if self.results['overall_metrics']:
            report.append("OVERALL METRICS")
            report.append("-" * 40)
            for metric, value in self.results['overall_metrics'].items():
                report.append(f"{metric}: {value}")
            report.append("")
        
        # Individual Stock Results
        report.append("INDIVIDUAL STOCK ANALYSIS")
        report.append("-" * 40)
        for symbol, result in self.results['individual_results'].items():
            report.append(f"\n{symbol}:")
            report.append(f"  Recommendation: {result.get('recommendation', 'N/A')}")
            report.append(f"  Confidence: {result.get('confidence', 0):.2f}")
            report.append(f"  Composite Score: {result.get('composite_score', 0):.3f}")
            report.append(f"  Data Quality: {result.get('data_quality', 'N/A')}")
            report.append(f"  Issues: {len(result.get('issues', []))}")
        
        # Issues Found
        if self.results['issues_found']:
            report.append("\nISSUES IDENTIFIED")
            report.append("-" * 40)
            for issue in self.results['issues_found']:
                report.append(f"[{issue['severity'].upper()}] {issue['category']}: {issue['issue']}")
        
        # Feature Functionality
        if self.results['feature_functionality']:
            report.append("\nFEATURE FUNCTIONALITY ASSESSMENT")
            report.append("-" * 40)
            for feature, status in self.results['feature_functionality'].items():
                report.append(f"{feature}: {status}")
        
        return "\n".join(report)

class StockAccuracyTester:
    """Individual stock accuracy tester"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.state = State()
        self.pipeline = TechnicalAnalysisPipeline()
        self.engine = EnhancedRecommendationEngine()
        self.issues = []
        
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test for the stock"""
        logger.info(f"Starting comprehensive test for {self.symbol}")
        result = {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'recommendation': None,
            'confidence': 0.0,
            'composite_score': 0.0,
            'data_quality': 'unknown',
            'issues': [],
            'component_results': {},
            'confidence_breakdown': {}
        }
        
        try:
            # Step 1: Data Collection and Quality Assessment
            logger.info(f"Step 1: Data collection for {self.symbol}")
            data_quality = self._assess_data_quality()
            result['data_quality'] = data_quality['status']
            if data_quality['issues']:
                result['issues'].extend(data_quality['issues'])
            
            # Step 2: Technical Analysis
            logger.info(f"Step 2: Technical analysis for {self.symbol}")
            tech_result = self._test_technical_analysis()
            result['component_results']['technical'] = tech_result
            
            # Step 3: Fundamental Analysis
            logger.info(f"Step 3: Fundamental analysis for {self.symbol}")
            fund_result = self._test_fundamental_analysis()
            result['component_results']['fundamental'] = fund_result
            
            # Step 4: Sentiment Analysis
            logger.info(f"Step 4: Sentiment analysis for {self.symbol}")
            sentiment_result = self._test_sentiment_analysis()
            result['component_results']['sentiment'] = sentiment_result
            
            # Step 5: Risk Assessment
            logger.info(f"Step 5: Risk assessment for {self.symbol}")
            risk_result = self._test_risk_assessment()
            result['component_results']['risk'] = risk_result
            
            # Step 6: Macro Analysis
            logger.info(f"Step 6: Macro analysis for {self.symbol}")
            macro_result = self._test_macro_analysis()
            result['component_results']['macro'] = macro_result
            
            # Step 7: Final Recommendation
            logger.info(f"Step 7: Final recommendation for {self.symbol}")
            final_result = self._test_final_recommendation()
            result['recommendation'] = final_result['action']
            result['confidence'] = final_result['confidence']
            result['composite_score'] = final_result['composite_score']
            result['confidence_breakdown'] = final_result['confidence_breakdown']
            
            logger.info(f"Completed test for {self.symbol}: {result['recommendation']} (confidence: {result['confidence']:.2f})")
            
        except Exception as e:
            logger.error(f"Critical error testing {self.symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            result['issues'].append(f"Critical test failure: {str(e)}")
        
        return result
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess data quality for the stock"""
        try:
            # Fetch historical data
            raw_data = get_stock_history(self.symbol, period="6mo", interval="1d")
            if not raw_data:
                return {'status': 'failed', 'issues': ['No data retrieved']}
            
            # Clean data
            cleaned_data = clean_stock_data(raw_data)
            if not cleaned_data:
                return {'status': 'failed', 'issues': ['Data cleaning failed']}
            
            # Store in state
            self.state["stock_data"] = {self.symbol: pd.DataFrame(cleaned_data)}
            
            # Quality checks
            issues = []
            df = pd.DataFrame(cleaned_data)
            
            if len(df) < 50:
                issues.append("Insufficient data points")
            
            if df['close'].isna().sum() > 0:
                issues.append("Missing close prices")
            
            if (df['close'] <= 0).any():
                issues.append("Zero or negative prices")
            
            if df['volume'].isna().sum() > 0:
                issues.append("Missing volume data")
            
            status = 'good' if not issues else 'warning' if len(issues) < 3 else 'poor'
            
            return {'status': status, 'issues': issues, 'data_points': len(df)}
            
        except Exception as e:
            return {'status': 'error', 'issues': [f"Data quality assessment failed: {str(e)}"]}
    
    def _test_technical_analysis(self) -> Dict[str, Any]:
        """Test technical analysis components"""
        try:
            df = self.state["stock_data"][self.symbol]
            
            # Create a temporary state with just this stock's data for the pipeline
            temp_state = {"stock_data": {self.symbol: df}}
            
            # Run technical analysis pipeline using the run method
            pipeline_result = self.pipeline.run(temp_state)
            technical_signals = pipeline_result.get("technical_signals", {})
            signals = technical_signals.get(self.symbol, {})
            
            # Store in state
            self.state["technical_signals"] = {self.symbol: signals}
            
            # Analyze results
            result = {
                'status': 'success',
                'signals_generated': len(signals) if signals else 0,
                'signal_types': list(signals.keys()) if signals else [],
                'issues': []
            }
            
            if not signals:
                result['issues'].append("No technical signals generated")
            elif 'error' in signals:
                result['issues'].append(f"Technical analysis error: {signals['error']}")
                result['status'] = 'error'
            
            # Check for basic indicators in the raw_signals
            raw_signals = signals.get('raw_signals', {})
            basic_indicators = ['RSI', 'MACD', 'SMA', 'EMA']
            missing_basic = [ind for ind in basic_indicators if ind not in raw_signals]
            if missing_basic:
                result['issues'].append(f"Missing basic indicators: {missing_basic}")
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'signals_generated': 0, 'signal_types': [], 'issues': [str(e)]}
    
    def _test_fundamental_analysis(self) -> Dict[str, Any]:
        """Test fundamental analysis"""
        try:
            # Run fundamental analysis using the function
            result_state = fundamental_analysis_agent(self.state)
            fundamental_data = result_state.get("fundamental_analysis", {}).get(self.symbol, {})
            
            # Store in state
            self.state["fundamental_analysis"] = {self.symbol: fundamental_data}
            
            result = {
                'status': 'success',
                'metrics_available': len(fundamental_data) if fundamental_data else 0,
                'valuation_status': fundamental_data.get('valuations', 'unknown') if fundamental_data else 'unknown',
                'issues': []
            }
            
            if not fundamental_data or 'error' in fundamental_data:
                result['issues'].append("Fundamental analysis failed or returned errors")
                result['status'] = 'warning'
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'metrics_available': 0, 'valuation_status': 'error', 'issues': [str(e)]}
    
    def _test_sentiment_analysis(self) -> Dict[str, Any]:
        """Test sentiment analysis"""
        try:
            # Run sentiment analysis using the function
            result_state = sentiment_analysis_agent(self.state)
            sentiment_data = result_state.get("sentiment_scores", {}).get(self.symbol, {})
            
            # Store in state
            self.state["sentiment_scores"] = {self.symbol: sentiment_data}
            
            result = {
                'status': 'success',
                'sentiment_score': sentiment_data.get('compound', 0) if sentiment_data else 0,
                'news_count': len(sentiment_data.get('news', [])) if sentiment_data else 0,
                'issues': []
            }
            
            if not sentiment_data or 'error' in sentiment_data:
                result['issues'].append("Sentiment analysis failed")
                result['status'] = 'warning'
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'sentiment_score': 0, 'news_count': 0, 'issues': [str(e)]}
    
    def _test_risk_assessment(self) -> Dict[str, Any]:
        """Test risk assessment"""
        try:
            # Run risk assessment using the function
            result_state = risk_assessment_agent(self.state)
            risk_data = result_state.get("risk_metrics", {}).get(self.symbol, {})
            
            # Store in state
            self.state["risk_metrics"] = {self.symbol: risk_data}
            
            result = {
                'status': 'success',
                'risk_level': risk_data.get('risk_level', 'unknown') if risk_data else 'unknown',
                'volatility': risk_data.get('volatility', 0) if risk_data else 0,
                'sharpe_ratio': risk_data.get('sharpe_ratio', 0) if risk_data else 0,
                'issues': []
            }
            
            if not risk_data or 'error' in risk_data:
                result['issues'].append("Risk assessment failed")
                result['status'] = 'warning'
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'risk_level': 'error', 'volatility': 0, 'sharpe_ratio': 0, 'issues': [str(e)]}
    
    def _test_macro_analysis(self) -> Dict[str, Any]:
        """Test macro analysis"""
        try:
            # Run macro analysis using the function
            result_state = macro_analysis_agent(self.state)
            macro_data = result_state.get("macro_scores", {})
            
            # Store in state
            self.state["macro_scores"] = macro_data
            
            result = {
                'status': 'success',
                'composite_score': macro_data.get('composite', 0) if macro_data else 0,
                'components_analyzed': len(macro_data) if macro_data else 0,
                'issues': []
            }
            
            if not macro_data or 'error' in macro_data:
                result['issues'].append("Macro analysis failed")
                result['status'] = 'warning'
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'composite_score': 0, 'components_analyzed': 0, 'issues': [str(e)]}
    
    def _test_final_recommendation(self) -> Dict[str, Any]:
        """Test final recommendation generation"""
        try:
            # Generate final recommendation
            final_state = final_recommendation_agent(self.state)
            
            recommendation = final_state.get("final_recommendation", {}).get(self.symbol, {})
            
            # Extract confidence breakdown
            confidence_breakdown = {}
            if "factors" in locals():
                # This would come from the engine if we had direct access
                pass
            
            result = {
                'action': recommendation.get('action', 'HOLD'),
                'confidence': recommendation.get('confidence', 0),
                'composite_score': recommendation.get('composite_score', 0),
                'reasoning': recommendation.get('reasoning', ''),
                'confidence_breakdown': confidence_breakdown,
                'issues': []
            }
            
            if not recommendation or recommendation.get('action') == 'HOLD' and recommendation.get('confidence', 0) < 0.3:
                result['issues'].append("Low confidence HOLD recommendation")
            
            return result
            
        except Exception as e:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'composite_score': 0,
                'reasoning': f"Error: {str(e)}",
                'confidence_breakdown': {},
                'issues': [str(e)]
            }

def main():
    """Main test execution"""
    logger.info("Starting comprehensive accuracy test for stock recommendation bot")
    
    # Test symbols
    test_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    
    # Initialize report
    report = AccuracyTestReport()
    
    # Test each stock
    for symbol in test_symbols:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {symbol}")
            logger.info(f"{'='*60}")
            
            tester = StockAccuracyTester(symbol)
            result = tester.run_comprehensive_test()
            
            report.add_stock_result(symbol, result)
            
            # Add issues to global report
            for issue in result.get('issues', []):
                report.add_issue('general', issue, 'medium')
            
            for component, comp_result in result.get('component_results', {}).items():
                for issue in comp_result.get('issues', []):
                    report.add_issue(component, issue, 'medium')
            
            # Assess feature functionality
            for component, comp_result in result.get('component_results', {}).items():
                if comp_result.get('status') == 'success':
                    report.results['feature_functionality'][component] = 'functional'
                elif comp_result.get('status') == 'warning':
                    report.results['feature_functionality'][component] = 'partial'
                else:
                    report.results['feature_functionality'][component] = 'failed'
            
        except Exception as e:
            logger.error(f"Failed to test {symbol}: {str(e)}")
            report.add_issue('test_execution', f"Failed to test {symbol}: {str(e)}", 'high')
    
    # Calculate overall metrics
    all_confidences = []
    all_scores = []
    for symbol, result in report.results['individual_results'].items():
        all_confidences.append(result.get('confidence', 0))
        all_scores.append(result.get('composite_score', 0))
    
    if all_confidences:
        report.results['overall_metrics']['average_confidence'] = f"{np.mean(all_confidences):.2f}"
        report.results['overall_metrics']['confidence_std'] = f"{np.std(all_confidences):.2f}"
    
    if all_scores:
        report.results['overall_metrics']['average_composite_score'] = f"{np.mean(all_scores):.3f}"
    
    report.results['overall_metrics']['total_issues'] = len(report.results['issues_found'])
    report.results['overall_metrics']['functional_components'] = sum(1 for f in report.results['feature_functionality'].values() if f == 'functional')
    
    # Generate and save report
    final_report = report.generate_report()
    print("\n" + final_report)
    
    # Save detailed JSON report
    report_filename = f"accuracy_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(report.results, f, indent=2, default=str)
    
    logger.info(f"\nDetailed report saved to: {report_filename}")
    logger.info("Comprehensive accuracy test completed")

if __name__ == "__main__":
    main()