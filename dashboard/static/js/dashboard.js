// Trading Dashboard JavaScript

class TradingDashboard {
    constructor() {
        this.ws = null;
        this.charts = {};
        this.currentSection = 'overview';
        this.isConnected = false;

        this.init();
    }

    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.loadInitialData();
        this.initializeCharts();
        this.startPeriodicUpdates();
    }

    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${Date.now()}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.isConnected = true;
            this.showConnectionStatus(true);
            // FIXED: Removed console.log, using structured logging if needed
            logger.info('WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (e) {
                // FIXED: Removed console.error, using structured logging if needed
                logger.error('WebSocket message parse error:', e);
            }
        };

        this.ws.onclose = () => {
            this.isConnected = false;
            this.showConnectionStatus(false);
            // FIXED: Removed console.log, using structured logging if needed
            logger.warn('WebSocket disconnected');

            // Attempt to reconnect after 5 seconds
            setTimeout(() => this.setupWebSocket(), 5000);
        };

        this.ws.onerror = (error) => {
            // FIXED: Removed console.error, using structured logging if needed
            logger.error('WebSocket error:', error);
        };
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = e.target.closest('.nav-link').getAttribute('onclick').match(/'([^']+)'/)[1];
                this.showSection(section);
            });
        });

        // Backtest form
        document.getElementById('backtestStrategy').addEventListener('change', () => {
            this.updateBacktestParams();
        });
    }

    showSection(sectionName) {
        // Hide all sections
        document.querySelectorAll('.dashboard-section').forEach(section => {
            section.classList.remove('active');
        });

        // Show selected section
        document.getElementById(`${sectionName}Section`).classList.add('active');

        // Update navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[onclick="showSection('${sectionName}')"]`).classList.add('active');

        this.currentSection = sectionName;

        // Load section-specific data
        this.loadSectionData(sectionName);
    }

    async loadInitialData() {
        try {
            const [portfolio, strategies, risk, market, alerts] = await Promise.all([
                this.fetchData('/api/portfolio'),
                this.fetchData('/api/strategies'),
                this.fetchData('/api/risk-metrics'),
                this.fetchData('/api/market-data'),
                this.fetchData('/api/alerts')
            ]);

            this.updatePortfolioData(portfolio);
            this.updateStrategiesData(strategies);
            this.updateRiskData(risk);
            this.updateMarketData(market);
            this.updateAlerts(alerts);

        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showError('Failed to load dashboard data');
        }
    }

    async loadSectionData(section) {
        switch (section) {
            case 'portfolio':
                const portfolio = await this.fetchData('/api/portfolio');
                this.updatePortfolioData(portfolio);
                break;
            case 'strategies':
                const strategies = await this.fetchData('/api/strategies');
                this.updateStrategiesData(strategies);
                break;
            case 'risk':
                const risk = await this.fetchData('/api/risk-metrics');
                this.updateRiskData(risk);
                break;
            case 'market':
                const market = await this.fetchData('/api/market-data');
                this.updateMarketData(market);
                break;
        }
    }

    async fetchData(url) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return await response.json();
    }

    updatePortfolioData(data) {
        // Update overview cards
        const portfolioValue = data.final_portfolio_value || 0;
        const totalReturn = data.total_return || 0;
        const maxDrawdown = data.max_drawdown || 0;

        document.getElementById('portfolioValue').textContent = `₹${portfolioValue.toLocaleString('en-IN', {maximumFractionDigits: 2})}`;
        document.getElementById('totalPnL').textContent = `₹${(portfolioValue * totalReturn).toLocaleString('en-IN', {maximumFractionDigits: 2})}`;
        document.getElementById('todayReturn').textContent = `${(totalReturn * 100).toFixed(2)}%`;
        document.getElementById('drawdown').textContent = `${(maxDrawdown * 100).toFixed(2)}%`;

        // Update portfolio table
        this.updatePortfolioTable(data.positions || {});
    }

    updatePortfolioTable(positions) {
        const tbody = document.getElementById('portfolioTableBody');
        tbody.innerHTML = '';

        if (Object.keys(positions).length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No holdings data available</td></tr>';
            return;
        }

        Object.entries(positions).forEach(([symbol, position]) => {
            const row = document.createElement('tr');

            const pnl = position.unrealized_pnl || 0;
            const pnlPercent = position.unrealized_pnl_percent || 0;

            row.innerHTML = `
                <td>${symbol}</td>
                <td>${position.quantity || 0}</td>
                <td>₹${(position.avg_price || 0).toFixed(2)}</td>
                <td>₹${(position.current_price || 0).toFixed(2)}</td>
                <td>₹${(position.market_value || 0).toFixed(2)}</td>
                <td class="${pnl >= 0 ? 'text-profit' : 'text-loss'}">₹${pnl.toFixed(2)}</td>
                <td class="${pnlPercent >= 0 ? 'text-profit' : 'text-loss'}">${pnlPercent.toFixed(2)}%</td>
            `;

            tbody.appendChild(row);
        });
    }

    updateStrategiesData(data) {
        // Update active strategies count
        const activeCount = Object.keys(data).length;
        document.getElementById('activeStrategies').textContent = activeCount;

        // Update strategies chart
        this.updateStrategiesChart(data);
    }

    updateRiskData(data) {
        // Update risk score
        const riskLevel = this.calculateRiskLevel(data);
        document.getElementById('riskScore').textContent = riskLevel;
        document.getElementById('riskScore').className = `text-${riskLevel.toLowerCase()}`;

        // Update risk chart
        this.updateRiskChart(data);

        // Update risk alerts
        this.updateRiskAlerts(data);
    }

    updateMarketData(data) {
        const tbody = document.getElementById('marketTableBody');
        tbody.innerHTML = '';

        Object.entries(data).forEach(([symbol, marketData]) => {
            const row = document.createElement('tr');

            const change = marketData.change || 0;
            const changePercent = marketData.change_percent || 0;

            row.innerHTML = `
                <td>${symbol}</td>
                <td>₹${(marketData.price || 0).toFixed(2)}</td>
                <td class="${change >= 0 ? 'text-profit' : 'text-loss'}">${change.toFixed(2)}</td>
                <td class="${changePercent >= 0 ? 'text-profit' : 'text-loss'}">${changePercent.toFixed(2)}%</td>
                <td>${(marketData.volume || 0).toLocaleString('en-IN')}</td>
                <td>₹${(marketData.market_cap || 0).toLocaleString('en-IN')}</td>
            `;

            tbody.appendChild(row);
        });
    }

    updateAlerts(alerts) {
        const container = document.getElementById('alertsContainer');
        container.innerHTML = '';

        if (!alerts || alerts.length === 0) {
            return;
        }

        alerts.forEach(alert => {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${alert.type === 'warning' ? 'warning' : 'danger'} alert-custom fade show`;
            alertDiv.innerHTML = `
                <i class="fas fa-exclamation-triangle"></i>
                <strong>${alert.type.toUpperCase()}:</strong> ${alert.message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            container.appendChild(alertDiv);
        });
    }

    updateRiskAlerts(riskData) {
        const container = document.getElementById('riskAlerts');
        container.innerHTML = '';

        // Check for risk thresholds
        const alerts = [];

        Object.entries(riskData).forEach(([symbol, metrics]) => {
            const volatility = metrics.volatility || 0;
            const var95 = metrics.var_95 || 0;

            if (volatility > 0.25) {
                alerts.push({
                    type: 'warning',
                    message: `High volatility for ${symbol}: ${(volatility * 100).toFixed(1)}%`
                });
            }

            if (Math.abs(var95) > 0.05) {
                alerts.push({
                    type: 'danger',
                    message: `High VaR for ${symbol}: ${(var95 * 100).toFixed(1)}%`
                });
            }
        });

        if (alerts.length === 0) {
            container.innerHTML = '<p class="text-muted">No active risk alerts</p>';
            return;
        }

        alerts.forEach(alert => {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${alert.type} alert-custom mb-2`;
            alertDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${alert.message}`;
            container.appendChild(alertDiv);
        });
    }

    calculateRiskLevel(riskData) {
        // Simple risk scoring based on various metrics
        let riskScore = 0;

        Object.values(riskData).forEach(metrics => {
            if (metrics.volatility > 0.2) riskScore += 1;
            if (Math.abs(metrics.var_95 || 0) > 0.03) riskScore += 1;
            if (metrics.sharpe_ratio < 1) riskScore += 1;
        });

        if (riskScore <= 1) return 'Low';
        if (riskScore <= 3) return 'Medium';
        return 'High';
    }

    initializeCharts() {
        // Portfolio performance chart
        const portfolioCtx = document.getElementById('portfolioChart').getContext('2d');
        this.charts.portfolio = new Chart(portfolioCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Portfolio Value',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '₹' + value.toLocaleString('en-IN');
                            }
                        }
                    }
                }
            }
        });

        // Strategies performance chart
        const strategiesCtx = document.getElementById('strategiesChart').getContext('2d');
        this.charts.strategies = new Chart(strategiesCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Win Rate',
                    data: [],
                    backgroundColor: '#28a745'
                }, {
                    label: 'Sharpe Ratio',
                    data: [],
                    backgroundColor: '#ffc107'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });

        // Risk metrics chart
        const riskCtx = document.getElementById('riskChart').getContext('2d');
        this.charts.risk = new Chart(riskCtx, {
            type: 'radar',
            data: {
                labels: ['Volatility', 'VaR 95%', 'Max Drawdown', 'Sharpe Ratio', 'Sortino Ratio'],
                datasets: [{
                    label: 'Risk Metrics',
                    data: [],
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    pointBackgroundColor: '#dc3545'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }

    updateStrategiesChart(data) {
        const labels = Object.keys(data);
        const winRates = labels.map(symbol => data[symbol].win_rate || 0);
        const sharpeRatios = labels.map(symbol => data[symbol].sharpe_ratio || 0);

        this.charts.strategies.data.labels = labels;
        this.charts.strategies.data.datasets[0].data = winRates;
        this.charts.strategies.data.datasets[1].data = sharpeRatios;
        this.charts.strategies.update();
    }

    updateRiskChart(data) {
        // Aggregate risk metrics across all symbols
        const aggregated = {
            volatility: 0,
            var_95: 0,
            max_drawdown: 0,
            sharpe_ratio: 0,
            sortino_ratio: 0
        };

        const symbols = Object.keys(data);
        symbols.forEach(symbol => {
            const metrics = data[symbol];
            aggregated.volatility += metrics.volatility || 0;
            aggregated.var_95 += Math.abs(metrics.var_95 || 0);
            aggregated.max_drawdown += metrics.max_drawdown || 0;
            aggregated.sharpe_ratio += metrics.sharpe_ratio || 0;
            aggregated.sortino_ratio += metrics.sortino_ratio || 0;
        });

        // Average the metrics
        const count = symbols.length || 1;
        Object.keys(aggregated).forEach(key => {
            aggregated[key] /= count;
        });

        this.charts.risk.data.datasets[0].data = Object.values(aggregated);
        this.charts.risk.update();
    }

    handleWebSocketMessage(data) {
        if (data.type === 'dashboard_update') {
            this.updateMarketData(data.data.market_data || {});
            this.updateAlerts(data.data.alerts || []);
        }
    }

    showConnectionStatus(connected) {
        const statusDiv = document.getElementById('wsStatus');
        if (connected) {
            statusDiv.style.display = 'block';
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 3000);
        }
    }

    async runAnalysis() {
        try {
            this.showLoading('Running analysis...');

            const response = await fetch('/api/run-analysis', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    stocks: ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
                })
            });

            if (response.ok) {
                this.showSuccess('Analysis completed successfully');
                await this.loadInitialData();
            } else {
                throw new Error('Analysis failed');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('Analysis failed. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    async runBacktest() {
        try {
            this.showLoading('Running backtest...');

            const strategy = document.getElementById('backtestStrategy').value;
            const startDate = document.getElementById('backtestStartDate').value;
            const endDate = document.getElementById('backtestEndDate').value;

            const response = await fetch('/api/run-backtest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    strategy: strategy,
                    start_date: startDate,
                    end_date: endDate
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.displayBacktestResults(result);
                this.showSuccess('Backtest completed successfully');
            } else {
                throw new Error('Backtest failed');
            }
        } catch (error) {
            console.error('Backtest error:', error);
            this.showError('Backtest failed. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    displayBacktestResults(result) {
        const container = document.getElementById('backtestResults');
        const data = result.results;

        container.innerHTML = `
            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body text-center">
                            <h5>Total Return</h5>
                            <h3 class="${data.total_return >= 0 ? 'text-success' : 'text-danger'}">
                                ${(data.total_return * 100).toFixed(2)}%
                            </h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body text-center">
                            <h5>Max Drawdown</h5>
                            <h3 class="text-warning">${(data.max_drawdown * 100).toFixed(2)}%</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body text-center">
                            <h5>Sharpe Ratio</h5>
                            <h3 class="text-info">${(data.sharpe_ratio || 0).toFixed(2)}</h3>
                        </div>
                    </div>
                </div>
            </div>
            <div class="mt-4">
                <h6>Performance Analysis</h6>
                <p>${result.analysis.performance_rating || 'Analysis pending'}</p>
            </div>
        `;
    }

    updateBacktestParams() {
        // Update backtest parameters based on selected strategy
        const strategy = document.getElementById('backtestStrategy').value;

        // Set default date range (last 2 years)
        const endDate = new Date();
        const startDate = new Date();
        startDate.setFullYear(endDate.getFullYear() - 2);

        document.getElementById('backtestStartDate').value = startDate.toISOString().split('T')[0];
        document.getElementById('backtestEndDate').value = endDate.toISOString().split('T')[0];
    }

    startPeriodicUpdates() {
        // Update data every 5 minutes
        setInterval(() => {
            if (this.isConnected) {
                this.loadSectionData(this.currentSection);
            }
        }, 300000);
    }

    showLoading(message = 'Loading...') {
        // Create loading overlay
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.id = 'loadingOverlay';
        overlay.innerHTML = `
            <div class="text-center">
                <div class="loading-spinner mb-2"></div>
                <p class="mb-0">${message}</p>
            </div>
        `;

        document.body.appendChild(overlay);
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.remove();
        }
    }

    showSuccess(message) {
        this.showToast(message, 'success');
    }

    showError(message) {
        this.showToast(message, 'danger');
    }

    showToast(message, type) {
        const toastContainer = document.createElement('div');
        toastContainer.className = 'position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';

        toastContainer.innerHTML = `
            <div class="toast show" role="alert">
                <div class="toast-body alert alert-${type}">
                    ${message}
                    <button type="button" class="btn-close ms-2" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;

        document.body.appendChild(toastContainer);

        // Auto remove after 5 seconds
        setTimeout(() => {
            toastContainer.remove();
        }, 5000);
    }
}

async function logout() {
    try {
        await fetch('/auth/logout', { method: 'POST' });
        window.location.href = '/login';
    } catch (error) {
        console.error('Logout error:', error);
        window.location.href = '/login';
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new TradingDashboard();
});