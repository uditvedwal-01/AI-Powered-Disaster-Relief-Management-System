/**
 * ML Charts and Visualizations for Disaster Resource Management System
 */

// Chart.js configuration
Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.color = '#6c757d';

class MLCharts {
    constructor() {
        this.charts = {};
    }

    /**
     * Create demand forecasting chart
     */
    createDemandChart(canvasId, predictions) {
        const ctx = document.getElementById(canvasId);
        if (!ctx || !predictions) return;

        const items = Object.keys(predictions);
        const currentStock = items.map(item => predictions[item].current_stock);
        const predictedDemand = items.map(item => predictions[item].predicted_demand);
        const itemNames = items.map(item => predictions[item].item_name);

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: itemNames,
                datasets: [
                    {
                        label: 'Current Stock',
                        data: currentStock,
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Predicted Demand',
                        data: predictedDemand,
                        backgroundColor: 'rgba(255, 206, 86, 0.6)',
                        borderColor: 'rgba(255, 206, 86, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Demand Forecasting Analysis'
                    },
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Quantity'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Items'
                        }
                    }
                }
            }
        });
    }

    /**
     * Create risk assessment gauge
     */
    createRiskGauge(canvasId, riskScore, riskLevel) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        const maxScore = 100;
        const percentage = (riskScore / maxScore) * 100;
        
        let color;
        if (riskLevel === 'High') color = '#dc3545';
        else if (riskLevel === 'Medium') color = '#ffc107';
        else color = '#28a745';

        new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [percentage, 100 - percentage],
                    backgroundColor: [color, '#e9ecef'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });

        // Add center text
        const centerText = document.createElement('div');
        centerText.style.position = 'absolute';
        centerText.style.top = '50%';
        centerText.style.left = '50%';
        centerText.style.transform = 'translate(-50%, -50%)';
        centerText.style.textAlign = 'center';
        centerText.innerHTML = `
            <div style="font-size: 24px; font-weight: bold; color: ${color}">${riskScore}</div>
            <div style="font-size: 12px; color: #6c757d">${riskLevel}</div>
        `;
        
        const container = ctx.parentElement;
        container.style.position = 'relative';
        container.appendChild(centerText);
    }

    /**
     * Create trend analysis chart
     */
    createTrendChart(canvasId, trendData) {
        const ctx = document.getElementById(canvasId);
        if (!ctx || !trendData) return;

        // Generate sample trend data for visualization
        const days = 30;
        const dates = [];
        const values = [];
        
        for (let i = 0; i < days; i++) {
            const date = new Date();
            date.setDate(date.getDate() - (days - i - 1));
            dates.push(date.toLocaleDateString());
            
            // Simulate trend based on trend direction
            let baseValue = trendData.avg_daily || 5;
            if (trendData.trend_direction === 'increasing') {
                baseValue += Math.random() * 2 + (i * 0.1);
            } else if (trendData.trend_direction === 'decreasing') {
                baseValue -= Math.random() * 2 - (i * 0.1);
            } else {
                baseValue += (Math.random() - 0.5) * 2;
            }
            values.push(Math.max(0, Math.round(baseValue)));
        }

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: 'Daily Distributions',
                    data: values,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Distribution Trends (Last 30 Days)'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Distributions'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    }
                }
            }
        });
    }

    /**
     * Create warehouse utilization chart
     */
    createWarehouseChart(canvasId, warehouses, items) {
        const ctx = document.getElementById(canvasId);
        if (!ctx || !warehouses || !items) return;

        const warehouseData = warehouses.map(warehouse => {
            const warehouseItems = items.filter(item => item.WarehouseID === warehouse.WarehouseID);
            const usedCapacity = warehouseItems.reduce((sum, item) => sum + item.Quantity, 0);
            const utilization = warehouse.Capacity > 0 ? (usedCapacity / warehouse.Capacity) * 100 : 0;
            
            return {
                name: warehouse.Location,
                utilization: Math.round(utilization),
                used: usedCapacity,
                total: warehouse.Capacity
            };
        });

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: warehouseData.map(w => w.name),
                datasets: [{
                    label: 'Utilization %',
                    data: warehouseData.map(w => w.utilization),
                    backgroundColor: warehouseData.map(w => 
                        w.utilization > 90 ? 'rgba(220, 53, 69, 0.6)' :
                        w.utilization > 70 ? 'rgba(255, 193, 7, 0.6)' :
                        'rgba(40, 167, 69, 0.6)'
                    ),
                    borderColor: warehouseData.map(w => 
                        w.utilization > 90 ? 'rgba(220, 53, 69, 1)' :
                        w.utilization > 70 ? 'rgba(255, 193, 7, 1)' :
                        'rgba(40, 167, 69, 1)'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Warehouse Utilization'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Utilization %'
                        }
                    }
                }
            }
        });
    }

    /**
     * Initialize all charts for the prediction page
     */
    initializePredictionCharts(mlData, trendAnalysis, warehouses, items) {
        // Demand forecasting chart
        if (mlData && mlData.demand_predictions) {
            this.createDemandChart('demandChart', mlData.demand_predictions);
        }

        // Risk assessment gauge
        if (mlData && mlData.risk_assessment) {
            this.createRiskGauge('riskGauge', 
                mlData.risk_assessment.risk_score, 
                mlData.risk_assessment.risk_level
            );
        }

        // Trend analysis chart
        if (trendAnalysis) {
            this.createTrendChart('trendChart', trendAnalysis);
        }

        // Warehouse utilization chart
        if (warehouses && items) {
            this.createWarehouseChart('warehouseChart', warehouses, items);
        }
    }
}

// Global instance
const mlCharts = new MLCharts();

// Auto-initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on the prediction page
    if (document.getElementById('demandChart')) {
        // Get data from global variables (set by Flask template)
        if (typeof window.mlData !== 'undefined') {
            mlCharts.initializePredictionCharts(
                window.mlData,
                window.trendAnalysis,
                window.warehouses,
                window.items
            );
        }
    }
});
