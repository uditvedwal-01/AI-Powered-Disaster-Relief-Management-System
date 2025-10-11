"""
ML Service Layer for Disaster Resource Management System
Handles ML operations and data processing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from ml_models import ml_predictor


class MLService:
    """Service class for ML operations"""
    
    def __init__(self):
        self.predictor = ml_predictor
    
    def get_prediction_data(self, disaster_id: int, warehouses, items, beneficiaries, distributions):
        """Get comprehensive prediction data for a disaster"""
        try:
            # Train all models if we have enough data
            demand_trained = False
            risk_trained = False
            trend_trained = False
            
            if len(distributions) >= 5:
                demand_trained = self.predictor.train_demand_model(warehouses, items, beneficiaries, distributions)
            
            if len(distributions) >= 3:
                disaster = self._get_disaster_info(disaster_id)
                risk_trained = self.predictor.train_risk_model(disaster, warehouses, items, beneficiaries, distributions)
            
            if len(distributions) >= 10:
                trend_trained = self.predictor.train_trend_model(distributions)
            
            # Get predictions
            demand_predictions = self.predictor.predict_demand(items, beneficiaries, distributions)
            
            # Get allocation recommendations
            allocation_recs = self.predictor.get_allocation_recommendations(
                warehouses, items, beneficiaries, distributions
            )
            
            # Get risk assessment (ML-based if trained)
            disaster = self._get_disaster_info(disaster_id)
            risk_assessment = self.predictor.get_risk_assessment(
                disaster, warehouses, items, beneficiaries, distributions
            )
            
            # Get trend analysis (ML-based if trained)
            trend_analysis = self.predictor.predict_trend_ml(distributions) if trend_trained else self.predictor._fallback_trend_analysis(distributions)
            
            return {
                'demand_predictions': demand_predictions,
                'allocation_recommendations': allocation_recs,
                'risk_assessment': risk_assessment,
                'trend_analysis': trend_analysis,
                'model_trained': demand_trained,
                'risk_trained': risk_trained,
                'trend_trained': trend_trained,
                'data_quality': self._assess_data_quality(items, distributions, beneficiaries)
            }
            
        except Exception as e:
            print(f"Error in ML prediction: {str(e)}")
            return self._get_fallback_data(items, distributions)
    
    def _get_disaster_info(self, disaster_id: int):
        """Get disaster information for risk assessment"""
        # This would typically query the database
        # For now, return a mock disaster object
        class MockDisaster:
            def __init__(self):
                self.DisasterID = disaster_id
                self.Name = f"Disaster {disaster_id}"
                self.Type = "Natural"
                self.Severity = "Medium"
        
        return MockDisaster()
    
    def _assess_data_quality(self, items, distributions, beneficiaries):
        """Assess the quality of available data for ML predictions"""
        quality_score = 0
        issues = []
        
        # Check data availability
        if len(items) == 0:
            issues.append("No items available")
            quality_score -= 30
        elif len(items) < 5:
            issues.append("Limited item variety")
            quality_score -= 10
        
        if len(distributions) == 0:
            issues.append("No distribution history")
            quality_score -= 40
        elif len(distributions) < 10:
            issues.append("Limited distribution data")
            quality_score -= 20
        
        if len(beneficiaries) == 0:
            issues.append("No beneficiary data")
            quality_score -= 20
        elif len(beneficiaries) < 5:
            issues.append("Limited beneficiary data")
            quality_score -= 10
        
        # Check data recency
        if distributions:
            recent_distributions = [d for d in distributions 
                                 if (datetime.now().date() - d.Date).days <= 30]
            if len(recent_distributions) < len(distributions) * 0.3:
                issues.append("Outdated distribution data")
                quality_score -= 15
        
        # Determine quality level
        if quality_score >= 80:
            quality_level = "Excellent"
        elif quality_score >= 60:
            quality_level = "Good"
        elif quality_score >= 40:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
        return {
            'quality_level': quality_level,
            'quality_score': max(0, quality_score),
            'issues': issues,
            'recommendations': self._get_data_quality_recommendations(issues)
        }
    
    def _get_data_quality_recommendations(self, issues):
        """Get recommendations to improve data quality"""
        recommendations = []
        
        if "No items available" in issues:
            recommendations.append("Add more relief items to the system")
        
        if "No distribution history" in issues:
            recommendations.append("Record distribution activities to enable predictions")
        
        if "Limited distribution data" in issues:
            recommendations.append("Record more distribution activities for better predictions")
        
        if "No beneficiary data" in issues:
            recommendations.append("Add beneficiary information")
        
        if "Outdated distribution data" in issues:
            recommendations.append("Record recent distribution activities")
        
        if not recommendations:
            recommendations.append("Data quality is good - continue regular data entry")
        
        return recommendations
    
    def _get_fallback_data(self, items, distributions):
        """Get fallback data when ML fails"""
        return {
            'demand_predictions': self._simple_demand_predictions(items),
            'allocation_recommendations': [
                {
                    'type': 'info',
                    'message': 'ML predictions unavailable - using basic heuristics',
                    'priority': 'low'
                }
            ],
            'risk_assessment': {
                'risk_level': 'Unknown',
                'risk_score': 0,
                'risk_factors': ['Insufficient data for assessment'],
                'recommendations': ['Collect more data to enable ML predictions']
            },
            'model_trained': False,
            'data_quality': {
                'quality_level': 'Poor',
                'quality_score': 0,
                'issues': ['ML service unavailable'],
                'recommendations': ['Check ML service configuration']
            }
        }
    
    def _simple_demand_predictions(self, items):
        """Simple heuristic-based predictions"""
        predictions = {}
        
        for item in items:
            # Simple prediction based on current stock
            predicted_demand = max(item.Quantity * 1.2, 10)
            
            predictions[item.ItemID] = {
                'item_name': item.Name,
                'current_stock': item.Quantity,
                'predicted_demand': int(predicted_demand),
                'daily_average': round(predicted_demand / 7, 1),
                'stock_status': 'Low' if item.Quantity < predicted_demand else 'Adequate',
                'recommendation': 'Monitor stock levels'
            }
        
        return predictions
    
    def get_trend_analysis(self, distributions, days=30):
        """Analyze distribution trends over time"""
        if not distributions:
            return {
                'trend': 'No data',
                'total_distributions': 0,
                'avg_daily': 0,
                'peak_day': None,
                'trend_direction': 'stable'
            }
        
        # Filter recent distributions
        cutoff_date = datetime.now().date() - timedelta(days=days)
        recent_distributions = [d for d in distributions if d.Date >= cutoff_date]
        
        if not recent_distributions:
            return {
                'trend': 'No recent data',
                'total_distributions': len(distributions),
                'avg_daily': 0,
                'peak_day': None,
                'trend_direction': 'stable'
            }
        
        # Group by date
        daily_counts = {}
        for dist in recent_distributions:
            date_key = dist.Date
            if date_key not in daily_counts:
                daily_counts[date_key] = 0
            daily_counts[date_key] += 1
        
        # Calculate trends
        dates = sorted(daily_counts.keys())
        counts = [daily_counts[date] for date in dates]
        
        if len(counts) >= 2:
            # Simple trend calculation
            first_half = counts[:len(counts)//2]
            second_half = counts[len(counts)//2:]
            
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            
            if avg_second > avg_first * 1.1:
                trend_direction = 'increasing'
            elif avg_second < avg_first * 0.9:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'stable'
        
        # Find peak day
        peak_day = max(daily_counts.items(), key=lambda x: x[1])[0] if daily_counts else None
        
        return {
            'trend': f'{len(recent_distributions)} distributions in last {days} days',
            'total_distributions': len(recent_distributions),
            'avg_daily': round(len(recent_distributions) / days, 1),
            'peak_day': peak_day,
            'trend_direction': trend_direction
        }
    
    def get_optimization_suggestions(self, warehouses, items, distributions):
        """Get optimization suggestions based on current data"""
        suggestions = []
        
        # Warehouse optimization
        for warehouse in warehouses:
            warehouse_items = [item for item in items if item.WarehouseID == warehouse.WarehouseID]
            total_stock = sum(item.Quantity for item in warehouse_items)
            
            if warehouse.Capacity > 0:
                utilization = (total_stock / warehouse.Capacity) * 100
                
                if utilization > 90:
                    suggestions.append({
                        'type': 'warehouse',
                        'priority': 'high',
                        'message': f"Warehouse {warehouse.Location} is {utilization:.1f}% full",
                        'action': 'Consider redistributing items or expanding capacity'
                    })
                elif utilization < 20:
                    suggestions.append({
                        'type': 'warehouse',
                        'priority': 'medium',
                        'message': f"Warehouse {warehouse.Location} is underutilized ({utilization:.1f}%)",
                        'action': 'Consider consolidating with other warehouses'
                    })
        
        # Item optimization
        item_activity = {}
        for dist in distributions:
            item_id = dist.ItemID
            if item_id not in item_activity:
                item_activity[item_id] = 0
            item_activity[item_id] += 1
        
        if item_activity:
            avg_activity = sum(item_activity.values()) / len(item_activity)
            
            for item in items:
                activity = item_activity.get(item.ItemID, 0)
                if activity > avg_activity * 2:
                    suggestions.append({
                        'type': 'item',
                        'priority': 'high',
                        'message': f"High demand for {item.Name}",
                        'action': 'Increase stock levels and consider multiple warehouses'
                    })
                elif activity == 0 and item.Quantity > 0:
                    suggestions.append({
                        'type': 'item',
                        'priority': 'low',
                        'message': f"No recent demand for {item.Name}",
                        'action': 'Consider redistributing to areas with higher demand'
                    })
        
        return suggestions


# Global ML service instance
ml_service = MLService()
