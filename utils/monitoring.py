import pandas as pd
import numpy as np
from datetime import datetime

class PerformanceMonitoringDashboard:
    def __init__(self):
        self.performance_history = []
        self.drift_history = []
        self.feature_importance_data = {}
        
    def update_performance(self, metrics):
        """Update performance metrics"""
        timestamp = datetime.now()
        metrics['timestamp'] = timestamp
        self.performance_history.append(metrics)
        
        # Keep only last 1000 records
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def update_drift(self, drift_data):
        """Update drift detection data"""
        timestamp = datetime.now()
        drift_data['timestamp'] = timestamp
        self.drift_history.append(drift_data)
        
        if len(self.drift_history) > 1000:
            self.drift_history = self.drift_history[-1000:]
    
    def update_feature_importance(self, feature_data):
        """Update feature importance data"""
        self.feature_importance_data = feature_data
    
    def generate_alert_report(self):
        """Generate alert report for significant events"""
        alerts = []
        
        # Check for performance degradation
        if len(self.performance_history) >= 10:
            recent_performance = self.performance_history[-10:]
            recent_confidences = [p.get('confidence', 0) for p in recent_performance]
            historical_confidences = [p.get('confidence', 0) for p in self.performance_history[:10]]
            
            if recent_confidences and historical_confidences:
                avg_recent_confidence = np.mean(recent_confidences)
                avg_historical_confidence = np.mean(historical_confidences)
                
                if avg_recent_confidence < avg_historical_confidence * 0.9:  # 10% degradation
                    alerts.append({
                        'type': 'CONFIDENCE_DEGRADATION',
                        'severity': 'MEDIUM',
                        'message': f"Model confidence degraded by {((avg_historical_confidence - avg_recent_confidence) / avg_historical_confidence * 100):.1f}%",
                        'timestamp': datetime.now()
                    })
        
        # Check for recent drift
        if self.drift_history:
            recent_drifts = [d for d in self.drift_history 
                           if (datetime.now() - d['timestamp']).total_seconds() / 3600 < 24]
            if recent_drifts:
                max_drift = max(d.get('drift_score', 0) for d in recent_drifts)
                if max_drift > 0.15:  # High drift threshold
                    alerts.append({
                        'type': 'HIGH_DRIFT',
                        'severity': 'MEDIUM',
                        'message': f"High model drift detected: {max_drift:.3f}",
                        'timestamp': datetime.now()
                    })
        
        return alerts
    
    def get_performance_summary(self):
        """Get overall performance summary"""
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        df = pd.DataFrame(self.performance_history)
        summary = {
            'total_predictions': len(self.performance_history),
            'current_confidence': df['confidence'].iloc[-1] if 'confidence' in df.columns else 0,
            'avg_confidence_7d': df['confidence'].tail(100).mean() if 'confidence' in df.columns else 0,
            'drift_events_24h': len([d for d in self.drift_history 
                                   if (datetime.now() - d['timestamp']).total_seconds() / 3600 < 24]),
            'alerts': self.generate_alert_report()
        }
        
        return summary
    
    def get_recent_performance(self, hours=24):
        """Get performance data from recent hours"""
        cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
        recent_data = [p for p in self.performance_history 
                      if p['timestamp'] > cutoff_time]
        return recent_data