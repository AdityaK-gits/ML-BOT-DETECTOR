from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

def calculate_typing_speed(keystroke_timestamps: List[float]) -> float:
    """Calculate typing speed in characters per minute."""
    if len(keystroke_timestamps) < 2:
        return 0.0
    
    time_diff = keystroke_timestamps[-1] - keystroke_timestamps[0]
    if time_diff == 0:
        return 0.0
    
    cpm = (len(keystroke_timestamps) / time_diff) * 60
    return cpm

def analyze_mouse_movements(movements: List[Dict[str, float]]) -> Dict[str, float]:
    """Analyze mouse movement patterns."""
    if not movements:
        return {
            'avg_speed': 0.0,
            'straightness': 0.0,
            'acceleration': 0.0
        }
    
    speeds = []
    accelerations = []
    angles = []
    
    for i in range(1, len(movements)):
        prev = movements[i-1]
        curr = movements[i]
        
        # Calculate distance
        dx = curr['x'] - prev['x']
        dy = curr['y'] - prev['y']
        distance = (dx**2 + dy**2) ** 0.5
        
        # Calculate time difference in seconds
        time_diff = (curr['timestamp'] - prev['timestamp']) / 1000.0  # assuming ms timestamps
        
        if time_diff > 0:
            speed = distance / time_diff
            speeds.append(speed)
            
            # Calculate angle
            if i > 1:
                prev_dx = prev['x'] - movements[i-2]['x']
                prev_dy = prev['y'] - movements[i-2]['y']
                if (dx, dy) != (0, 0) and (prev_dx, prev_dy) != (0, 0):
                    dot = dx * prev_dx + dy * prev_dy
                    det = dx * prev_dy - dy * prev_dx
                    angle = np.arctan2(det, dot)
                    angles.append(abs(angle))
    
    # Calculate statistics
    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    straightness = 1.0 - (np.mean(angles) / np.pi) if angles else 0.0  # 1 = perfectly straight
    
    return {
        'avg_speed': avg_speed,
        'straightness': straightness,
        'acceleration': np.mean(accelerations) if accelerations else 0.0
    }

def detect_automation_patterns(activity: Dict) -> Dict[str, float]:
    """Detect potential automation patterns in user activity."""
    features = {}
    
    # Check for perfect timing patterns (e.g., exactly every X seconds)
    if 'click_timestamps' in activity and len(activity['click_timestamps']) > 2:
        timestamps = sorted(activity['click_timestamps'])
        intervals = np.diff(timestamps)
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        features['click_regularity'] = float(cv)
    
    # Check for mouse movement patterns
    if 'mouse_movements' in activity and len(activity['mouse_movements']) > 1:
        movement_analysis = analyze_mouse_movements(activity['mouse_movements'])
        features.update({
            'mouse_avg_speed': movement_analysis['avg_speed'],
            'mouse_straightness': movement_analysis['straightness']
        })
    
    # Check for typing patterns
    if 'keystroke_timestamps' in activity and len(activity['keystroke_timestamps']) > 1:
        typing_speed = calculate_typing_speed(activity['keystroke_timestamps'])
        features['typing_speed_cpm'] = typing_speed
    
    return features
