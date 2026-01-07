# ML Bot Detector

A machine learning-based bot detection system that analyzes user behavior patterns to distinguish between human users and automated bots.

## Features

- Real-time bot detection using behavioral analysis
- Multiple detection methods:
  - Mouse movement analysis
  - Keystroke dynamics
  - Click patterns
  - Request timing
- RESTful API for easy integration
- Extensible model architecture

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ml-bot-detector
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the API

Start the FastAPI development server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Detect Bot
`POST /detect-bot`

Detects if a user is likely a bot based on their activity.

**Request Body Example:**
```json
{
    "user_id": "user123",
    "timestamp": "2023-01-01T12:00:00Z",
    "request_path": "/login",
    "request_duration": 0.42,
    "mouse_movements": [
        {"x": 100, "y": 200, "timestamp": 1672574400000},
        {"x": 101, "y": 202, "timestamp": 1672574400010}
    ],
    "click_pattern": [
        {"x": 100, "y": 200, "timestamp": 1672574400000, "button": "left"}
    ],
    "typing_speed": 240.5,
    "scroll_behavior": {"speed": 10.5, "direction": "down"}
}
```

**Response Example:**
```json
{
    "user_id": "user123",
    "is_bot": false,
    "bot_probability": 0.23,
    "features_used": {
        "request_duration": 0.42,
        "typing_speed": 240.5,
        "num_mouse_movements": 2,
        "num_clicks": 1
    }
}
```

### Health Check
`GET /health`

Check if the API is running.

**Response:**
```json
{
    "status": "healthy"
}
```

## Model Training

To train a new model:

1. Prepare your training data in CSV format with labeled examples of bot and human activities.
2. Use the `train_model.py` script (coming soon) to train a new XGBoost model.
3. Save the trained model as `bot_detection_model.pkl` in the project root.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
