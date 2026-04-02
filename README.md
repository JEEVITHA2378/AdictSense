# AdictSense

AdictSense (also known as DigiGuard) is a machine learning-powered web application designed to predict and monitor digital addiction risk levels based on usage features.

## Features
- **Risk Prediction:** Utilizes a custom ML model to predict user risk levels (Low, Medium, High) based on screen time, social media usage, gaming, and night usage.
- **Authentication:** Built-in integration with Supabase for user authentication and data management.
- **Flask Backend:** A lightweight and robust Python Flask backend API handling data and predictions.
- **Interactive UI:** A clean interface to submit usage data and receive confident risk evaluations.

## Getting Started

### Prerequisites
- Python 3.8+
- Local or remotely hosted Supabase Account (for Database & Auth)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JEEVITHA2378/AdictSense.git
   cd AdictSense
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables by creating a `.env` file with the following:
   ```env
   FLASK_SECRET_KEY=your_secret_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```

4. Train the Machine Learning Model (generates `model.pkl`):
   ```bash
   python train_model.py
   ```

5. Run the web application:
   ```bash
   python app.py
   ```

The application will be running at `http://127.0.0.1:5000`.
