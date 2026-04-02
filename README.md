# 🍦 Ice Cream Sales Prediction Web App

A Flask-based web application that predicts ice cream sales using Linear Regression machine learning model trained on historical sales data.

## 📋 Overview

This project demonstrates a complete machine learning pipeline with a web interface:

- **Machine Learning**: Linear Regression model trained on temperature and rainfall data
- **Backend**: Flask REST API for predictions
- **Frontend**: Responsive HTML/CSS/Bootstrap interface
- **Model Storage**: Pickled scikit-learn model for easy deployment

## 🎯 Features

✅ Real-time ice cream sales predictions  
✅ Input validation with helpful error messages  
✅ Responsive mobile-friendly design  
✅ Fast model inference (<100ms)  
✅ Clean, intuitive user interface  
✅ Model performance metrics (R² Score, MSE)

## 📊 Dataset

**File**: `ice-cream.csv`

**Columns**:

- `Date`: Transaction date
- `DayOfWeek`: Day of the week
- `Month`: Month of the year
- `Temperature`: Temperature in °C
- `Rainfall`: Rainfall in inches
- `IceCreamsSold`: Number of ice creams sold (target variable)

## 🏗️ Project Structure

```
icecream-app/
├── static/
│   └── style.css              # Custom CSS styling
├── templates/
│   └── index.html             # Main HTML template
├── model/
│   └── model.pkl              # Trained model (generated)
├── ice-cream.csv              # Dataset
├── app.py                      # Flask application
├── train_model.py             # Model training script
└── README.md                  # This file
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
pip install flask pandas numpy scikit-learn
```

### Step 2: Train the Model

Navigate to the project directory and run:

```bash
python train_model.py
```

**Expected Output**:

```
Dataset Overview:
Total records: xxx

Training R² Score: 0.xxxx
Testing R² Score: 0.xxxx
Training MSE: xxx.xxxx
Testing MSE: xxx.xxxx

✓ Model saved successfully at: model/model.pkl
```

### Step 3: Start the Flask Server

```bash
python app.py
```

**Output**:

```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### Step 4: Access the Application

Open your web browser and visit:

```
http://localhost:5000
```

## 📱 Usage

1. **Enter Temperature**: Input the current temperature in Celsius (40-85°C)
2. **Enter Rainfall**: Input the rainfall amount in inches (0-2")
3. **Click "Predict Sales"**: The model will predict ice cream sales
4. **View Results**: Predicted sales quantity will be displayed

## 📈 Model Details

### Algorithm: Linear Regression

Linear Regression finds the best-fit line through the data points using the equation:

```
IceCreamsSold = β₀ + β₁×Temperature + β₂×Rainfall
```

### Model Performance

The script `train_model.py` evaluates the model using:

- **R² Score**: Measures how well the model explains the variance (0-1, higher is better)
- **Mean Squared Error (MSE)**: Average squared difference between actual and predicted values
- **RMSE**: Root Mean Squared Error (in units of ice creams sold)

### Train/Test Split

- Training set: 80% of data
- Testing set: 20% of data
- Random state: 42 (for reproducibility)

## 🔌 API Endpoints

### GET `/`

Returns the main prediction interface (HTML page)

### POST `/predict`

Submits prediction request

**Request Body**:

```json
{
  "temperature": 65.5,
  "rainfall": 0.5
}
```

**Response**:

```json
{
  "temperature": 65.5,
  "rainfall": 0.5,
  "predicted_sales": 87.25,
  "success": true
}
```

### GET `/health`

Health check endpoint

**Response**:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🧪 Testing

### Manual Testing

Test with sample inputs:

- **Cold day**: Temp=50°C, Rain=0.5" → Low sales expected
- **Hot day**: Temp=75°C, Rain=0.1" → High sales expected
- **Edge cases**: Temp=40°C, Temp=85°C

### Input Validation

The application validates:

- ✓ Numeric input only
- ✓ Temperature range (40-85°C)
- ✓ Rainfall range (0-2 inches)
- ✓ Required fields

## 🚀 Deployment

To run in production:

1. Set Flask environment:

```bash
set FLASK_ENV=production  # Windows
export FLASK_ENV=production  # Mac/Linux
```

2. Use a production WSGI server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app.py
```

## 📝 File Descriptions

### `train_model.py`

- Loads ice-cream.csv
- Splits data (80-20 train-test)
- Trains Linear Regression model
- Evaluates with R² and MSE
- Saves model as pickle file

### `app.py`

- Flask application with 3 routes
- Loads trained model at startup
- Handles prediction requests
- Validates user inputs
- Returns JSON predictions

### `templates/index.html`

- Single-page responsive interface
- Bootstrap 5 styling
- JavaScript form handling
- Real-time prediction display
- Error message display

### `static/style.css`

- Custom gradient styling
- Responsive design (mobile-friendly)
- Animation effects
- Accessibility features
- Dark/light mode compatible

## 🐛 Troubleshooting

### "Model not found" Error

**Solution**: Run `python train_model.py` first to train and save the model

### "ModuleNotFoundError" for Flask, pandas, etc.

**Solution**: Install dependencies with `pip install flask pandas numpy scikit-learn`

### Application won't start on port 5000

**Solution**: Either:

- Close other applications using port 5000
- Modify port in `app.py`: `app.run(port=5001)`

### Predictions seem unrealistic

**Solution**:

- Check input ranges are correct
- Verify model was trained with `train_model.py`
- Check model performance metrics in training output

## 📚 Technologies Used

| Technology       | Purpose                        |
| ---------------- | ------------------------------ |
| **Python 3**     | Programming language           |
| **Flask**        | Web framework                  |
| **scikit-learn** | ML library (Linear Regression) |
| **pandas**       | Data processing                |
| **numpy**        | Numerical computations         |
| **pickle**       | Model serialization            |
| **Bootstrap 5**  | UI framework                   |
| **HTML/CSS/JS**  | Frontend                       |

## 📊 Expected Results

With the provided dataset:

- Training R² Score: ~0.85-0.95
- Testing R² Score: ~0.80-0.90
- RMSE: ~10-15 units

_Exact values depend on data distribution and train-test split_

## 🎓 Learning Outcomes

This project demonstrates:

1. Machine Learning workflow (train → evaluate → save)
2. REST API development with Flask
3. Frontend-backend integration
4. Model deployment and serving
5. Input validation and error handling
6. Responsive web design

## 🔐 Security Notes

- Input validation prevents invalid data
- Model handles edge cases gracefully
- No sensitive data exposure
- CSRF protection via Flask's built-in security

## 📄 License

This project is open-source and available for educational purposes.

## ✨ Future Enhancements

- Multi-feature prediction interface
- historical data visualization
- Model performance dashboard
- Database integration for storing predictions
- Authentication & user accounts
- Advanced ML models (Random Forest, XGBoost)
- Batch prediction capability

## 🤝 Contributing

Feel free to fork, modify, and improve this project!

## 📞 Support

For issues or questions, check the troubleshooting section above.

---

**Happy Predicting!** 🍦✨
