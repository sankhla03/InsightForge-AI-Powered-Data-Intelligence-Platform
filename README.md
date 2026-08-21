# InsightForge - ML Pipeline Application

A comprehensive Django-based machine learning pipeline application for data preprocessing, visualization, model training, and prediction.

## Features

### 1. User Authentication
- User registration and login
- MongoDB-based authentication system
- Session management

### 2. Dataset Management
- Upload datasets (CSV files)
- View and manage uploaded datasets
- Support for multiple data formats

### 3. Data Preprocessing
- **Data Cleaning**: Handle missing values, data type conversions
- **Feature Selection**: Select relevant features for ML models
- **Outlier Detection**: Identify and handle outliers in data
- **Label Noise Handling**: Detect and manage noisy labels
- **Feature Scaling**: Normalize and standardize features

### 4. Data Visualization
- Interactive charts using Plotly
- Dashboard with various visualization types
- Confusion matrices, residual plots, actual vs predicted plots

### 5. Machine Learning Engine
- Model training with multiple algorithms
- Support for classification and regression
- Model prediction capabilities
- Model persistence (save/load trained models)

### 6. Reports
- Generate comprehensive ML pipeline reports
- Feature importance analysis
- Save and view historical reports

## Technology Stack

- **Backend**: Django 6.0
- **Database**: SQLite3 (default), MongoDB (authentication)
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Plotly
- **Machine Learning**: Scikit-learn, Pandas, NumPy

## Project Structure

```
insightForge/
├── accounts/           # User authentication
├── datasets/           # Dataset upload and management
├── preprocessing/     # Data preprocessing modules
├── visualization/     # Data visualization
├── ml_engine/         # Machine learning engine
├── reports/           # Report generation
├── insightForge/      # Django project settings
├── static/            # Static files (CSS, JS, images)
├── templates/        # HTML templates
└── manage.py          # Django management script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sankhla03/insightForge.git
cd insightForge
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install django pandas numpy scikit-learn plotly pymongo
```

4. Run migrations:
```bash
cd insightForge
python manage.py migrate
```

5. Start the development server:
```bash
python manage.py runserver
```

6. Open your browser and navigate to:
```
http://127.0.0.1:8000
```

## Usage

1. **Register/Login**: Create an account or login to access the platform
2. **Upload Data**: Navigate to Datasets and upload your CSV file
3. **Preprocess Data**: Use the preprocessing tools to clean and transform your data
4. **Visualize**: View your data through various visualizations
5. **Train Model**: Select features and train a machine learning model
6. **Predict**: Use the trained model for predictions
7. **Generate Reports**: Create comprehensive reports of your ML pipeline

## Modules

### Preprocessing
- `clean.html` - Data cleaning interface
- `features.html` - Feature selection
- `outliers.html` - Outlier detection
- `label_noise.html` - Label noise handling
- `scale_features.html` - Feature scaling

### Visualization
- Interactive dashboard with Plotly charts
- Confusion matrices
- Residual plots
- Actual vs Predicted plots

### ML Engine
- Model training interface
- Multiple algorithm support
- Model prediction
- Model persistence

### Reports
- Pipeline reports
- Feature importance analysis
- Saved reports management

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

 under the MIT License.

## AuthorThis project is licensed

- Ashok sankhla

## Acknowledgments

- Django Framework
- Plotly for visualizations
- Scikit-learn for machine learning

