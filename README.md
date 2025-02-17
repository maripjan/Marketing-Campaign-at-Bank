
### README.md


# Marketing Campaign at Bank

This project aims to predict the success of marketing campaigns conducted by a bank. The goal is to determine whether a customer will subscribe to a term deposit based on various features such as age, job, marital status, education, and more.

## Project Structure

- `api.py`: Flask application to serve predictions.
- `custom_pipeline.py`: Custom preprocessing pipeline for data transformation.
- `ml_pipeline.pkl`: Trained machine learning model and preprocessing pipeline.
- `requirements.txt`: List of required libraries.
- `README.md`: Project documentation.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip, uv or poetry (Python package managers)

### Setting Up the Environment

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/marketing-campaign-at-bank.git
   cd marketing-campaign-at-bank
   ```

2. **Create a virtual environment:**

   ```sh
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - On Windows:

     ```sh
     venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```sh
     source venv/bin/activate
     ```

4. **Install the required libraries:**

   ```sh
   pip install -r requirements.txt
   ```

### Running the Flask Application

1. **Start the Flask app:**

   ```sh
   python api.py
   ```

2. **Make predictions:**

   Send a POST request to the `/predict` endpoint with the input data in JSON format. For example:

   ```json
   {
     "X": [
       {
         "age": 30,
         "job": "admin.",
         "marital": "married",
         "education": "university.degree",
         "default": "no",
         "housing": "yes",
         "loan": "no",
         "contact": "cellular",
         "month": "may",
         "day_of_week": "mon",
         "duration": 300,
         "campaign": 1,
         "pdays": 999,
         "previous": 0,
         "poutcome": "nonexistent",
         "emp.var.rate": 1.1,
         "cons.price.idx": 93.994,
         "cons.conf.idx": -36.4,
         "euribor3m": 4.857,
         "nr.employed": 5191
       }
     ]
   }
   ```

   You can use tools like Postman or cURL to send the request.

### Deploying to Azure

1. **Create a `Procfile`:**

   ```sh
   web: python api.py
   ```

2. **Deploy to Azure:**

   ```sh
   az webapp up --name <your-app-name> --resource-group <your-resource-group> --runtime "PYTHON:3.8"
   ```

### Deploying to Google Cloud Platform (GCP)

1. **Create an `app.yaml` file:**

   ```yaml
   runtime: python38

   handlers:
   - url: /.*
     script: auto
   ```

2. **Deploy to GCP:**

   ```sh
   gcloud app deploy
   ```

### Monitoring

For monitoring, you can integrate Prometheus and Grafana or use cloud-specific monitoring tools like Azure Monitor or Google Cloud Monitoring.

