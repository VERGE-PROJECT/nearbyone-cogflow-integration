# Informer Time-Series Forecasting with CogFlow Integration

A comprehensive implementation of the Informer transformer model for time-series forecasting, integrated with CogFlow for experiment tracking, model management, and automated deployment on Kubernetes with KServe.

## Overview

This project implements the Informer model - a state-of-the-art transformer architecture specifically designed for efficient long-sequence time-series forecasting. The integration with CogFlow provides:

- **Automated MLOps Pipeline**: End-to-end workflow from data preprocessing to model serving
- **Experiment Tracking**: Comprehensive logging of parameters, metrics, and artifacts using MLflow
- **Model Serving**: Automated deployment using KServe on Kubernetes
- **Scalable Architecture**: Docker-based deployment with configurable resources

### Key Features

- **Efficient Long-Sequence Forecasting**: Leverages sparse attention mechanisms to handle long sequences with reduced computational complexity
- **Production-Ready Pipeline**: Complete MLOps workflow with preprocessing, training, and serving components
- **Cloud-Native Deployment**: Kubernetes-native serving with KServe for scalable inference
- **Comprehensive Logging**: Detailed experiment tracking with parameters, metrics, and model artifacts

## Architecture

The project follows a modular pipeline architecture:

```
Data Input → Preprocessing → Model Training → Model Serving
     ↓             ↓              ↓             ↓
   CSV File    Parquet +      Trained      KServe
              Config JSON     Model +     Inference
                             Artifacts    Service
```

### Components

1. **Preprocessing Component**: Converts CSV data to Parquet format and prepares training configurations
2. **Training Component**: Trains the Informer model with experiment tracking and artifact logging
3. **Serving Component**: Deploys the trained model as a KServe inference service

## Quick Start

### Prerequisites

- Python 3.10+
- Docker
- Kubernetes cluster with KServe installed
- CogFlow framework access

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nearbyone-cogflow-integration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Build the Docker image:
```bash
docker build -t burntt/nby-cogflow-informer:latest .
```

### Running the Pipeline

The main pipeline can be executed using the CogFlow framework:

```python
import cogflow as cf

# Initialize CogFlow client
client = cf.client()

# Run the pipeline
client.create_run_from_pipeline_func(
    informer_pipeline,
    arguments={
        "file": "/data/processed_data.csv",
        "isvc": "informer-serving-inference"
    }
)
```

### Alternative Execution

For standalone execution without the full pipeline:

```bash
python run_cf_server.py
```

## Configuration

### Model Parameters

The model can be configured through the preprocessing stage. Key parameters include:

- `seq_len`: Input sequence length (default: 12)
- `pred_len`: Prediction sequence length (default: 6)
- `d_model`: Model dimension (default: 32)
- `n_heads`: Number of attention heads (default: 4)
- `e_layers`: Number of encoder layers (default: 1)
- `d_layers`: Number of decoder layers (default: 1)
- `learning_rate`: Learning rate (default: 0.00001)
- `train_epochs`: Number of training epochs (default: 1)

### Data Format

Input data should be in CSV format with:
- Time series data with timestamp column
- Target variable for forecasting
- Optional feature columns

Example data structure:
```csv
timestamp,cpu_utilization,memory_usage,network_io
2023-01-01 00:00:00,0.75,0.65,1024
2023-01-01 00:01:00,0.73,0.67,1156
...
```

## Model Architecture

The Informer model includes several key innovations:

- **ProbSparse Self-Attention**: Reduces attention complexity from O(L²) to O(L log L)
- **Self-Attention Distilling**: Highlights dominating attention by halving cascading layer input
- **Generative Style Decoder**: Avoids cumulative error accumulation during inference

### Directory Structure

```
├── exp/                    # Experiment classes
│   ├── exp_informer.py    # Main Informer experiment class
│   └── exp_basic.py       # Base experiment class
├── models/                 # Model architecture
│   ├── model.py           # Main Informer model
│   ├── attn.py           # Attention mechanisms
│   ├── encoder.py        # Encoder implementation
│   ├── decoder.py        # Decoder implementation
│   └── embed.py          # Embedding layers
├── data/                   # Data handling
│   ├── data_loader.py    # Data loading utilities
│   └── processed_data.csv # Sample dataset
├── utils/                  # Utility functions
├── run_cf_server.py       # Main pipeline script
└── Dockerfile            # Container configuration
```

## Deployment

### Kubernetes Deployment

The model is automatically deployed to Kubernetes using KServe:

1. **Model Training**: The pipeline trains the model and saves artifacts to CogFlow
2. **Service Creation**: A KServe InferenceService is created automatically
3. **Endpoint Exposure**: The model becomes available for inference requests

### Inference

Once deployed, the model can be accessed via REST API:

```bash
curl -X POST "http://<inference-service-url>/v1/models/informer-serving-inference:predict" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "data": [[0.75, 0.65, 1024], [0.73, 0.67, 1156], ...]
      }
    ]
  }'
```

## Monitoring and Logging

### Experiment Tracking

All experiments are tracked using CogFlow/MLflow:

- **Parameters**: Model hyperparameters and configuration
- **Metrics**: Training and validation metrics (MAE, MSE, RMSE, R²)
- **Artifacts**: Model files, configuration files, and training logs

### Model Metrics

The pipeline logs the following metrics:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the research work on time-series forecasting in edge computing environments.

## References

- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)
- [CogFlow Framework Documentation](https://cogflow.ai)
- [KServe Documentation](https://kserve.github.io/website/)

## Support

For questions and issues, please refer to the project documentation or create an issue in the repository.
