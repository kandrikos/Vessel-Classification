# ** Under development **

## Underwater Acoustic Vessel Classification QA Pipeline

This README outlines the quality assurance and testing pipeline implemented for the Underwater Acoustic Vessel Classification project. The QA pipeline provides comprehensive testing, continuous integration/deployment, and model performance tracking to ensure consistent, high-quality model performance.

### Overview

This QA pipeline extends our underwater acoustic vessel classification system with testing and deployment practices. The implementation showcases practices in ML model testing, performance validation, and deployment automation.

**Key Features:**
- Comprehensive multi-level testing framework
- Automated CI/CD pipeline using GitHub Actions
- Model performance tracking and historical comparison
- Quality gates to prevent performance regression
- Automated deployment to multiple environments
- Notification system for test results and deployment status

### Directory Structure

The QA pipeline components are organized as follows:

```
ML-VTUAD/
├── .github/
│   └── workflows/                 # GitHub Actions configurations
│       ├── ci.yml                 # Continuous Integration workflow
│       ├── cd.yml                 # Continuous Deployment workflow
│       └── quality.yml            # Quality gates workflow
├── tests/
│   ├── unit/                      # Unit tests for individual components
│   │   ├── test_classifier.py     # Tests for vessel classifier
│   │   └── test_data_generator.py # Tests for data generator
│   ├── integration/               # Integration tests for pipelines
│   │   └── test_pipeline_integration.py # Tests for complete pipeline
│   ├── model/                     # Model validation tests
│   │   ├── test_model_validation.py    # Tests for model properties
│   │   └── test_model_performance.py   # Tests for model metrics
│   └── e2e/                       # Reserved for end-to-end tests
├── utils/
│   ├── metrics_tracker.py         # Track model metrics over time
│   ├── notification.py            # Alert and notification system
│   ├── prepare_deployment.py      # Model optimization
│   └── deploy_model.py            # Model deployment utility
└── [existing project files]       # Original classification codebase
```

### Testing Framework

The testing framework comprises four levels of tests to ensure comprehensive validation:

#### 1. Unit Tests

These tests verify that individual components work correctly in isolation:

- **Test Classifier**: Validates model creation, compilation, and output shapes
- **Test Data Generator**: Ensures correct data loading and preprocessing

Run with:
```bash
pytest tests/unit/
```

#### 2. Integration Tests

These tests verify that components work correctly together:

- **Test Pipeline Integration**: Validates the entire model training and evaluation pipeline

Run with:
```bash
pytest tests/integration/
```

#### 3. Model Validation Tests

These tests verify model quality and performance characteristics:

- **Model Properties**: Tests size, complexity, inference speed, and numerical stability
- **Performance Metrics**: Tests accuracy, precision, recall, and F1 score against thresholds

Run with:
```bash
pytest tests/model/
```

#### 4. End-to-End Tests (Future Implementation)

The `tests/e2e/` directory is reserved for future implementation of end-to-end tests that will simulate complete user workflows from data preprocessing through prediction and evaluation.

### CI/CD Pipeline

Our CI/CD pipeline utilizes GitHub Actions for automation:

#### Continuous Integration

The CI workflow (`ci.yml`) is triggered on push to main/develop branches and on pull requests:

1. **Code Quality**: Runs linting and style checks
2. **Unit Tests**: Executes tests and generates coverage reports
3. **Integration Tests**: Validates pipeline functionality

#### Quality Gates

The quality gates workflow (`quality.yml`) is triggered when model code is modified:

1. **Model Validation**: Runs comprehensive model validation tests
2. **Performance Tracking**: Updates metrics history and checks for regression

#### Continuous Deployment

The CD workflow (`cd.yml`) is triggered on push to main branch and manual dispatch:

1. **Model Training**: Runs full training with current code
2. **Model Deployment**: Prepares and deploys model to target environment

### Model Performance Tracking

The metrics tracking system maintains a historical record of model performance:

- Records key metrics for each model version
- Generates trend visualizations to track progress
- Provides alerts when performance degrades
- Identifies historically best versions for each metric

### Deployment Process

Our deployment process is designed for flexibility and reliability:

#### Model Preparation

The preparation utility (`prepare_deployment.py`):
- Converts the model to optimized formats
- Applies quantization when appropriate
- Generates metadata and signature files

#### Deployment Management

The deployment utility (`deploy_model.py`):
- Supports multiple environments (dev/staging/production)
- Provides local and cloud deployment options
- Maintains a versioned model registry
- Creates symbolic links to the latest version

## Usage Guide

### Running the Complete Test Suite

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=xml
```

### Training with Quality Verification

```bash
# Train with default configuration
python main.py

# Train with custom configuration
python main.py --config config/vessel_classification.yaml
```

### Performance Analysis

```bash
# Update metrics history
python utils/metrics_tracker.py --commit <commit_id> --save-metrics

# Generate trend visualizations
python utils/metrics_tracker.py --plot-trends --output-dir reports/performance
```

### Deployment

```bash
# Prepare model for deployment
python utils/prepare_deployment.py --input vessel_classification_results/best_model.keras --output deployment/model

# Deploy to production
python utils/deploy_model.py --model deployment/model --environment production

# Send notification
python utils/notification.py --event model_deployed --version <version>
```