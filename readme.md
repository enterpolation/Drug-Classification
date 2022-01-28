# Drug Classification

## Research

`Data`: [Kaggle](https://www.kaggle.com/prathamtripathi/drug-classification).

### EDA

The data came from a pharmaceutical company that collected information about patients and
the types of medications that ended up being used as treatment. Our task is to use this
data to determine what type of medication would be appropriate for a person with a
particular set of symptoms and characteristics.

### Model Selection

Three machine learning methods were used for this task: `KNN`, `Logistic`
Regression and `Decision Tree`. The macro-averaged `F1-measure` was used for estimate
quality, hyperparameters were selected on cross-validation. After conducting a basic EDA,
hypothesis and prediction, the following results were obtained:

| Score      | KNN  | Logistic Regression | Decision Tree |
|------------|------|---------------------|---------------|
| `CV`       | 0.80 | 0.95                | 0.93          |
| `Macro F1` | 0.91 | 0.94                | 0.97          |

The best quality was obtained using logistic regression and the decision tree. They showed
approximately the same results on the cross-validation and on the test sample. For
predictions on this data, I personally would use the decision tree, as it gave a higher
performance on the macro-averaging of the F1-measure. Also, the tree almost perfectly
selected the predicates, which confirmed the earlier hypothesis, indicating good
interpretability.

## Deployment

This model is designed as a server application. The server runs as a `docker` container.

### How to run
- Install and run Docker

- Build Docker image from inside the server directory:
`docker build -t application .`

- Run Docker container using:
`docker run -p 5000:5000 -d application`

- Go to `localhost:5000` to see the input form.

### Input form
Input example:

| Age (int) | Na_to_K (float) | BP_HIGH (binary) | BP_LOW (binary) | BP_NORMAL (binary) | Cholesterol_HIGH (binary) | Cholesterol_NORMAL (binary) |
|-----------|:----------------|------------------|-----------------|--------------------|---------------------------|-----------------------------|
| 23        | 25.355          | 1                | 0               | 0                  | 1                         | 0                           |

Result:

| Prediction | Probability |
|------------|-------------|
| drugY      | 1.0         |
