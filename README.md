\# Heart Disease Prediction API — FastAPI Lab



A modified version of the MLOps FastAPI Lab, using the \*\*UCI Cleveland Heart Disease dataset\*\* and a \*\*Random Forest Classifier\*\* instead of the original Iris dataset and Decision Tree.



\## Changes from Original Lab

| Original | This Version |

|---|---|

| Iris dataset | UCI Cleveland Heart Disease dataset |

| Decision Tree Classifier | Random Forest Classifier (100 trees, max\_depth=6) |

| 4 features | 13 clinical features |

| 1 endpoint (`/predict`) | 4 endpoints: `/`, `/health`, `/model-info`, `/predict` |



\## Project Structure

```

fastapi\_lab\_heart/

├── model/

│   └── heart\_model.pkl

├── src/

│   ├── \_\_init\_\_.py

│   ├── data.py

│   ├── train.py

│   ├── predict.py

│   └── main.py

├── requirements.txt

└── README.md

```



\## Setup \& Run



\### 1. Install dependencies

```bash

pip install -r requirements.txt

```



\### 2. Train the model

```bash

cd src

python train.py

```



\### 3. Start the API

```bash

uvicorn main:app --reload

```



\### 4. Test the API

Open your browser at: \*\*http://127.0.0.1:8000/docs\*\*



\## Endpoints



| Method | Endpoint | Description |

|---|---|---|

| GET | `/` | Root welcome message |

| GET | `/health` | Health check |

| GET | `/model-info` | Model metadata |

| POST | `/predict` | Predict heart disease |



\## Sample Prediction Request



```json

{

&nbsp; "age": 63,

&nbsp; "sex": 1,

&nbsp; "cp": 3,

&nbsp; "trestbps": 145,

&nbsp; "chol": 233,

&nbsp; "fbs": 1,

&nbsp; "restecg": 0,

&nbsp; "thalach": 150,

&nbsp; "exang": 0,

&nbsp; "oldpeak": 2.3,

&nbsp; "slope": 0,

&nbsp; "ca": 0,

&nbsp; "thal": 1

}

```



\## Sample Response

```json

{

&nbsp; "prediction": 1,

&nbsp; "label": "Heart Disease Detected",

&nbsp; "confidence": 0.82

}

```



\## Dataset

\- \*\*Source:\*\* \[UCI Machine Learning Repository – Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)

\- \*\*Instances:\*\* 303 (after dropping missing values)

\- \*\*Features:\*\* 13 clinical attributes (age, sex, chest pain type, etc.)

\- \*\*Target:\*\* Binary (0 = no disease, 1 = disease present)



\## Reference

Based on: \[raminmohammadi/MLOps – FastAPI Lab](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API\_Labs/FastAPI\_Labs)

