import sys
from train import deeprec

path = sys.argv[1]
prediction = deeprec.predict(path)
print(f"Prediction: {prediction}")
print(deeprec.predict_proba(path))
