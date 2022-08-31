#from deep_emotion_recognition import DeepEmotionRecognizer
import sys
from train import deeprec
#deeprec = DeepEmotionRecognizer(emotions=['angry', 'sad', 'calm', 'happy'], n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)
# train the model
#deeprec.train()
path = sys.argv[1]
prediction = deeprec.predict(path)
print(f"Prediction: {prediction}")
print(deeprec.predict_proba(path))
