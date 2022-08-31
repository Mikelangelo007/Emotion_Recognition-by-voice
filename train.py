from deep_emotion_recognition import DeepEmotionRecognizer

deeprec = DeepEmotionRecognizer(emotions=['angry', 'sad', 'calm', 'happy'], n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)
deeprec.train()
def information():
	print(deeprec.test_score())
	print("If the answer is angry then training is successfull")
	prediction = deeprec.predict('data/validation/Actor_10/03-02-05-02-02-02-10_angry.wav')
	print(f"Prediction: {prediction}")



