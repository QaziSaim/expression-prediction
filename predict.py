import numpy as np

emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
text_to_predict = input("Enter the text to predict: ")




prediction = model.predict(text_to_predict)


predicted_emotion_index = np.argmax(prediction)
predicted_emotion = emotion_map[predicted_emotion_index]

print(f"Predicted emotion: {predicted_emotion}")