from AudioPreprocessor import AudioPreprocessor
from AudioClassifier import AudioClassifier
import pandas as pd
import numpy as np

BASE_URL = 'https://www.lifehack.org/794639/famous-speeches'
# Preprocess the audio files
preprocessor = AudioPreprocessor(url=BASE_URL)
train_features, test_features = preprocessor.process_audio()
train_df = pd.DataFrame(train_features, columns=['features', 'label'])
test_df = pd.DataFrame(test_features, columns=['features', 'label'])

# Train the model
classifier = AudioClassifier(epochs=40, batch_size=16)
train_label = classifier.labelMapper(train_df['label'])

val_label = classifier.labelMapper(test_df['label'])
train_label = classifier.label_encoder(train_label)
val_label = classifier.label_encoder(val_label)

train_features = np.array(train_df['features'].to_list())
test_features = np.array(test_df['features'].to_list())

x_train, x_test, y_train, y_test = classifier.split_dataset(train_features, train_label)
grid_result = classifier.grid_search(x_train, y_train)
classifier.save_logs(grid_result)
best_model = classifier.train_best_fit(grid_result, x_train, y_train)

print("TEST SET EVALUATION")
classifier.evaluate_model(best_model, x_test, y_test)

print("VALIDATION SET EVALUATION")
classifier.evaluate_model(best_model, test_features, val_label)
classifier.save_model(best_model)


