from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from pytube import YouTube
import os
import librosa
import soundfile as sf
import random
from tqdm import tqdm
import numpy as np


class AudioPreprocessor:
    def __init__(self, url):
        self.url = url
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.binary_location = "/usr/bin/google-chrome"
        self.driver = webdriver.Chrome(executable_path="./chromedriver", options=self.chrome_options)

    def get_audio_urls(self):
        self.driver.get(self.url)
        youtube_urls = []
        video_elements = self.driver.find_elements(By.CLASS_NAME, "rll-youtube-player")
        for video_element in video_elements:
            uuid = video_element.get_attribute('data-id')
            youtube_url = f"https://www.youtube.com/watch?v={uuid}"
            youtube_urls.append(youtube_url)
        self.driver.quit()
        return youtube_urls


    def download_audio(self, youtube_url):
        try:
            yt = YouTube(youtube_url)
            file_name = yt.title.split(" ")
            file_name = "_".join(file_name[:3])
            os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Audio'), exist_ok=True)
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Audio', f"{file_name}.wav")
            yt.streams.filter(only_audio=True).first().download(filename = output_path)
            print(f"Downloaded {youtube_url}")
            return output_path
        except:
            print(f"Failed to download {youtube_url}")
            pass

    def download_all_audio(self):
        youtube_urls = self.get_audio_urls()
        for youtube_url in youtube_urls:
            downloaded_file = self.download_audio(youtube_url)
            if downloaded_file:
                self.split_audio(downloaded_file)
                os.remove(downloaded_file)

    def split_audio(self, audio_file_path):
        y, sr = librosa.load(audio_file_path)

        # Define the duration of each chunk in seconds
        chunk_duration = 15  # 10 seconds

        # Calculate the total number of chunks
        total_chunks = len(y) // (chunk_duration * sr) + 1
        if total_chunks < 40:
            return None
        # Create a directory to save the chunks
        output_dir = audio_file_path.split('/')[-1].split('.')[:-1]
        print(output_dir)
        output_dir = '_'.join(output_dir)
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Audio', output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Loop through each chunk and save it as a separate audio file
        for i in range(total_chunks):
            start_sample = i * chunk_duration * sr
            end_sample = min((i + 1) * chunk_duration * sr, len(y))
            chunk = y[start_sample:end_sample]
            chunk_file = os.path.join(output_dir, f'chunk_{i+1}.wav')
            # librosa.output.write_wav(chunk_file, chunk, sr)
            sf.write(chunk_file, chunk, sr)

            print(f'Saved chunk {i+1} to {chunk_file}')

        print('Splitting audio into chunks complete.')

    def split_dataset(self):
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset')
        os.makedirs(output_path, exist_ok=True)
        for speaker in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Audio')):
            speaker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Audio', speaker)
            if len(os.listdir(speaker_path)) < 40:
                continue
            audio_files = [os.path.join(speaker_path,item) for item in os.listdir(speaker_path)]
            choosed_files = random.sample(audio_files, 20)
            train_path = os.path.join(output_path, 'train', speaker)
            test_path = os.path.join(output_path, 'test', speaker)
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)
            for file in audio_files:
                if file in choosed_files:
                    os.rename(file, os.path.join(test_path, os.path.basename(file)))
                else:
                    os.rename(file, os.path.join(train_path, os.path.basename(file)))

    def audio_feature(self, file_name):
        #load the file (audio)
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        #we extract mfcc
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        #in order to find out scaled feature we do mean of transpose of value
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        return mfccs_scaled_features
        
    def extract_features(self):
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset')
        train_features = []
        test_features = []
        for splt in os.listdir(dataset_path):
            splt_dir = os.path.join(dataset_path, splt)
            for speaker in os.listdir(splt_dir):
                speaker_path = os.path.join(splt_dir, speaker)
                for file in tqdm(os.listdir(speaker_path)):
                    input_file = os.path.join(speaker_path, file)
                    feature = self.audio_feature(input_file)
                    if splt == 'train':
                        train_features.append([feature, speaker])
                    else:
                        test_features.append([feature, speaker])
        return train_features, test_features
        

    def process_audio(self):
        self.download_all_audio()
        self.split_dataset()
        train_features, test_features = self.extract_features()
        return train_features, test_features
        
        
