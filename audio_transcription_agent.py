import whisper
import torch
from pydub import AudioSegment
from pydub.playback import play

# checkout the existing models using the following command
whisper.available_models()

# select a model
model_type = "large-v3"
model = whisper.load_model(model_type)
audio_file_path =  "./assets/data_pipeline_recording_snippet.mp3"
audio_result = model.transcribe(audio_file_path)

transcript_location = "./results/transcript_for_large.txt"

with open(transcript_location,'w') as f :
    f.write(audio_result["text"])


