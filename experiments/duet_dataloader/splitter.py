from pydub import AudioSegment
from os import path
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

for genre in genres:
    for i in range(100):
        AUDIO_FILE = path.join('/datasets/duet/genres/', genre, f'{genre}.{i:05d}.wav')
        sound = AudioSegment.from_file(AUDIO_FILE)
        segment_len = len(sound) // 6
        for seg in range(6):
            segment = sound[seg*segment_len:(seg+1)*segment_len]
            segment.export(f"data/{genre}.{i:05d}.{seg}.wav", format="wav")