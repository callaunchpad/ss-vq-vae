genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
num_files = 100
with open(f'rnn_input', 'w') as f:
    for genre in genres:
        for i in range(num_files):
            for j in range(6):
                f.write(f'/datasets/duet/genres/{genre}.{i:05d}.{j}.wav\n')