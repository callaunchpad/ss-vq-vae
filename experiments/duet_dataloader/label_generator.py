#genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
num_files = 100
num_segs = 6
with open(f'run-test/full2/labels', 'w') as f:
    for i in range(10):
        for _ in range(num_files*num_segs):
            print(i, file=f)