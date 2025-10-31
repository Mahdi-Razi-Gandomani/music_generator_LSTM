import os
import numpy as np
from music21 import converter, note, chord, stream
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load midi files
filepath = "input/"
midi_files = []
for file in os.listdir(filepath):
    if file.endswith(".mid"):
        midi = converter.parse(filepath + file)
        midi_files.append(midi)

# Extract notes and chords
def extract_notes(midi):
    elements_to_parse = midi.flatten().notes
    notes = []
    for element in elements_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))  # Add note pitch
        elif isinstance(element, chord.Chord):
            notes.append(".".join(str(n) for n in element.normalOrder))  # Add chord notes
    return notes

notes = []
for midi in midi_files:
    notes.extend(extract_notes(midi))

# Create a sorted list of unique notes and mapping dictionaries
symb = sorted(set(notes))
symbol_to_index = {symbol: index for index, symbol in enumerate(symb)}
index_to_symbol = {index: symbol for index, symbol in enumerate(symb)}

# Prepare input sequences and targets for the model
sequence_length = 50
features = []
targets = []
for i in range(0, len(notes) - sequence_length, 1):
    feature = notes[i:i + sequence_length]
    target = notes[i + sequence_length]
    features.append([symbol_to_index[j] for j in feature])
    targets.append(symbol_to_index[target])

X = np.reshape(features, (len(features), sequence_length, 1)) / float(len(symb))
y = to_categorical(targets, num_classes=len(symb))

# Define the model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, batch_size=128, epochs=200)

# Convert a list of notes into a music21 stream
def melody_from_notes(s):
    melody = []
    offset = 0
    for i in s:
        if "." in i or i.isdigit():  # Handle chords
            chord_notes = i.split(".")
            notes = []
            for j in chord_notes:
                inst_note = int(j)
                note_s = note.Note(inst_note)
                notes.append(note_s)
            chord_s = chord.Chord(notes)
            chord_s.offset = offset
            melody.append(chord_s)
        else:  # Handle single notes
            note_s = note.Note(i)
            note_s.offset = offset
            melody.append(note_s)
        offset += 1  # Increment offset
    return stream.Stream(melody)

# Generate music using the trained model
def generator(note_length):
    patt = X[np.random.randint(0, len(X) - 1)]
    notes = []
    for _ in range(note_length):
        patt = patt.reshape(1, sequence_length, 1)
        prediction = model.predict(patt, verbose=0)
        index = np.argmax(prediction)
        notes.append(index_to_symbol[index])
        patt = np.append(patt[0], index)
        patt = patt[1:].reshape(sequence_length, 1)
    return melody_from_notes(notes)

# Generate and save a midi file
note_length = 150
generated_midi = generator(note_length)
generated_midi.write('midi', 'generated_midi.mid')
