# Music Generation with LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to generate music. The model is trained on MIDI files from Bach's compositions and can generate new musical sequences.

---

## Requirements

To run this code, you need the following Python libraries:

- `numpy`
- `music21`
- `tensorflow`

---

## Code Structure

### MIDI File Processing
- **Loading MIDI Files**: 
  - The script loads MIDI files from a specified directory (`bach/`).
  - Files with the `.mid` extension are parsed using `music21.converter`.

- **Extracting Notes and Chords**:
  - Notes and chords are extracted from the MIDI files.
  - Single notes are represented by their pitch, while chords are represented by their normalized note values joined with a dot.

### Data Preparation
  - A sorted list of unique symbols (notes and chords) is created.
  - Two dictionaries are generated:
    - `symbol_to_index`: Maps each symbol to a unique integer index.
    - `index_to_symbol`: Maps each integer index back to its corresponding symbol.

### Model Definition
  - The model consists of two LSTM layers. Dropout layers are added after each LSTM layer to prevent overfitting.
  - A fully connected layer with 256 units is added after the second LSTM layer.
  - The output layer uses a softmax activation to predict the probability distribution over all possible notes.

### Training
  - Categorical cross-entropy is used as the loss function to measure the difference between predicted and actual notes.
  - The Adam optimizer is used to update the model weights during training.

### Music Generation
- **Sequence Generation**:
  - A random input sequence is selected from the training data.
  - The model predicts the next note in the sequence iteratively, updating the input sequence each time.

- **MIDI File Creation**:
  - The generated sequence of notes is converted into a `music21` stream.
  - The stream is saved as a MIDI file (`generated_midi.mid`).

---

## Usage

1. Place MIDI files in the `bach/` directory.

2. Run the script to train the model and generate music:

   ```bash
   python music_LSTM.py

3. The generated MIDI file will be saved as `generated_midi.mid`.
    
