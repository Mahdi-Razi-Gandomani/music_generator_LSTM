# Music Generation with LSTM

This project generates original MIDI music using a **Long Short-Term Memory (LSTM)** neural network trained on existing MIDI files. The model learns sequences of notes and chords to compose new melodies.

---

## Code Structure

### MIDI File Processing
- **Loading MIDI Files**: 
  - Loads MIDI files from an input directory.
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

### 0. Clone the Repository

```bash
git clone https://github.com/Mahdi-Razi-Gandomani/music_generator_LSTM.git
cd music_generator_LSTM
```

### 1. Add MIDI Files

Place your training `.mid` files inside the `input/` folder.  
These will be used to train the model.

### 2. Run the Script

Execute the Python file:

```bash
python3 music_LSTM.py
```

The script will:

- Parse and extract notes from your input files  
- Train an LSTM model  
- Generate a new sequence of notes  
- Save it as `generated_midi.mid` in your working directory  

### 3. Play the Generated Music

Open `generated_midi.mid` in a MIDI player.
