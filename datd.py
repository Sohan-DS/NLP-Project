import json
import spacy
from spacy import displacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import time
import pickle

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

file_1 = load_json("json_files/file1.json")
file_2 = load_json("json_files/file2.json")
file_3 = load_json("json_files/file3.json")
file_4 = load_json("json_files/file4.json")
file_5 = load_json("json_files/file5.json")
file_6 = load_json("json_files/file6.json")
file_7 = load_json("json_files/file7.json")
file_8 = load_json("json_files/file8.json")
file_9 = load_json("json_files/file9.json")
file_10 = load_json("json_files/file10.json")
file_11 = load_json("json_files/file11.json")
file_12 = load_json("json_files/file12.json")
file_13 = load_json("json_files/file13.json")
file_14 = load_json("json_files/file14.json")
file_15 = load_json("json_files/file15.json")

# Convert JSON data to training format
def convert_data(data):
    training_data = []
    for item in data.get("annotations", []):
        if item is None:
            continue
        try:
            text, annotations = item
            if not text or not annotations or "entities" not in annotations:
                continue
            entities = [(start, end, label) for start, end, label in annotations["entities"]]
            training_data.append((text, {"entities": entities}))
        except ValueError:
            continue
    return training_data

# Combine training data from all files
training_data = []
files = [file_1, file_2, file_3, file_4, file_5, file_6, file_7, file_8, file_9, file_10, file_11, file_12, file_13, file_14, file_15]
for file in files:
    training_data.extend(convert_data(file))

# Split data into training and validation sets
random.shuffle(training_data)
split = int(len(training_data) * 0.8)
train_data = training_data[:split]
val_data = training_data[split:]

# Create a blank spaCy model
nlp = spacy.blank("en")

# Create the NER component and add it to the pipeline
ner = nlp.add_pipe("ner")

# Add new labels to the NER component
for _, annotations in training_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Disable other pipeline components (if any)
pipe_exceptions = ["ner"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Training the NER model
epochs = 10
batch_size = 32

with nlp.disable_pipes(*unaffected_pipes):
    optimizer = nlp.begin_training()
    for epoch in range(epochs):
        start_time = time.time()
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=batch_size)
        for batch in batches:
            texts, annotations = zip(*batch)
            examples = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in zip(texts, annotations)]
            nlp.update(examples, drop=0.5, losses=losses)

        # Calculate validation loss
        val_loss = 0.0
        for text, ann in val_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, ann)
            updates = ner.update([example], sgd=optimizer, drop=0.0)
            val_loss += updates["ner"]

        end_time = time.time()
        print(f"Epoch {epoch+1}/{epochs}")
        print(f" - {len(train_data)}/{len(train_data)} - {round(end_time - start_time, 2)}s - loss: {losses['ner']} - val_loss: {val_loss}")

# Save the trained model using pickle
with open("datd_model.pkl", "wb") as f:
    pickle.dump(nlp, f)
import pickle
import spacy
from spacy import displacy

# Load the model using pickle
with open("datd_model.pkl", "rb") as f:
    nlp = pickle.load(f)

# Input text
test_text = '''Just happened a terrible car crash
Heard about #earthquake is different cities, stay safe everyone.
there is a forest fire at spot pond, geese are fleeing across the street, I cannot save them all
Apocalypse lighting. #Spokane #wildfires
Typhoon Soudelor kills 28 in China and Taiwan
Arson suspect linked to 30 fires caught in Northern California http://t.co/HkFPyNb4PS
'''

# Process the text
doc = nlp(test_text)

# Define custom colors for entities
colors = {
    "DISASTER": "pink",
    "LOCATION": "#8da0cb",
    "DATE": "yellow",
    "NUMBER": "grey",
}

# Create options for displaCy with custom colors
options = {"colors": colors}

# Visualize the recognized entities with custom colors
displacy.render(doc, style="ent", jupyter=True, options=options)