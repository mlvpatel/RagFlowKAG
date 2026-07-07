"""Golden evaluation set: documents and questions with their relevant document.

Small and hand labeled so the retrieval metrics are meaningful and repeatable.
Each document has a distinct topic. Each question is labeled with the set of
document ids that should be retrieved to answer it.
"""

DOCUMENTS = {
    1: "Paris is the capital of France. The Eiffel Tower is a famous landmark in Paris.",
    2: "Python is a high level programming language created by Guido van Rossum in 1991.",
    3: "The mitochondria is the organelle that produces energy in a cell.",
    4: "Mount Everest is the highest mountain above sea level, located in the Himalayas.",
    5: "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
    6: "The Great Wall of China is over thirteen thousand miles long and was built over many centuries.",
}

QUESTIONS = [
    ("What is the capital of France?", {1}),
    ("Where is the Eiffel Tower located?", {1}),
    ("Who created the Python programming language?", {2}),
    ("What produces energy in a cell?", {3}),
    ("What is the highest mountain on Earth?", {4}),
    ("Where is Mount Everest?", {4}),
    ("How do plants convert sunlight into energy?", {5}),
    ("How long is the Great Wall of China?", {6}),
]
