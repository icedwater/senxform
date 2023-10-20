#! /usr/bin/env python
"""
Exploring basic sentence similarity with various models
"""

from sentence_transformers import SentenceTransformer, util

def create_model(model_name="all-MiniLM-L12-v2"):
    """
    Create the model using the specified model name.
    """
    return SentenceTransformer(model_name)

def add_sentence(sentences :list, sentence :str) -> list:
    """
    Add a sentence to a list and return the list.
    """
    sentences.append(sentence)
    return sentences

def find_embeddings(sentences :list, model) -> list:
    """
    Process each sentence in a list and return only their embeddings.
    """
    return model.encode(sentences)

def choose_model() -> str:
    """
    Display a menu and let user choose a model to compute similarity.
    """
    choice = int(input("Use (1) L6 or (2) L12 model? "))
    if choice == 1:
        return "all-MiniLM-L6-v2"
    elif choice == 2:
        return "all-MiniLM-L12-v2"

def main():
    """
    Dumb main function.
    """
    model_name = choose_model()
    model = create_model(model_name=model_name)

    base_list = []
    base_list = add_sentence(base_list, "hello")
    base_list = add_sentence(base_list, "hi")

    embeddings = []
    embeddings = find_embeddings(base_list, model)

    print(f"Sentences: {base_list}")
    print(f"Cosine similarity: {util.cos_sim(embeddings[0], embeddings[1])}.")

if __name__ == "__main__":
    main()
