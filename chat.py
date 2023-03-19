import random
import json
import openai
import torch

from model import NeuralNet
from nltk_util import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
openai.api_key="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"
with open('dataset.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(sentence):
    message=sentence
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["intent"]:
                return (f"{random.choice(intent['responses'])}")
    else:
        #return (f"please ask queries related to this website")
        chatting=[{"role":"system","content":"You are a kind helpful assistant"}]
        chatting.append(
            {"role": "user","content":message},
        )
        chat=openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=chatting)
        reply=chat.choices[0].message.content
        chatting.append({"role":"assistant","content":reply})
        return reply

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)