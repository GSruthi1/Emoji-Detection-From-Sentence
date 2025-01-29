import torch
import json
import pickle
import numpy as np
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template

#Create an app object using the Flask class. 
app = Flask(__name__)

# Tokenize tweets using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode labels
label_encoder = LabelEncoder()
with open('label_encoder_classes.json', 'r') as f:
    classes = json.load(f)
label_encoder.classes_ = np.array(classes)


class EmojiPredictionModel(nn.Module):
    def __init__(self, n_classes):
        super(EmojiPredictionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(768, 64)
    
    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[1]
        output = self.drop(pooled_output)
        return self.out(output)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmojiPredictionModel(len(label_encoder.classes_))
model.load_state_dict(torch.load('emoji_prediction_model.pth'))
model = model.to(device)

#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']  # Corrected to match the input field's name attribute
        
        # Assuming tokenizer is initialized somewhere globally
        inputs = tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=50,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            _, prediction = torch.max(outputs, dim=1)

        # Assuming label_encoder is initialized somewhere globally
        output = label_encoder.inverse_transform(prediction.cpu().numpy())
        return render_template('index.html', prediction_text='Predicted emoji: {}'.format(output))


#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run()