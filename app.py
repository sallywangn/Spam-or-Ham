# Create API of ML model using flask

'''
This code takes in a file upload (.eml format) and POST request an performs the prediction using loaded model and returns
the result.
'''

# Import libraries
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import mailparser
from bs4 import BeautifulSoup
import re

app = Flask(__name__)

# Load the model
model = pickle.load(open('./log_reg_model.pkl','rb'))

# Read in spambase word list from columns.csv using Pandas
spambase_df = pd.read_csv('./ML/columns.csv', header=None)
spambase_list = list(spambase_df.iloc[:,0])

@app.route('/')
def home():
    return render_template("index.html")

# Additional route created for future use
# @app.route('/success', methods=['POST'])
# def success():
#     if request.method == 'POST':   
#         f = request.files['file'] 
#         f.save(f.filename)   
#         return render_template("acknowledgment.html", name = f.filename)

@app.route('/predict', methods=['POST'])
def predict():
    # Call variables
    output = ""
    text_1 = ""
    # Get the data from the POST request.
    if request.method == "POST":
        f = request.files['file'] 
        f.save(f.filename)
        
        def clean_text(text):
            p1 = r'[^\x00-\x7F]+' # non-ASCII characters
            p2 = r'[^a-z ]' # characters not in lowercase alphabet
            p3 = r'( {2,})' # more than one space between words in string
            text = re.sub(p1, '', text)
            text = re.sub(p2, '', text)
            text = re.sub(p3, ' ', text)
            return text
        
        # Parse from string (copy/paste text)
        mail = mailparser.parse_from_file(f.filename)
        print('HTML is', bool(mail.text_html))
        print('Text is', bool(mail.text_plain))

        # Content type check (plain text or HTML)
        # File parse error check (for content type)
        if not bool(mail.text_html) and not bool(mail.text_plain):
            output = "Error - Content-Type missing"
            return render_template("results.html", output=output)
        
        elif bool(mail.text_plain):
            raw_text = mail.text_plain[0]
            # Strip whitespace at string ends, replace newline and nbsp characters, and set lowercase
            text_1 = raw_text.strip().replace('\n',' ').replace('\xa0',' ').lower()
        
        elif bool(mail.text_html):
            # Raw text
            soup = BeautifulSoup(mail.text_html[0], 'html.parser')
            # Strip whitespace, newlines, and lowercase
            text_1 = soup.get_text(separator='\n', strip=True).replace('\n'," ").lower()
        
        # Run regex clean function
        cleaned_text = clean_text(text_1)
        # Word list from parsed/cleaned text
        word_list = cleaned_text.split()
        
        # Get word count of parsed/cleaned text
        word_df = pd.DataFrame({'words':word_list})
        word_count_df = pd.DataFrame(word_df.groupby('words').value_counts())
        word_count_df = word_count_df.rename(columns={0:'count'})
        word_count_df.reset_index(names='word',inplace=True)
        # Get word count list to match against 'spambase_list'
        word_count_list = list(word_count_df.iloc[:,0])

        # Create sample_dict
        sample_dict = {}
        # Build sample
        for word in spambase_list:
            if word not in word_count_list:
                # Set word count to 0 for 'spambase_list' words not in 'word_count_list'
                sample_dict[word] = 0
            else:
                # Get count value of word from 'word_count_df'
                sample_dict[word] = word_count_df[word_count_df['word']==word]['count'].values[0]
        
        # Create X_sample from sample e-mail text information (word count)
        X_sample = pd.Series(sample_dict)
        # Sample prediction
        prediction = model.predict([X_sample])[0]

        if prediction:
            output = 'SPAM!'
        else:
            output = 'HAM!'
        
        return render_template("results.html", output=output, tables=[word_count_df.to_html(classes='data', header='true')])

if __name__ == '__main__':
    app.run(debug=True)
