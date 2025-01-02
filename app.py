from flask import Flask, render_template, request
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = pickle.load(open('random_forest_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Route to handle home page and email input
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get email content from the form
        email_content = request.form['email_content']
        
        # Vectorize the email content using the loaded vectorizer
        email_vectorized = vectorizer.transform([email_content])
        
        # Predict using the loaded model
        prediction = model.predict(email_vectorized)
        
        # Display the result based on the prediction
        if prediction[0] == 1:
            result = 'This is a SPAM email.'
        else:
            result = 'This is NOT a SPAM email.'
        
        return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
