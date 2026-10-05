"""
Phishing Email Detection Model using Scikit-Learn
Classifies email text as 'Safe' or 'Phishing'.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Dataset: Sample emails (Safe vs Phishing)
data = {
    "text": [
        "Dear customer, your bank account is locked. Click http://secure-login-verify.com immediately",
        "Hey, are we still meeting tomorrow for lunch at the cafeteria?",
        "URGENT: Your PayPal has been suspended! Update your password at http://pay-pal-fake.ru now",
        "Attached is the project presentation schedule for next week's review meeting.",
        "Congratulations! You won a $1,000 Walmart gift card. Claim now at http://win-free-rewards.biz",
        "Hi, please review the lab report by this Friday before class.",
        "Security Alert: Unusual login from unknown device. Verify credentials: http://account-update-web.com",
        "Team, please find the minutes of today's technical standup attached in the email.",
        "Your Netflix subscription has expired! Click http://netflix-billing-renew.com to renew",
        "Don't forget to push your code commits to the GitHub repository before midnight.",
        "Final Notice: Wire transfer pending. Enter your details at http://wire-transfer-gate.cc",
        "Let me know when you're free so we can discuss the capstone architecture."
    ],
    "label": [
        "Phishing", "Safe", "Phishing", "Safe", "Phishing", "Safe",
        "Phishing", "Safe", "Phishing", "Safe", "Phishing", "Safe"
    ]
}

df = pd.DataFrame(data)

# 2. Split dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], 
    df["label"], 
    test_size=0.33, 
    random_state=42
)

# 3. Build ML Pipeline (TF-IDF Feature Extractor + Multinomial Naive Bayes)
model_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", MultinomialNB())
])

# 4. Train the Model
model_pipeline.fit(X_train, y_train)

# 5. Evaluate the Model
predictions = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions, labels=["Safe", "Phishing"])

print("=" * 45)
print("PHISHING EMAIL DETECTION MODEL RESULTS")
print("=" * 45)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Confusion Matrix [Labels: Safe, Phishing]:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("=" * 45)

# 6. Interactive Testing Function
def test_email(email_content: str):
    result = model_pipeline.predict([email_content])[0]
    probabilities = model_pipeline.predict_proba([email_content])[0]
    labels = model_pipeline.classes_
    prob_dict = dict(zip(labels, probabilities))
    
    print(f"\nIncoming Email: \"{email_content}\"")
    print(f"Prediction: [{result.upper()}]")
    print(f"Confidence: Safe: {prob_dict['Safe']*100:.1f}%, Phishing: {prob_dict['Phishing']*100:.1f}%")

if __name__ == "__main__":
    sample_safe = "Can you send me the lecture notes for operating systems?"
    sample_phish = "Your iCloud storage is full. Restore access at http://icloud-storage-fix.xyz"
    
    test_email(sample_safe)
    test_email(sample_phish)
