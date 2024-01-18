# credit_card_churn_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class CreditCardChurnModel:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self._preprocess_data()

    def _preprocess_data(self):
        # Drop unnecessary columns for modeling
        self.df = self.df.drop(["CLIENTNUM", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"], axis=1)

        # Convert categorical variables to numerical using Label Encoding
        le = LabelEncoder()
        self.df["Gender"] = le.fit_transform(self.df["Gender"])
        self.df["Education_Level"] = le.fit_transform(self.df["Education_Level"])
        self.df["Marital_Status"] = le.fit_transform(self.df["Marital_Status"])
        self.df["Income_Category"] = le.fit_transform(self.df["Income_Category"])
        self.df["Card_Category"] = le.fit_transform(self.df["Card_Category"])
        self.df["Attrition_Flag"] = le.fit_transform(self.df["Attrition_Flag"])

    def _split_data(self):
        # Split the data into features (X) and target variable (y)
        X = self.df.drop("Attrition_Flag", axis=1)
        y = self.df["Attrition_Flag"]

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        # Create and train a logistic regression model
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)

        # Evaluate the model
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        classification_rep = classification_report(self.y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Classification Report:\n{classification_rep}")
