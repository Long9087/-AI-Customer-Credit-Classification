# main.py
from credit_card_churn_model import CreditCardChurnModel

def main():
    # Update the path to your CSV file
    csv_path = "Dataset/BankChurners.csv"
    
    # Example usage
    credit_card_model = CreditCardChurnModel(csv_path)
    credit_card_model._split_data()
    credit_card_model.train_model()
    credit_card_model.evaluate_model()

if __name__ == "__main__":
    main()
