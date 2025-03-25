import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re

# Define the neural network
class FoodPredictionNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FoodPredictionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load and preprocess the data
def load_and_preprocess_data(filename):
    data = pd.read_csv(filename)
    
    data = clean_data(data)

    # Select features for prediction (example: Q1, Q2_cleaned, etc.)
    feature_columns = [
    "Q1",
    "Q2",
    "Q4",
    # "Q5: What movie do you think of when thinking of this food item?", Removing for now
    "Q8",
    "Week day lunch",
    "Week day dinner",
    "Weekend lunch",
    "Weekend dinner",
    "At a party",
    "Late night snack",
    "soda",
    "other",
    "tea",
    "alcohol",
    "water",
    "soup",
    "juice",
    "milk",
    "unknown",
    "smoothie",
    "asian alcohol",
    "asian pop",
    "milkshake",
    "Parents",
    "Siblings",
    "Friends",
    "Teachers",
    "Strangers"
    ]
    features = data[feature_columns].fillna(0).values
    labels = data['Label'].map({'Pizza': 0, 'Shawarma': 1, 'Sushi': 2}).values  # Convert labels to integers

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test, scaler

def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, dim=1)  # Get the predicted class
    correct = (predicted == labels).sum().item()  # Count correct predictions
    accuracy = correct / labels.size(0)  # Calculate accuracy
    return accuracy

# Train the model
def train_model(X_train, y_train, input_size, hidden_size, output_size, num_epochs=100, learning_rate=0.01):
    model = FoodPredictionNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        train_accuracy = compute_accuracy(outputs, y_train)
        test_outputs = model(X_test)
        test_accuracy = compute_accuracy(test_outputs, y_test)

        if (epoch + 1) % 10 == 0:
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

    return model

# Save the model
def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def clean_data(df):
    """
    Clean and format the data in the dataframe
    """
    q1_label = "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)"
    q2_label = "Q2: How many ingredients would you expect this food item to contain?"
    q3_label = "Q3: In what setting would you expect this food to be served? Please check all that apply"
    q4_label = "Q4: How much would you expect to pay for one serving of this food item?"
    q5_label = "Q5: What movie do you think of when thinking of this food item?"
    q6_label = "Q6: What drink would you pair with this food item?"
    q7_label = "Q7: When you think about this food item, who does it remind you of?"
    q8_label = "Q8: How much hot sauce would you add to this food item?"
    
    # Categorical Feature categories
    Q3_categories = ["Week day lunch", "Week day dinner", "Weekend lunch", "Weekend dinner", "At a party", "Late night snack"]
    Q6_categories = [
        "soda", "other", "tea", "alcohol", "water", "soup", "juice", "milk", 
        "unknown", "smoothie", "asian alcohol", "asian pop", "milkshake"
    ]
    Q7_categories = ["Parents", "Siblings", "Friends", "Teachers", "Strangers"]
    Q8_categories = ["None", "A little (mild)", "A moderate amount (medium)", "A lot (hot)", "I will have some of this food item with my hot sauce"]

    # Clean Q2
    df['Q2'] = df[q2_label].apply(clean_Q2)

    # Convert cleaned to numeric
    df['Q2'] = pd.to_numeric(df['Q2'], errors='coerce')

    # Extract columns for Q3, Q6, Q7, and Q8
    q3 = df[q3_label]
    q6 = df[q6_label]
    q7 = df[q7_label]
    q8 = df[q8_label]

    # One hot encode Q3, Q6, and Q7
    q3 = np.array([one_hot_encode(response, Q3_categories) for response in q3])
    q6 = np.array([one_hot_encode(response, Q6_categories) for response in q6])
    q7 = np.array([one_hot_encode(response, Q7_categories) for response in q7])

    # Create new columns for Q3, Q6, and Q7
    for i, category in enumerate(Q3_categories):
        df[category] = q3[:, i]
    for i, category in enumerate(Q6_categories):
        df[category] = q6[:, i]
    for i, category in enumerate(Q7_categories):
        df[category] = q7[:, i]

    # Convert Q8 to ordinal scale (0, 1, 2, 3, 4)
    ordinal_mapping = {category: idx for idx, category in enumerate(Q8_categories)}
    df['Q8'] = df[q8_label].map(ordinal_mapping)

    df = df.rename(columns={q4_label: 'Q4'})
    # Remove dollar signs from Q4
    df["Q4"] = df["Q4"].str.replace("$", "")

    # Drop original categorical columns
    df = df.drop(columns=[
        q3_label, q6_label, q7_label, q8_label
    ])

    # Change Q1 label
    df = df.rename(columns={q1_label: 'Q1'})
    # Make sure its numeric
    df['Q1'] = pd.to_numeric(df['Q1'], errors='coerce')

    return df


def clean_Q2(q2):
    """
    Clean column Q2 in csv file, converting to all numerical discrete values.
    Handles:
    - Numerical values (e.g., "5")
    - Ranges (e.g., "4-6")
    - Textual descriptions (e.g., "around 3")
    - Spelled-out numbers (e.g., "three")
    - Ingredient lists (e.g., "bread, meat, cheese")
    - Multi-line ingredient lists (e.g., "I would expect it to contain:\n* Bread\n* Cheese")
    - #NAME? and other non-numeric values
    """
    # Handle NaN values
    if pd.isna(q2):
        return pd.NA
    
    # Convert to string if not already
    q2 = str(q2).strip()
    
    # Handle "I don't know" or similar cases
    if '#NAME?' in q2 or 'don\'t know' in q2.lower() or 'dont know' in q2.lower() or 'no idea' in q2.lower():
        print(f"Skipping Q2 value: {q2}")
        return pd.NA
    
    # Handle ranges like "4-6" or "5 to 7"
    range_match = re.search(r'(\d+)\s*[-~to]+\s*(\d+)', q2)
    if range_match:
        low = int(range_match.group(1))
        high = int(range_match.group(2))
        return (low + high) // 2  # Return the floored average
    
    # Handle single numbers
    single_number_match = re.search(r'\d+', q2)
    if single_number_match:
        return int(single_number_match.group(0))
    
    # Handle textual descriptions like "around 5" or "about 3"
    textual_match = re.search(r'(around|about|approximately|~)\s*(\d+)', q2, re.IGNORECASE)
    if textual_match:
        return int(textual_match.group(2))
    
    # Handle cases where the number is spelled out (e.g., "three")
    spelled_out_numbers = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    for word, num in spelled_out_numbers.items():
        if word in q2.lower():
            return num
    
    # Handle ingredient lists
    # Check if the text looks like a list of ingredients (may be an issue for those who answered in sentences)
    if ',' in q2 or '\n' in q2 or ' and ' in q2.lower():
        lines = q2.split('\n')
        ingredients = []
        for line in lines:
            line = re.sub(r'[*\-•]', '', line).strip()
            if line:
                parts = re.split(r'[,&]| and ', line)
                ingredients.extend([part.strip() for part in parts if part.strip()])
        unique_ingredients = set(ingredients)
        return len(unique_ingredients)
    
    # If no number or ingredients are found, return 'n/a'
    print(f"Could not parse Q2 value: {q2}")
    return pd.NA

def one_hot_encode(response_str, categories):
    result = np.zeros(len(categories))
    if isinstance(response_str, str):  # Check if it's a string (not NaN)
        selections = response_str.split(',')
        for i, category in enumerate(categories):
            if category in selections:
                result[i] = 1
    return result

def evaluate_model(model, X_test, y_test):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        outputs = model(X_test)
        _, predicted = torch.max(outputs, dim=1)  # Get the predicted class
        correct = (predicted == y_test).sum().item()  # Count correct predictions
        accuracy = correct / y_test.size(0)  # Calculate accuracy
    return accuracy

# Example usage
if __name__ == "__main__":
    # Load and preprocess the data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("data/cleaned_data.csv")

    # Define model parameters
    input_size = X_train.shape[1]
    hidden_size = 16
    output_size = 3  # Number of classes (Pizza, Shawarma, Sushi)

    # Train the model
    model = train_model(X_train, y_train, input_size, hidden_size, output_size)

    # Save the model
    save_model(model, 'model.pth')

    # Save the scaler (optional, but recommended)
    import joblib
    joblib.dump(scaler, 'scaler.pkl')

    test_accuracy = evaluate_model(model, X_test, y_test)
    print(f'Test Accuracy: {test_accuracy:.4f}')