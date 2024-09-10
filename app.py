from joblib import load
import requests
import pickle
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from flask import Flask, render_template, request, redirect, url_for
import os
from flask_bcrypt import Bcrypt
import csv

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Set your secret key for Flask sessions
bcrypt = Bcrypt(app)

# Fertilizer information dictionary
fertilizer_info = {
    "UREA": {
        "description": "Urea is the most important nitrogenous fertilizer in the market, with the highest Nitrogen content (about 46 percent).",
        "crops": ["Wheat", "Sugar beet", "Perennial grasses", "Grain corn"],
        "how_to_use": [
            "Apply at sowing or as top dressing.",
            "Use in combination with earth or sand.",
        ],
        "price": "Rs 5,360 per tonne",
        "image_path": "static/img/fertilizer/urea.jpg",
    },
    "DAP": {
        "description": "DAP is an excellent source of phosphorus (P) and nitrogen (N) for plant nutrition.",
        "crops": ["Cereals", "Vegetables", "Orchards"],
        "how_to_use": [
            "Apply at the base of the plant during sowing.",
            "Mix well with soil.",
        ],
        "price": "Rs 24,000 per tonne",
        "image_path": "static/img/fertilizer/Dap.png",
    },
    "10-26-26": {
        "description": "A balanced fertilizer providing a good ratio of Nitrogen, Phosphorous, and Potassium.",
        "crops": ["All types of crops"],
        "how_to_use": [
            "Apply during planting or as top dressing.",
            "Mix well with soil.",
        ],
        "price": "Rs 18,000 per tonne",
        "image_path": "static/img/fertilizer/10-26-26.jpg",
    },
    "14-35-14": {
        "description": "Provides essential nutrients for plant growth, particularly Phosphorus.",
        "crops": ["Cereals", "Legumes", "Vegetables"],
        "how_to_use": [
            "Apply before planting.",
            "Mix thoroughly with soil.",
        ],
        "price": "Rs 22,000 per tonne",
        "image_path": "static/img/fertilizer/14-35-14.jpg",
    },
    "17-17-17": {
        "description": "A balanced NPK fertilizer suitable for a wide range of crops.",
        "crops": ["Vegetables", "Fruit trees", "Field crops"],
        "how_to_use": [
            "Apply during the growing season.",
            "Distribute evenly around the base of plants.",
        ],
        "price": "Rs 25,000 per tonne",
        "image_path": "static/img/fertilizer/17-17-17.jpg",
    },
    "20-20": {
        "description": "A balanced fertilizer with equal amounts of Nitrogen and Phosphorus.",
        "crops": ["General use for various crops"],
        "how_to_use": [
            "Apply during planting or early growth stages.",
            "Mix well with soil.",
        ],
        "price": "Rs 21,000 per tonne",
        "image_path": "static/img/fertilizer/20-20.jpg",
    },
    "28-28": {
        "description": "High Nitrogen fertilizer ideal for crops requiring increased Nitrogen levels.",
        "crops": ["Corn", "Wheat", "Soybeans"],
        "how_to_use": [
            "Apply at planting or as top dressing.",
            "Incorporate into the soil.",
        ],
        "price": "Rs 26,000 per tonne",
        "image_path": "static/img/fertilizer/28-28.png",
    },
}


class ConvNet(nn.Module):
    def __init__(self, num_classes=8):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(in_features=32 * 75 * 75, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


# Load the model
checkpoint = torch.load("wheatdisease.model", map_location=torch.device("cpu"))
model = ConvNet(num_classes=8)
model.load_state_dict(checkpoint)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Class labels for the model
classes = [
    "Brownrust",
    "Crown and Root Rot",
    "Healthy Wheat",
    "Leaf rust",
    "sepotoria",
    "strip rust",
    "wheat loose smut",
    "yellow rust",
]

# Fertilizer recommendations based on the disease
fertilizer_recommendations = {
    "Brownrust": "Ammonium sulfate or urea, phosphatic fertilizers, superphosphate and potassium fertilizers like potassium chloride.",
    "Crown and Root Rot": "Thiophanate-methyl fungicides",
    "Healthy Wheat": "The Wheat is Healthy",
    "Leaf rust": "Ammonium nitrate or urea",
    "sepotoria": "Ammonium sulfate or urea",
    "strip rust": "Ammonium nitrate, phosphorus, and potassium",
    "wheat loose smut": "Azotobacter",
    "yellow rust": "Roline275 or Aviator235Xpro",
}


def preprocess_image(image):
    image = Image.fromarray(image)
    image = image.resize((150, 150))
    image = np.array(image) / 255.0  # Normalize the image
    image = torch.tensor(image, dtype=torch.float32)
    image = image.permute(2, 0, 1)  # Convert to (C, H, W) format
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    return image


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/contactmail", methods=["GET"])
def contactmail_form():
    return render_template("contactmail.html")


@app.route("/contactmail", methods=["POST"])
def contactmail():
    # Retrieve form data
    name = request.form.get("name")
    email = request.form.get("email")
    subject = request.form.get("subject")
    message = request.form.get("message")

    # CSV file path
    csv_file = "contact.csv"

    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_file)

    # Open the CSV file in append mode
    with open(csv_file, mode="a", newline="") as file:
        fieldnames = ["Name", "Email", "Subject", "Message"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # If the file doesn't exist, write the header
        if not file_exists:
            writer.writeheader()

        # Write the form data to the CSV file
        writer.writerow(
            {"Name": name, "Email": email, "Subject": subject, "Message": message}
        )

    return redirect(url_for("index"))


@app.route("/fertilizer")
def fertilizer():
    return render_template("fertilizer.html")


@app.route("/wheat")
def wheat():
    return render_template("wheat.html")


@app.route("/disease")
def disease():
    return render_template("disease.html")


@app.route("/weather", methods=["GET", "POST"])
def weather():
    return render_template("weather.html")


@app.route("/weather_today", methods=["POST"])
def weather_today():
    city = request.form.get("city")
    api_key = "db5dbc8cf25affca2c9f131fce71faad"
    weather_api_url = (
        f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    )
    response = requests.get(weather_api_url)
    if response.status_code == 200:
        weather_data = response.json()
        temperature = weather_data["main"]["temp"]
        description = weather_data["weather"][0]["description"]
        return render_template(
            "weather.html", city=city, temperature=temperature, description=description
        )
    else:
        return render_template(
            "weather.html", error="City not found. Please try again."
        )


@app.route("/yield")
def yield_page():
    yield_info = {
        "title": "Yield Information",
        "description": "This page contains information about crop yield.",
    }
    return render_template("yield.html", yield_info=yield_info)


@app.route("/yield_predict", methods=["POST"])
def yield_prediction():
    model = load("wheat_yield_prediction_model.joblib")
    with open("label_encoders.pkl", "rb") as file:
        label_encoders = pickle.load(file)

    state = request.form["state"]
    district = request.form["district"]
    month = request.form["month"]
    print(state)
    print(district)
    print(month)

    if not month:
        return "Please provide a valid month"
    state_encoded = label_encoders["State"].transform([state])[0]
    district_encoded = label_encoders["District"].transform([district])[0]
    month_encoded = label_encoders["Month"].transform([month])[0]
    prediction = model.predict([[state_encoded, district_encoded, month_encoded]])
    print(prediction[0])
    return str(prediction[0])


@app.route("/predict", methods=["POST"])
def predict():
    nitrogen_str = request.form.get("Nitrogen", "").strip()
    potassium_str = request.form.get("Potassium", "").strip()
    phosphorous_str = request.form.get("Phosphorous", "").strip()

    # Check if the inputs are valid
    try:
        nitrogen = float(nitrogen_str) if nitrogen_str else 0
        potassium = float(potassium_str) if potassium_str else 0
        phosphorous = float(phosphorous_str) if phosphorous_str else 0
    except ValueError as e:
        return f"Invalid input: {e}"

    # Load models and scaler
    with open("knn_model.pkl", "rb") as model_file:
        knn_model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    # Predict fertilizer
    try:
        numerical_result = knn_model.predict(
            scaler.transform([[nitrogen, potassium, phosphorous]])
        )
        category_mapping = {
            0: "DAP",
            1: "14-35-14",
            2: "17-17-17",
            3: "10-26-26",
            4: "28-28",
            5: "20-20",
            6: "UREA",
        }
        categorical_result = category_mapping.get(numerical_result[0], "Unknown")

        return render_template(
            "fertilizer.html",
            result=categorical_result,
            fertilizer_info=fertilizer_info.get(categorical_result, None),
        )
    except Exception as e:
        return f"An error occurred during prediction: {e}"


@app.route("/prediction", methods=["POST"])
def prediction():
    if "image" not in request.files:
        return "No image provided"
    image = request.files["image"]
    if image.filename == "":
        return "No selected image file"
    if image:
        image_path = os.path.join("static/uploaded_images", image.filename)
        image.save(image_path)
        image_cv = cv2.imread(image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image_tensor = preprocess_image(image_cv)
        with torch.no_grad():
            outputs = model(image_tensor)
        # Debugging output
        print("Raw Model Outputs:", outputs)
        print("Predicted Class Scores:", outputs.cpu().numpy())
        _, predicted_class = torch.max(outputs, 1)
        class_label = classes[predicted_class.item()]
        print("Predicted Class Label:", class_label)
        fertilizer_recommendation = fertilizer_recommendations.get(
            class_label, "No recommendation found"
        )
        predicted_image = "uploaded_images/" + image.filename
        return render_template(
            "result.html",
            class_label=class_label,
            predicted_image=predicted_image,
            fertilizer=fertilizer_recommendation,
        )
    return "Prediction failed"


@app.route("/submit", methods=["POST"])
def submit_form():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        subject = request.form["subject"]
        message = request.form["message"]

        # Save data to a CSV file
        with open("contact.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([name, email, subject, message])

        return redirect(url_for("thank_you"))


@app.route("/thank_you")
def thank_you():
    return "Thank you for contacting us!"


if __name__ == "__main__":
    app.run(debug=True)
