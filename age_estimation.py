

import cv2
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("age_data.csv")

# Features
X = data[["brightness", "texture"]]

# Target
y = data["age"]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Fixed region where face is expected
    roi = frame[100:350, 200:450]

    # Calculate brightness directly from color image
    brightness = roi.mean()

    # Calculate texture directly from color image
    texture = roi.std()

    # Prepare input
    new_data = pd.DataFrame(
        [[brightness, texture]],
        columns=["brightness", "texture"]
    )

    # Predict age
    age = model.predict(new_data)[0]

    # Draw region
    cv2.rectangle(
        frame,
        (200, 100),
        (450, 350),
        (0, 255, 0),
        2
    )

    # Display predicted age
    cv2.putText(
        frame,
        "Estimated Age: " + str(round(age, 1)),
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Age Estimation", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

