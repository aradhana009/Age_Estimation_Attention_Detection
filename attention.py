

import cv2
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("attention_data.csv")

# Select features
X = data[["movement", "position"]]

# Select target
y = data["attention"]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Start webcam
cap = cv2.VideoCapture(0)

previous = None
previous_position = 50

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Fixed face region
    roi = frame[100:350, 200:450]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Calculate movement
    if previous is None:
        movement = 0
    else:
        difference = cv2.absdiff(gray, previous)
        movement = difference.mean()

    previous = gray.copy()

    # Calculate position using brightness
    left = gray[:, :125].mean()
    right = gray[:, 125:].mean()

    if left > right:
        position = 45
    else:
        position = 55

    # Smooth the position
    position = (previous_position + position) / 2
    previous_position = position

    # Prepare input with feature names
    new_data = pd.DataFrame(
        [[movement, position]],
        columns=["movement", "position"]
    )

    # Predict attention
    prediction = model.predict(new_data)[0]

    # Keep score between 0 and 100
    prediction = max(0, min(100, prediction))

    # Decide attention
    if prediction >= 60:
        status = "ATTENTIVE"
    else:
        status = "NOT ATTENTIVE"

    # Draw ROI
    cv2.rectangle(
        frame,
        (200, 100),
        (450, 350),
        (0, 255, 0),
        2
    )

    # Display movement
    cv2.putText(
        frame,
        "Movement: " + str(round(movement, 1)),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Display position
    cv2.putText(
        frame,
        "Position: " + str(round(position, 1)),
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Display attention score
    cv2.putText(
        frame,
        "Attention: " + str(round(prediction, 1)),
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Display status
    cv2.putText(
        frame,
        status,
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow("Real-Time Attention Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

