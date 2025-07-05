# code for giving the recommendation by getting the user input
# ===============================
# 4. Generate Recommendations
# ===============================
def get_recommendation(model, scaler, le_skin, le_label, input_data):
    """Generate personalized skincare recommendation"""
    try:
        # Process input
        skin_encoded = le_skin.transform([input_data['SkinTone']])
        features = np.array([
            [skin_encoded[0], input_data['Moisture'], input_data['Oil']]
        ])
        features = scaler.transform(features)
        
        # Predict
        pred = model.predict(features, verbose=0)
        rec_class = le_label.inverse_transform([np.argmax(pred)])
        
        # Get routine from JSON
        with open("C:/MiDerma/Recommendation_system.json") as f:
            routines = json.load(f)
        return routines.get(rec_class[0], "No recommendation found for this combination")
    
    except Exception as e:
        return f"Error generating recommendation: {str(e)}"

# ===============================
# 5. User Input & Prediction
# ===============================
def get_valid_input(prompt, input_type=float, min_val=None, max_val=None):
    """Validate user input with range checking"""
    while True:
        try:
            value = input_type(input(prompt))
            if min_val is not None and value < min_val:
                print(f"Value must be ≥ {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be ≤ {max_val}")
                continue
            return value
        except ValueError:
            print("Invalid input. Please try again.")

print("\n=== MiDerma Skin Analysis ===")
print("Available skin tones: fair, light, medium, tan_brown, deep_brown_black, olive")

skin_tone = input("Enter your skin tone: ").strip().lower()
while skin_tone not in OIL_THRESHOLDS:
    print("Invalid skin tone. Please choose from:", list(OIL_THRESHOLDS.keys()))
    skin_tone = input("Enter your skin tone: ").strip().lower()

moisture = get_valid_input(
    "Enter your skin moisture value (24-39): ",
    float, 24, 39
)
oil = get_valid_input(
    f"Enter your skin oil value ({OIL_THRESHOLDS[skin_tone]['low']}-{OIL_THRESHOLDS[skin_tone]['high']}): ",
    float,
    OIL_THRESHOLDS[skin_tone]['low'],
    OIL_THRESHOLDS[skin_tone]['high']
)

input_data = {
    'SkinTone': skin_tone,
    'Moisture': moisture,
    'Oil': oil
}

recommendation = get_recommendation(model, scaler, le_skin, le_label, input_data)
print("\n=== Recommended Routine ===")
print(recommendation)
