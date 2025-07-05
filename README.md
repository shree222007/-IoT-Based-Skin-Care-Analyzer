# -IoT-Based-Skin-Care-Analyzer
An IoT-based skincare analyzer using ESP32 and multiple sensors (TCS3200, Moisture, UV, DHT11). It collects skin and environmental data to predict skin type using a neural network. The system recommends personalized skincare routines based on live sensor inputs.

##  Hardware Components
- **ESP32** – Microcontroller
- **TCS3200** – Color sensor (detects oil level on skin)
- **Moisture Sensor** – Measures skin hydration
- **DHT11** – Temperature and humidity sensor
- **UV Sensor** – Detects UV light intensity

##  Software & Intelligence
- Sensor data is collected by ESP32 and sent via Wi-Fi
- A Feedforward Neural Network (FNN) or CNN processes the data
- Predicts skin type: **Dry**, **Oily**, or **Normal**
- Provides skincare suggestions accordingly

##  Features
- Real-time sensor readings
- Wi-Fi-enabled data transmission
- Flask-based API for predictions
- Clean circuit layout for academic use
- Lightweight neural network model

