# Smart Bird Feeder Scene Monitoring

## Project Overview
This project aims to build a simple embedded image classification system for a bird feeder scene.  
The system is designed to classify images into three categories:

- `bird`
- `squirrel`
- `empty`

The final goal is to detect when a squirrel appears near the feeder and trigger a response on the device.

## Hardware
- Arduino Nano 33 BLE Sense Lite
- OV7675 Camera
- Tiny Machine Learning Shield

## Software
- Arduino Cloud
- Edge Impulse

---

## Round 1 Model Performance

### Validation Result
- **Accuracy:** 70.1%
- **Loss:** 0.67

<img width="1084" height="1111" alt="image" src="https://github.com/user-attachments/assets/db29390c-d0c4-4120-9495-6af53bb718dc" />

### First Test Result
- **Accuracy:** 60.18%
- **Weighted Precision:** 0.69
- **Weighted Recall:** 0.68
- **Weighted F1 Score:** 0.68

<img width="1087" height="960" alt="image" src="https://github.com/user-attachments/assets/991a40e8-76b1-4a6e-a3e4-8697fd366e2e" />

### Initial Observations
- The model can already distinguish the three classes at a basic level.
- `bird` and `squirrel` perform better than `empty`.
- The `empty` class is currently the weakest category.
- The model still shows limited generalization on new test data.

---

## Final Round Model Performance

After the first-round baseline, the dataset was refined in order to improve class balance and reduce unstable predictions.

In the final round, more `empty` images were added to reduce false triggers in background scenes, and more `squirrel` images were later added to recover squirrel detection performance. This final round was used as the last model iteration for the project.

### Final Round Validation Result
- **Accuracy:** 73.6%
- **Loss:** 0.69

<img width="1091" height="1119" alt="image" src="https://github.com/user-attachments/assets/6b6077d3-c7a8-4cd8-af52-45ebf6c32b05" />

### Final Round Test Result
- **Accuracy:** 60.37%
- **Weighted Precision:** 0.72
- **Weighted Recall:** 0.71
- **Weighted F1 Score:** 0.72

<img width="1091" height="960" alt="image" src="https://github.com/user-attachments/assets/382c467a-0780-4def-863e-6b3b4b81b3c7" />

### Final Round Observations
- The final model achieved the best balance across the three classes.
- The `empty` class became more stable than in the first round.
- `squirrel` performance improved again after adding more squirrel samples.
- Although the overall test accuracy is still limited, this version was the most suitable model for the final embedded demo.

---

## Final Arduino Integration

After the final model was selected in Edge Impulse, it was deployed to the Arduino Nano 33 BLE Sense with the OV7675 camera.

The Arduino system performs:

- image capture from the OV7675 camera
- on-device inference using the exported Edge Impulse model
- state-based output using the built-in LED
- Serial Monitor logging for debugging and demonstration

<img width="1240" height="510" alt="image" src="https://github.com/user-attachments/assets/dadcb9df-72ac-43ae-abd8-9a049d8ee8be" />

---

## Final Device Logic

For the final device logic and LED flashing logic, please refer to Document README_arduino_led_logic.md
