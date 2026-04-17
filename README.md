# Smart Bird Feeder Intruder Alert

## Project Overview
This project aims to build a simple embedded image classification system for a bird feeder scene.  
The system is designed to classify images into three categories:

- **bird**
- **squirrel**
- **empty**

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
- **bird** and **squirrel** perform better than **empty**.
- The **empty** class is currently the weakest category.
- The model still shows limited generalization on new test data.

---
