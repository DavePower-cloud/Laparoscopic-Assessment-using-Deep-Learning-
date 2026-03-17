# 🧠 Laparoscopic Skill Assessment using Deep Learning  

![Python](https://img.shields.io/badge/Python-3.10+-blue)  
![Deep Learning](https://img.shields.io/badge/DeepLearning-3DCNN-orange)  
![Domain](https://img.shields.io/badge/Domain-Surgical%20Simulation-green)  
![Status](https://img.shields.io/badge/Status-Published%20Research-brightgreen)  
![License](https://img.shields.io/badge/License-MIT-lightgrey)  

Automated assessment of simulated laparoscopic surgical performance using **computer vision and deep learning (3DCNN)**.  

--- 

Automated assessment of simulated laparoscopic surgical performance using computer vision and deep learning (3DCNN). \

📌 Overview

This repository implements a deep learning framework for objective assessment of laparoscopic surgical skill using simulation video data.
The system uses a 3D Convolutional Neural Network (3DCNN) to extract spatiotemporal features and classify operator expertise:

- Novice

- Trainee

- Expert

This work demonstrates how AI can:

- reduce reliance on subjective human assessment

- scale evaluation in simulation-based training

- provide consistent, reproducible performance metrics

---

📄 Associated Publications
IEEE EMBC (2024)

Automated Assessment of Simulated Laparoscopic Surgical Performance using 3DCNN

Scientific Reports (Nature Portfolio, 2025)
Automated assessment of simulated laparoscopic surgical skill performance using deep learning

---

🎯 Motivation

Traditional surgical skill assessment:

- relies on expert raters

- is time-intensive

- introduces subjectivity
  
Even structured frameworks (e.g. OSATS) require manual scoring.
This project explores whether deep learning can directly infer skill from video, enabling scalable and objective assessment.

---

🗂 Dataset: LSPD (Laparoscopic Surgical Performance Dataset)
Characteristics

Tasks:

- Bands

- Stack

- Tower

Participants:

- Novice

- Trainee

- Expert

~100 videos expanded to 2244 samples via augmentation

Weak supervision:

- labels applied at video level

⚠️ Dataset is not publicly available due to ethical and privacy constraints.

---

🧪 Methodology 

Preprocessing

- Video trimming (OpenCV)

- Frame resizing (128×128)

- Normalisation

- Data augmentation:

  - Gaussian blur

  - Brightness / contrast

  - Salt & pepper noise

  - Horizontal flipping

---

Model Architecture (3DCNN)

- 4 × Conv3D layers (64 → 512 filters)

- Kernel size: 3×3×3

- Batch Normalisation

- Max Pooling (2×2×2)

- Dropout (0.1–0.5)

- Fully connected layers (1024 → 512)
Outputs:

  - Multi-class: novice / trainee / expert

  - Binary: novice vs expert

---

Training

- 5-fold cross-validation

- Train/test split: 80/20

- Validation split: 20% of training

- Optimiser: Adam

- Loss: Binary cross-entropy

---

📊 Results 

Multi-class Classification 

Skill	          Accuracy \
Stack	          79% \
Tower	          49% \
Bands	          54% 

Performance was highest for the stack task, while classification of the trainee group remained challenging due to heterogeneity.	

Binary Classification (Expert vs Novice) 

Skill	          Accuracy \
Stack	          91% \
Tower	          97% \
Bands	          79% 

Binary classification demonstrated strong discrimination between expert and novice participants, particularly for the tower task.

---


	
📈 Statistical Analysis of Task Difficulty

- Stack time: H = 14.08, p = 0.0087

- Stack score: H = 11.70, p = 0.0028

- Tower score: H = 19.83, p = 0.000049

- Bands time: H = 12.53, p = 0.00189 

Not significant:

- Tower time: H = 1.28, p = 0.52

- Bands score: H = 1.858, p = 0.39

---

👨‍⚕️ Inter-Rater Reliability

- Stack: κ = 1.00

- Tower: κ = 0.76

- Bands: κ = 0.72

---

👨‍⚕️ Human vs Model Performance

Multi-class agreement:

- Stack: κ = 0.40

- Tower: κ = 0.41

- Bands: κ = 0.12
  
Binary agreement:

- Stack: κ = 0.53

- Tower: κ = 0.90

- Bands: κ = -0.18
  
The model achieved higher and more consistent performance, particularly in binary classification.

---

🔑 Key Insights

- Strong discrimination between expert vs novice

- Tower task provides the clearest separation

- Trainee group remains difficult to classify

- Human agreement varies significantly

- AI enables consistent, scalable evaluation

---

🖼 Visual Overview
System Pipeline
Raw Video → Preprocessing → 3DCNN → Feature Extraction → Classification → Metrics
Model vs Human
Human: variable, subjective
Model: consistent, reproducible
Task Difficulty
Tower → Strong
Stack → Moderate
Bands → Weak

---

📊 Example Output
{
"task": "tower",
"prediction": "expert",
"confidence": 0.94
}

---

🧱 Repository Structure
src/
├── data.py
├── model.py
├── train.py
├── evaluate.py
└── utils.py
notebooks/
└── demo_assessment.ipynb
🚀 Quick Start
pip install -r requirements.txt
python src/train.py
python src/evaluate.py

---

💡 Key Contributions

- LSPD dataset for laparoscopic skill assessment

- 3DCNN-based spatiotemporal modelling

- Weakly supervised learning

- Scalable AI-driven evaluation

---

⚠️ Limitations

- Small dataset

- Trainee variability

- No frame-level labels

- Simulation-only data

---

🔮 Future Work

- Attention-based video models

- Multi-modal inputs

- Real-world deployment

- Integration into simulation systems

---

📜 License

MIT License

---

👤 Author
David Power
University College Cork
ASSERT Centre

---

📚 Citation

If you use this work, please cite the associated publications.
