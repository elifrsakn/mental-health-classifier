# 🧠 Mental Health Prediction Classifier with XAI

This project is a multi-class classification model designed to predict individuals' **mental health status** based on free-form text inputs. The model integrates **Explainable AI (XAI)** techniques to provide transparent and interpretable results.

---

## 📂 Dataset & Preprocessing

- **Size:** 53,043 samples, 3 columns  
- **Missing values:** 362 rows handled  
- **Embeddings:** `all-mpnet-base-v2` from SentenceTransformers (768-dimensional vectors)  
- **Dimensionality Reduction:** PCA reduced vectors from 768 → 400 while preserving **96.57%** of information  
- **Class Balance:** Applied `RandomUnderSampler` to equalize class counts (each class has 1201 samples)

---

## 🔮 Modeling

- **Model:** XGBoost (multi-class classifier)  
- **Class imbalance handling:** SMOTE applied only to the minority class (class 2)  
- **Evaluation:** 5-fold cross-validation  
- **Performance:** Balanced results across training and test sets

---

## 🔍 Explainability (XAI)

### ✅ LIME (Local Interpretable Model-agnostic Explanations)
- Provides HTML-based visual explanations for each prediction  
- Highlights most influential words in each input text  
- Enables comparison of correct vs. incorrect predictions to analyze model behavior

---

## 🎯 Workflow – Gradio Interface

The user enters **free text**, which flows through the following pipeline:

1. **MPNet →** Sentence is converted into a 768-dimensional semantic vector  
2. **PCA →** Dimensionality reduced to 400  
3. **RandomUnderSampler →** Class balance applied during training phase  
4. **XGBoost →** Predicts the probabilities for 7 mental health classes  
5. **Output →** Class with the highest probability is shown to the user

---

## 🧠 Target Mental Health Classes

The model predicts among the following **7 categories**:

- Anxiety  
- Bipolar Disorder  
- Depression  
- Personality Disorder  
- Normal  
- Stress  
- Suicidal Tendency

---

## 🖼 PCA 2D image

![PCA 2D image](https://i.ibb.co/TBrpkpM3/Ekran-Resmi-2025-05-04-15-06-25.png)

---

## 🧪 XAI – LIME Output for Index 6

![LIME Output](https://i.ibb.co/84XTbJ7W/Ekran-Resmi-2025-05-04-15-06-35.png)

---

## 🤖 Hugging Face + Gradio Interface

![Gradio UI](https://media-hosting.imagekit.io/92a2c5b14eb74e18/1745789408013.jpeg?Expires=1840968874&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=Ms1-hKOkcncYIunQnp9zpxHe07tvnYAylr-uforn-Gc3GIBLSWE0vUD5Ticg1mSMGotH2JEAVZ~Se8VI7O3zzGq2myVBJhufMmPYwcoap~fQbag9FHH7kfMNTLpEb8whQYcnIaT2Ft8-tgNrksoFOJ7oz71Nv8Sjs7dBv02XX0f5ZlW9ww-3tivWg1PMyWvygUS---S6rMqAL5LLBhjcl9EQ0TFq7hkaXMyxpk-3LnfbzL5WoHzgByf2HQE3z0Gz1qQHltHw9DPv7GlTiHrrTvWNXfKR7mhSwMpWvWZIreKc-QrZ7knXMNXEH3Bu7mQl5iCFfcN1IlqCHgrtfV83Fw__)

---

## 🧩 Project Pipeline – Mental Health Prediction Classifier with XAI

```mermaid
graph TD
A[📝 User Input\nRaw text sentence] --> B[🔡 Text Cleaning\n(lowercasing, removing symbols)]
B --> C[🤖 Sentence Embedding\nMPNet – 768 dimensions]
C --> D[📉 Dimensionality Reduction\nPCA – 768 ➝ 400]
D --> E[⚖️ Class Balancing\nRandom Under Sampler (train only)]
E --> F[🌲 XGBoost Classifier\nMulti-class prediction]
F --> G[🧠 Class Probabilities\n7-Class Softmax Output]
G --> H[🏷️ Final Prediction\nMost probable class shown]
F --> I[🧩 Explainability\nLIME visual explanation]
```

git clone https://github.com/elifsakin/mental-health-classifier-xai.git
cd mental-health-classifier-xai
pip install -r requirements.txt
python app.py  # or streamlit run app.py











