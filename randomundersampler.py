from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import joblib

# 📥 Dosya yolları
original_path = "/Users/elifsakin/Downloads/mental_health_embeddings_pca200.csv"
encoder_path = "/Users/elifsakin/Desktop/proje_final/label_encoder.pkl"
output_path = "/Users/elifsakin/Downloads/mpnet_randomundersampled.csv"

# 🔍 Veriyi oku
df_original = pd.read_csv(original_path)
encoder = joblib.load(encoder_path)

# 🎯 Özellikler ve hedef değişken
X = df_original.drop("label", axis=1)
y = df_original["label"]

# 🔁 RandomUnderSampling işlemi
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# 🧾 Yeni DataFrame oluştur
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled["label"] = y_resampled

# 💾 Yeni dosyayı kaydet
df_resampled.to_csv(output_path, index=False)
print(f"✅ Yeni dengelenmiş dosya başarıyla kaydedildi: {output_path}")
