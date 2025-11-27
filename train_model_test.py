import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns 
import matplotlib

# Matplotlib penceresinin hemen kapanmasını önlemek için uygun bir backend ayarlanır.
# Eğer bu satır sorun çıkarırsa, yorum satırı (hashtag) haline getirebilirsiniz.
try:
    matplotlib.use('TkAgg') 
except ImportError:
    pass # TkAgg mevcut değilse, varsayılanı kullanır

# Sabit Değerler (Önceki kodunuzla aynı olmalı)
IMG_WIDTH = 32
IMG_HEIGHT = 32
CHANNELS = 3
NUM_CLASSES = 43
DATA_DIR = '.' 
MODEL_PATH = 'TrafficSignModel_best.keras' 

# ==============================================================================
# EKSİK FONKSİYON 1: TEST VERİSİ YÜKLEME VE ÖN İŞLEME
# ==============================================================================

def load_test_data(data_dir):
    """Test.csv dosyasını okur, görüntüleri yükler, boyutlandırır ve normalleştirir."""
    
    test_csv_path = os.path.join(data_dir, 'Test.csv')
    test_df = pd.read_csv(test_csv_path)
    
    X_test = []
    y_true_labels = test_df['ClassId'].values
    
    for img_path in test_df['Path']:
        full_path = os.path.join(data_dir, img_path)
        img = cv2.imread(full_path)
        if img is not None:
            # Eğitimde uygulanan adımları aynen uygula
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            X_test.append(np.array(img))

    X_test = np.array(X_test)
    X_test = X_test / 255.0 # Normalleştirme
    
    return X_test, y_true_labels

# ==============================================================================
# EKSİK FONKSİYON 2: EĞİTİM GEÇMİŞİNİ YENİDEN ÇİZME
# ==============================================================================

def plot_saved_history(history_file='history.npy'):
    """Kaydedilen geçmiş dosyasından Loss/Accuracy grafiklerini çizer ve kaydeder."""
    if not os.path.exists(history_file):
        print(f"UYARI: Geçmiş dosyası ({history_file}) bulunamadı. Sadece değerlendirmeye devam ediliyor.")
        return

    history = np.load(history_file, allow_pickle='TRUE').item()
    
    plt.figure(figsize=(12, 5))

    # Kayıp (Loss) Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Eğitim Kaybı')
    plt.plot(history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Kayıp Eğrisi (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()

    # Doğruluk (Accuracy) Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.title('Doğruluk Eğrisi (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('history_graph.png') # Grafiği PNG olarak kaydet
    plt.show() # Grafiği açmaya çalış

# ==============================================================================
# 2. KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX) VE RAPOR FONKSİYONU - DÜZELTİLMİŞ
# ==============================================================================

def create_confusion_report(model, X_test, y_true_labels, data_dir):
    
    # 2.1. TAHMİNLERİ ALMA
    y_pred_probs = model.predict(X_test, verbose=0) 
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    
    # 2.2. SINIF İSİMLERİNİ ALMA VE HATA KONTROLÜ
    meta_csv_path = os.path.join(data_dir, 'Meta.csv')
    sorted_class_names = [str(i) for i in range(NUM_CLASSES)] # Varsayılan: Numaralar

    if os.path.exists(meta_csv_path):
        meta_df = pd.read_csv(meta_csv_path)
        
        # Sütun adı kontrolü (SignName yoksa uyarı verilir)
        if 'SignName' in meta_df.columns:
            class_names = meta_df.set_index('ClassId')['SignName'].to_dict()
            sorted_class_names = [class_names[i] for i in range(NUM_CLASSES) if i in class_names]
        else:
            print("\nUYARI: Levha isimleri (SignName) Meta.csv'de bulunamadı.")
            print("Sınıf etiketleri için sadece ClassId numaraları kullanılacaktır.")

    # 2.3. KARMAŞIKLIK MATRİSİNİ HESAPLAMA
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    # Matris görselleştirmesi
    plt.figure(figsize=(18, 15))
    sns.heatmap(
        cm, 
        annot=True,              
        fmt='d',                 
        cmap='Blues',            
        linewidths=.5,           
        linecolor='black',
        xticklabels=sorted_class_names, 
        yticklabels=sorted_class_names 
    )
    plt.title('Karmaşıklık Matrisi (Confusion Matrix)')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png') # Matrisi PNG olarak kaydet
    plt.show() # Matrisi açmaya çalış
    
    # 2.4. SINIFLANDIRMA RAPORUNU GÖSTERME (Konsol Çıktısı)
    print("\n--- SINIFLANDIRMA RAPORU (Classification Report) ---")
    print(classification_report(y_true_labels, y_pred_labels, target_names=sorted_class_names, zero_division=0))


# ==============================================================================
# ANA ÇALIŞMA FONKSİYONU
# ==============================================================================

if __name__ == '__main__':
    # 1. Grafik Geçmişini Göster (ve kaydet)
    plot_saved_history()

    # 2. Model Değerlendirmesi
    if not os.path.exists(MODEL_PATH):
        print(f"\n--- HATA: Model dosyası '{MODEL_PATH}' bulunamadı. Lütfen önce modeli eğitin ve kaydedin. ---")
    else:
        print(f"\nModel yükleniyor: {MODEL_PATH}")
        best_model = tf.keras.models.load_model(MODEL_PATH)
        
        print("Test verisi yükleniyor ve tahminler yapılıyor...")
        X_test, y_true_labels = load_test_data(DATA_DIR)

        # Modelin genel performansını değerlendirme
        y_test_one_hot = to_categorical(y_true_labels, num_classes=NUM_CLASSES)
        test_loss, test_acc = best_model.evaluate(X_test, y_test_one_hot, verbose=0)
        
        print("\n=============================================")
        print(f"| GENEL TEST SONUÇLARI |")
        print(f"| Test Doğruluğu (Accuracy): {test_acc*100:.2f}% |")
        print(f"| Test Kaybı (Loss): {test_loss:.4f}         |")
        print("=============================================\n")

        # 3. Karmaşıklık Matrisini ve Raporunu oluştur
        print("Karmaşıklık Matrisi ve Sınıflandırma Raporu oluşturuluyor...")
        create_confusion_report(best_model, X_test, y_true_labels, DATA_DIR)