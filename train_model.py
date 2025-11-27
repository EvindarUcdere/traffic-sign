import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Sabit Değerler
IMG_WIDTH = 32
IMG_HEIGHT = 32
CHANNELS = 3
NUM_CLASSES = 43 # GTSRB'de 43 farklı trafik levhası sınıfı vardır
EPOCHS = 50 # Yüksek bir değer, Erken Durdurma yönetecek
BATCH_SIZE = 64
DATA_DIR = '.' # Kodun, Train ve Test klasörleriyle aynı dizinde çalıştığını varsayar


# 1. VERİ YÜKLEME VE ÖN İŞLEME FONKSİYONU


def load_data(data_dir):
    '''
    GTSRB veri setini yükler ve görüntüleri ön işleme tabi tutar.
    '''
    images = []
    class_ids = []
    
    # Train.csv dosyasının yolunu oluşturma
    csv_file_path = os.path.join(data_dir, 'Train.csv')
    train_df = pd.read_csv(csv_file_path)
    
    # Görüntüleri klasörlerden okuma
    for index, row in train_df.iterrows():
        # Görüntü yolunu doğru şekilde oluşturma
        img_path = os.path.join(data_dir, row['Path'])
        try:
            # Görüntüyü BGR olarak yükle ve RGB'ye çevir (Keras/TensorFlow RGB formatını tercih eder)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Görüntüyü belirlenen boyuta yeniden boyutlandır
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(np.array(img))
                class_ids.append(row['ClassId'])
        except Exception as e:
            print(f"Hata oluştu: {img_path}: {e}")

    # NumPy dizilerine dönüştürme
    X = np.array(images)
    y = np.array(class_ids)
    
    return X, y


# 2. CNN MODELİNİ OLUŞTURMA (DROPOUT ile Güçlendirilmiş)


def create_cnn_model():
    model = Sequential()

    # Evrişim Bloğu 1
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25)) # Aşırı öğrenmeye karşı ilk önlem

    # Evrişim Bloğu 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25)) # Aşırı öğrenmeye karşı ikinci önlem

    # Tam Bağlı (Dense) Katmanlar
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5)) # Yoğun katmanda aşırı öğrenmeye karşı en güçlü önlem

    # Çıkış Katmanı
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    # Modeli derleme
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# 3. VERİ ARTIRMA VE EĞİTİM (Aşırı Öğrenme Kontrolü)


def train_model(model, X_train, y_train, X_val, y_val):
    
    # 3.1. VERİ ARTIRMA (Data Augmentation)
    # Eğitim setini çeşitlendirerek aşırı öğrenmeyi engeller.
    data_gen = ImageDataGenerator(
        rotation_range=10,        # Rastgele 10 dereceye kadar döndür
        zoom_range=0.15,          # Rastgele %15'e kadar yakınlaştır
        width_shift_range=0.1,    # Rastgele yatay kaydırma
        height_shift_range=0.1,   # Rastgele dikey kaydırma
        shear_range=0.15,         # Kesme dönüşümü
        horizontal_flip=False,    # Trafik levhalarında yatay çevirme olmaz!
        fill_mode='nearest'
    )
    
    # Eğitim verisi üzerinde artırmayı başlat
    data_gen.fit(X_train)
    
    # 3.2. GERİ ÇAĞRI FONKSİYONLARI (Callbacks)
    
    # a) Erken Durdurma (Overfitting'i anında durdurur)
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=8,              # val_loss 8 epoch boyunca iyileşmezse durdur
        restore_best_weights=True # En iyi ağırlıkları geri yükle
    )

    # b) Model Kaydetme
    model_checkpoint = ModelCheckpoint(
        'TrafficSignModel_best.keras',
        monitor='val_accuracy',  # Doğrulama doğruluğu en yüksek olanı kaydet
        save_best_only=True
    )

    # 3.3. EĞİTİMİ BAŞLATMA
    print("\n--- Model Eğitimi Başlatılıyor ---")
    history = model.fit(
        data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE), # Artırılmış veri akışını kullan
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, model_checkpoint]
    )
    
    return history


# 4. GRAFİKSEL SONUÇLARI GÖSTERME


def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Kayıp (Loss) Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Kayıp Eğrisi (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()

    # Doğruluk (Accuracy) Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.title('Doğruluk Eğrisi (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.show()


# ANA ÇALIŞMA FONKSİYONU


if __name__ == '__main__':
    # 1. Veriyi Yükle ve Hazırla
    print("1. Veri yükleniyor ve ön işleniyor...")
    X, y = load_data(DATA_DIR)
    
    # Veriyi Eğitme ve Doğrulama setlerine ayırma (%80 Eğitim, %20 Doğrulama)
    X_train, X_val, y_train_labels, y_val_labels = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Normalleştirme (0-255 -> 0-1)
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    
    # 3. One-Hot Encoding
    y_train = to_categorical(y_train_labels, num_classes=NUM_CLASSES)
    y_val = to_categorical(y_val_labels, num_classes=NUM_CLASSES)
    
    # 4. Modeli Oluştur ve Eğit
    print("2. CNN Modeli Oluşturuluyor...")
    model = create_cnn_model()
    
    print(model.summary())
    
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # 5. Sonuçları Görselleştir
    plot_history(history)
    
    # 6. Test Verisi Üzerinde Değerlendirme (Opsiyonel)
    print("\n--- Test Verisi Değerlendirmesi ---")
    
    # Test verisini yükle (Sadece X_test verisini yüklemek yeterli)
    test_csv_path = os.path.join(DATA_DIR, 'Test.csv')
    test_df = pd.read_csv(test_csv_path)
    
    X_test_paths = test_df['Path']
    y_test_labels = test_df['ClassId'].values
    
    X_test = []
    
    for img_path in X_test_paths:
        full_path = os.path.join(DATA_DIR, img_path)
        img = cv2.imread(full_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            X_test.append(np.array(img))

    X_test = np.array(X_test) / 255.0

    # En iyi modeli yükle
    best_model = tf.keras.models.load_model('TrafficSignModel_best.keras')
    
    # Modelin performansı
    test_loss, test_acc = best_model.evaluate(X_test, to_categorical(y_test_labels, num_classes=NUM_CLASSES), verbose=1)
    print(f"\nTest Doğruluğu: {test_acc*100:.2f}%")
    print(f"Test Kaybı: {test_loss:.4f}")