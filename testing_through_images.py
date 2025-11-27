import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# --- 1️⃣ Modeli yükle ---
MODEL_PATH = 'C:\\Users\\evind\\OneDrive\\Masaüstü\\traffic-sign\\TrafficSignModel_best.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# --- 2️⃣ Sınıf isimleri (GTSRB) ---
class_names = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 tons"
}

# --- 3️⃣ Test etmek istediğin görselin yolu ---
img_path = 'C:\\Users\\evind\\OneDrive\\Masaüstü\\traffic-sign\\images\\yol_calisma.jpg'

# --- 4️⃣ Görseli yükle ve ön işle ---
img = image.load_img(img_path, target_size=(32, 32))  # Modelin input size'ı
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize et

# --- 5️⃣ Tahmin yap ---
pred = model.predict(img_array)
class_idx = np.argmax(pred)
confidence = pred[0][class_idx] * 100

# --- 6️⃣ Sonucu göster ---
class_name = class_names[class_idx]

plt.imshow(img)
plt.axis('off')
plt.title(f"{class_name} ({confidence:.2f}%)", fontsize=12)
plt.show()

print(f"Tahmin edilen sınıf: {class_name}")
print(f"Modelin güven yüzdesi: %{confidence:.2f}")
