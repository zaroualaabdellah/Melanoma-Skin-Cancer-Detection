import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask import Flask, request, render_template, redirect , jsonify
import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="melanoma_db"
)
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="melanoma_db"
    )

cursor = db.cursor()

# --- Paramètres globaux ---
data_dir = "dataset/"  # Dossier contenant les images 'benign' et 'malignant'
image_size = (128, 128)
batch_size = 32
epochs = 20

# --- Charger les données ---
def load_data(data_dir):
    images, labels = [], []
    for label, class_name in enumerate(["benign", "malignant"]):
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, image_size)
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

print("Chargement des données...")
X, y = load_data(data_dir)
X = X / 255.0  # Normalisation des pixels (0 à 1)

# --- Division des données en ensemble d'entraînement et de validation ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Entraînement : {len(X_train)} images, Validation : {len(X_val)} images")

# --- Création du modèle CNN ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sortie binaire
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Augmentation des données ---
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

# --- Entraînement du modèle ---
print("Entraînement du modèle...")
training_history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs,
    verbose=1
)

# --- Visualisation des courbes ---
def plot_training_curves(history):
    plt.figure(figsize=(12, 4))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perte d\'entraînement')
    plt.plot(history.history['val_loss'], label='Perte de validation')
    plt.title('Courbe de perte')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()

    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Précision d\'entraînement')
    plt.plot(history.history['val_accuracy'], label='Précision de validation')
    plt.title('Courbe de précision')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.legend()

    plt.tight_layout()
    plt.savefig('static/training_curves.png')  # Save the figure

# Call the plotting function after model training
plot_training_curves(training_history)

# --- Évaluation du modèle ---
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Précision sur l'ensemble de validation : {accuracy * 100:.2f}%")

# --- Prédiction sur une nouvelle image ---
def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return "Malignant" if prediction[0][0] > 0.5 else "Benign"

# --- Configuration de Flask ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', history=training_history)

@app.route('/test')
def test_melanoma():
    return render_template('test.html')

@app.route('/patients')
def list_patients():
   # Liste des colonnes spécifiques avec leurs alias
    specific_columns_with_alias = {
        'id': 'Identifiant',
        'name': 'Nom',
        'gender': 'Sexe',
        'date_of_birth': 'Date de naissance',
        'melanoma_test_result': 'Résultat du test de mélanome',
        'upload_date': 'Date d\'importation'
    }

    # Créer une chaîne de caractères contenant les colonnes spécifiques avec alias
    columns_string = ', '.join(f"`{col}` AS `{alias}`" for col, alias in specific_columns_with_alias.items())

    # Exécuter la requête pour récupérer les données des colonnes spécifiques
    cursor.execute(f"SELECT {columns_string} FROM `patients`")
    patients = cursor.fetchall()

    # Utiliser les alias pour afficher les noms de colonnes dans le template
    columns = list(specific_columns_with_alias.values())  # Obtenir uniquement les alias

    return render_template('patients.html', patients=patients, columns=columns)



@app.route('/predict', methods=['POST'])
def upload_file():
    print("********************************************")
    name = request.form.get('name') 
    gender = request.form.get('gender')
    date_naissance = request.form.get('dob')
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Sauvegarder l'image temporairement
    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)

    # Prédire l'image
    result = predict_image(image_path)
    sql = """
    INSERT INTO patients (name, gender, date_of_birth, melanoma_test_result, image_path)
    VALUES (%s, %s, %s, %s, %s)"""
    values = (name, gender, date_naissance, result, 'image_path')
    cursor.execute(sql, values)
    db.commit()
    return render_template('result.html', result=result, name=name, sexe=gender, date_naissance=date_naissance)

@app.route('/api/gender_counts')
def get_gender_counts():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT gender, COUNT(*) FROM patients GROUP BY gender")
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    return jsonify(results)

@app.route('/api/test_result_counts')
def get_test_result_counts():
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT melanoma_test_result, COUNT(*) FROM patients GROUP BY melanoma_test_result")
        results = cursor.fetchall()

        # Format results as a list of dictionaries
        test_results = [{'melanoma_test_result': row[0], 'count': row[1]} for row in results]

        cursor.close()
        connection.close()
        
        return jsonify(test_results)  # Return formatted JSON response

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error message if something goes wrong
    
@app.route('/api/age_distribution')
def get_age_distribution():
    connection = get_db_connection()
    cursor = connection.cursor()

    # Calculate age groups based on the date_of_birth
    query = """
    SELECT 
        CASE 
            WHEN TIMESTAMPDIFF(YEAR, date_of_birth, CURDATE()) BETWEEN 0 AND 17 THEN '0-17'
            WHEN TIMESTAMPDIFF(YEAR, date_of_birth, CURDATE()) BETWEEN 18 AND 29 THEN '18-29'
            WHEN TIMESTAMPDIFF(YEAR, date_of_birth, CURDATE()) BETWEEN 30 AND 44 THEN '30-44'
            WHEN TIMESTAMPDIFF(YEAR, date_of_birth, CURDATE()) BETWEEN 45 AND 59 THEN '45-59'
            ELSE '60+' 
        END AS age_group,
        COUNT(*) AS count
    FROM patients
    WHERE melanoma_test_result = 'Malignant'
    GROUP BY age_group
    """

    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    connection.close()

    # Format the results as a list of dictionaries
    age_distribution = [{'age_group': row[0], 'count': row[1]} for row in results]

    return jsonify(age_distribution)

@app.route('/api/positive_test_counts_by_gender')
def get_positive_test_counts_by_gender():
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("""
            SELECT gender, COUNT(*) 
            FROM patients 
            WHERE melanoma_test_result = 'Malignant' 
            GROUP BY gender
        """)
        results = cursor.fetchall()

        # Format results as a list of dictionaries
        positive_test_counts = [{'gender': row[0], 'count': row[1]} for row in results]

        cursor.close()
        connection.close()
        
        return jsonify(positive_test_counts)  # Return formatted JSON response

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error message if something goes wrong
if __name__ == '__main__':
    app.run(debug=True)
