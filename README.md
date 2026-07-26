# Scikit_ML

Sammlung von Machine-Learning-Projekten, umgesetzt mit scikit-learn (und teilweise neuronalen Netzen). Jedes Notebook behandelt ein eigenständiges Problem — von Klassifikation bis Fraud Detection — und dient mir als Übungs- und Lernprojekt im Bereich Data Science / ML.

## 📂 Projekte

### 1. Titanic – Survival Prediction
**Datei:** `titanic_disaster.ipynb`

Klassisches Klassifikationsproblem: Vorhersage, ob ein Passagier die Titanic-Katastrophe überlebt hat, basierend auf Merkmalen wie Alter, Ticketklasse, Geschlecht und Familienstand an Bord.

- **Aufgabe:** Binäre Klassifikation
- **Verfahren:** *(hier ergänzen, z. B. Logistic Regression, Random Forest)*
- **Ergebnis:** *(hier Accuracy/F1-Score ergänzen)*

### 2. Give Me Some Credit – Kreditrisiko-Vorhersage
**Datei:** `GiveMeSomeCredit copy 4.ipynb`

Vorhersage, ob eine Person in den nächsten zwei Jahren mit ihrem Kredit in Zahlungsverzug gerät, basierend auf finanziellen Kennzahlen (Kreditauslastung, Einkommen, Anzahl offener Kredite u. a.).

- **Aufgabe:** Binäre Klassifikation, unausgeglichene Klassen
- **Verfahren:** *(hier ergänzen)*
- **Ergebnis:** *(hier AUC/Recall ergänzen)*

### 3. Credit Fraud Detection
**Datei:** `credit_fraud_detection.ipynb`

Erkennung betrügerischer Kreditkartentransaktionen in einem stark unausgeglichenen Datensatz (Betrugsfälle sind extrem selten).

- **Aufgabe:** Binäre Klassifikation, starkes Klassenungleichgewicht
- **Verfahren:** *(hier ergänzen, z. B. SMOTE, class weights)*
- **Ergebnis:** *(hier Precision/Recall/AUC-PR ergänzen)*

### 4. Porto Seguro – Safe Driver Prediction
**Datei:** `porto-seguro-safe-driver-prediction.ipynb`

Vorhersage, ob ein Versicherungsnehmer im nächsten Jahr eine Kfz-Versicherungsschaden-Anfrage stellen wird, basierend auf anonymisierten Fahrer- und Fahrzeugmerkmalen.

- **Aufgabe:** Binäre Klassifikation
- **Verfahren:** *(hier ergänzen)*
- **Ergebnis:** *(hier Gini-Koeffizient/AUC ergänzen)*

### 5. Covertype – Waldbedeckungs-Klassifikation (Neuronales Netz)
**Datei:** `Covertype_forest_NN.ipynb`

Vorhersage des Waldbedeckungstyps (7 Klassen) anhand kartografischer Merkmale wie Höhe, Hangneigung und Bodenart, umgesetzt mit einem neuronalen Netz.

- **Aufgabe:** Multiklassen-Klassifikation
- **Verfahren:** Neuronales Netz *(Framework ergänzen, z. B. TensorFlow/Keras/PyTorch)*
- **Ergebnis:** *(hier Accuracy ergänzen)*

## 🛠️ Verwendete Technologien

- Python 3.x
- scikit-learn
- pandas, numpy
- matplotlib / seaborn (Visualisierung)
- Jupyter Notebook

*(Liste ggf. um TensorFlow/Keras/PyTorch, imbalanced-learn, XGBoost etc. ergänzen, je nachdem was tatsächlich verwendet wird)*

## 🚀 Setup & Ausführung

1. Repository klonen:
   ```bash
   git clone https://github.com/marcel-maurer/Scikit_ML.git
   cd Scikit_ML
   ```

2. Virtuelle Umgebung erstellen und aktivieren:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

4. Jupyter Notebook starten:
   ```bash
   jupyter notebook
   ```

## 📊 Datenquellen

Die verwendeten Datensätze stammen von Kaggle:

- [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
- [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Porto Seguro's Safe Driver Prediction](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)
- [Forest Cover Type Prediction](https://www.kaggle.com/c/forest-cover-type-prediction)

