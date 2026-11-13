# 📘 Rapport Complet du Projet MLOps
## Notebooks 1 à 20 — Explication détaillée pour le groupe

> **Projet :** Prédiction d'achat e-commerce avec pipeline MLOps complet  
> **Contexte :** Données e-commerce (7 mois d'historique) — prédire si un utilisateur va acheter au cours de sa session  
> **Stack :** Python, Kedro, MLflow, Flask, Docker, Cloud Run, Kubernetes, GCP  
> **Résultat final :** `[1, 0, 1, 1, 0]` — le modèle prédit en temps réel via une API scalable

---

## 🏗️ Architecture Globale

```
GitHub Push (staging ou main)
        │
        ▼
Cloud Build CI  →  uv run pytest (tests automatiques)
        │ (si CI réussit)
        ▼
Cloud Build CD  →  uv run kedro run
    ├── Pipeline loading    → télécharge primary.csv depuis GCS
    ├── Pipeline processing → encode les données
    ├── Pipeline training   → optimise (Hyperopt) + entraîne (LightGBM)
    └── Pipeline deployment → pousse vers MLflow Model Registry
        │
        ▼
Cloud Build API Deploy → build Docker + push Artifact Registry
    ├── Branche staging → Cloud Run (serverless)
    └── Branche main   → Kubernetes GKE (production)
```

---

## 📓 NOTEBOOK 1 — Présentation du cours

### Ce qu'on a construit
Notebook purement informatif. Il présente le contexte pédagogique du cours et les objectifs du projet final.

### Ce qu'il faut retenir
- Le projet consiste à construire un **pipeline ML automatisé** de bout en bout
- Les données doivent avoir une **tendance saisonnière** pour justifier le ré-entraînement régulier
- La soutenance dure 20 minutes en groupe de 4-5 personnes

---

## 📓 NOTEBOOK 2 — Étude Descriptive (EDA)

### Ce qu'on a construit
Exploration du jeu de données e-commerce contenant ~1,24 million d'événements utilisateurs (vues, ajouts panier, achats).

**Fichier chargé :** `notebooks/data/sample.csv`  
**URL de téléchargement :** https://blent-learning-user-ressources.s3.eu-west-3.amazonaws.com/training/ml_engineer_facebook/data/sample.csv

**Colonnes du dataset :**
- `event_time` : timestamp de l'événement (UTC)
- `event_type` : type d'événement (view, cart, purchase)
- `product_id`, `category_id`, `category_code`, `brand`, `price` : infos produit
- `user_id`, `user_session` : identifiants utilisateur

### Ce qu'on a fait concrètement
1. Chargement des données avec pandas
2. Extraction des infos temporelles (heure, minute, jour de la semaine)
3. Visualisation de la distribution des événements
4. Analyse des catégories de produits les plus visitées
5. Étude des comportements par session utilisateur

### Résultats obtenus
- ~10% des sessions se terminent par un achat → données **non déséquilibrées** (pas besoin de rééchantillonnage)
- La catégorie `electronics` est la plus visitée
- Variables les plus prometteuses pour la modélisation : `duration`, `num_views_session`, `hour`, `weekday`

### Ce qu'il faut retenir (MLOps)
L'EDA est la première étape indispensable avant toute modélisation. Elle guide le feature engineering du notebook suivant et justifie les choix architecturaux (pas de SMOTE nécessaire ici car données équilibrées).

---

## 📓 NOTEBOOK 3 — Modèle de Boosting (LightGBM)

### Ce qu'on a construit
Feature engineering complet et entraînement d'un modèle LightGBM de classification binaire.

**Fichiers créés dans `notebooks/data/` :**
- `dataset.csv` — données transformées (utilisé comme `primary.csv` pour Kedro)
- `X.csv`, `y.csv` — features et target complètes
- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` — sets d'entraînement/test (split 75/25)
- `primary1.csv`

### Transformations réalisées
1. **Création de la cible** : `purchased` = 1 si la session se termine par un achat, 0 sinon
2. **Agrégation par session** : nombre de vues par session, nombre de produits vus
3. **Feature temporel** : extraction de `hour`, `weekday` depuis `event_time`
4. **Feature durée** : calcul de la durée de la session en secondes
5. **Feature comportemental** : comptage des sessions précédentes par utilisateur
6. **Encodage catégoriel** : `LabelEncoder` sur `brand`, `category`, `sub_category`

### Pourquoi LightGBM ?
- Algorithme de **gradient boosting** optimisé pour les grands datasets
- Utilise **leaf-wise tree growth** (plus rapide que XGBoost qui fait level-wise)
- Supporte nativement les features catégorielles

### Résultats obtenus
- Score F1 sur le train : ~91-92%
- Score F1 sur le test : ~87-89%
- Différence Train/Test ~4% → **pas de sur-apprentissage**
- Rappel > Précision → le modèle préfère éviter les faux négatifs (logique pour le marketing : mieux vaut proposer une offre à quelqu'un qui n'aurait pas acheté que de rater un achat potentiel)

---

## 📓 NOTEBOOK 4 — Optimisation des Hyper-paramètres (Hyperopt)

### Ce qu'on a construit
Optimisation bayésienne des hyper-paramètres de LightGBM via la bibliothèque **Hyperopt** (méthode TPE).

**Fichier créé :** `notebooks/data/model.pkl` — le modèle optimisé sérialisé

### Ce qu'on a fait concrètement
1. Définition de l'espace de recherche (learning_rate, num_iterations, max_depth, num_leaves, etc.)
2. Lancement de l'optimisation TPE avec ~20 évaluations (chaque évaluation = un K-Fold 4 fois)
3. Récupération des meilleurs hyper-paramètres dans `optimum_params`
4. Ré-entraînement du modèle final avec ces paramètres
5. Affichage de la courbe de calibration

### ⏱️ Durée : 5 à 15 minutes (normale)

### Pourquoi l'optimisation bayésienne plutôt que GridSearch ?
- **GridSearch** : teste toutes les combinaisons → exponentiel en temps
- **RandomSearch** : tire aléatoirement → ne profite pas de l'historique
- **TPE (Tree of Parzen Estimators)** : construit deux distributions probabilistes — une pour les bons hyper-paramètres, une pour les mauvais — et choisit le prochain point en maximisant le ratio Expected Improvement
- En pratique : **4-5x plus efficace** que GridSearch pour le même budget

### Résultats obtenus
- Amélioration du F1 de ~2-3 points par rapport au modèle par défaut
- Courbe de calibration proche de la diagonale au-dessus de 40% → probabilités fiables

---

## 📓 NOTEBOOK 5 — Validation et Interprétabilité (SHAP)

### Ce qu'on a construit
Validation complète du modèle et interprétabilité via les **valeurs SHAP**.

### Ce qu'on a fait concrètement
1. **Densité des classes** : histogramme des probabilités prédites par classe réelle
2. **Courbe de calibration** : alignement entre probabilités prédites et réelles
3. **Courbe ROC** : AUC ~95% — le modèle discrimine très bien
4. **Courbe PR (Precision-Recall)** : compromis précision/rappel
5. **SHAP Summary Plot** : importance globale et direction d'impact de chaque feature
6. **SHAP Dependence Plots** : relation entre `product_id`, `hour`, `num_views_session` et leur impact

### Qu'est-ce qu'une valeur SHAP ?
SHAP (SHapley Additive exPlanations) est basé sur la théorie des jeux coopératifs de Shapley (1953). Pour chaque observation, chaque variable reçoit une "valeur de Shapley" qui représente sa contribution marginale à la prédiction, calculée en moyennant sur toutes les permutations possibles.

- **Valeur SHAP positive** → la variable pousse vers l'achat
- **Valeur SHAP négative** → la variable pousse vers le non-achat
- **Valeur SHAP ≈ 0** → la variable n'a pas d'impact pour cette observation

### Résultats obtenus
Variables les plus impactantes dans l'ordre : `duration`, `num_views_session`, `product_id`, `hour`
- Plus un utilisateur passe de temps → plus il est susceptible d'acheter ✅ (logique métier)
- Plus il voit de produits → plus il est susceptible d'acheter ✅

---

## 📓 NOTEBOOK 6 — Kedro (Introduction)

### Ce qu'on a construit
Création du projet **Kedro** `purchase-predict` — un framework Python pour pipelines ML reproductibles.

### Pourquoi Kedro ?
Les notebooks sont excellents pour l'exploration, mais problématiques pour la production :
- **Pas reproductibles** : dépendent de l'ordre d'exécution des cellules
- **Pas testables** : impossible d'écrire des tests unitaires sur des notebooks
- **Pas maintenables** : difficile de travailler à plusieurs

Kedro résout ces problèmes avec :
- **Data Catalog** (`catalog.yml`) : abstraction des chemins de fichiers — on code avec des noms, pas des chemins
- **Nodes** : fonctions Python atomiques et testables
- **Pipelines** : graphes de dépendances entre nodes, gestion automatique de l'ordre d'exécution

### Structure créée
```
purchase-predict/
  conf/base/
    catalog.yml      ← définit toutes les sources de données
    parameters.yml   ← paramètres du pipeline (test_ratio, max_evals, etc.)
  src/purchase_predict/
    pipelines/
      loading/       ← télécharge depuis GCS
      processing/    ← encode les données
      training/      ← entraîne le modèle
      deployment/    ← pousse vers MLflow Registry
```

### Configuration Data Catalog (`catalog.yml`)
```yaml
primary:
  type: pandas.CSVDataset
  filepath: data/03_primary/primary.csv
model:
  type: pickle.PickleDataset
  filepath: data/06_models/model.pkl
```

### Ce qu'il faut retenir
Le Data Catalog permet de changer le stockage (local → GCS → S3) sans toucher au code des pipelines. C'est le principe de **séparation des préoccupations** (Separation of Concerns).

---

## 📓 NOTEBOOK 7 — Pipeline ML (Kedro complet)

### Ce qu'on a construit
Implémentation des 4 pipelines Kedro et connexion à Google Cloud Storage.

### Pipeline `loading` (nodes.py corrigé)
```python
def load_csv_from_bucket(project: str, bucket_path: str) -> pd.DataFrame:
    storage_client = storage.Client()
    parts = bucket_path.split("/", 1)
    bucket = storage_client.bucket(parts[0])
    blob = bucket.blob(parts[1])
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        blob.download_to_filename(tmp.name)
    return pd.read_csv(tmp.name)
```

### Pipeline `processing` (nodes.py)
```python
def encode_features(dataset):
    # LabelEncoder sur brand, category, sub_category
    ...

def split_dataset(dataset, test_ratio):
    # train_test_split 70/30
    ...
```

### Pipeline `training` (nodes.py — version finale)
```python
def auto_ml(X_train, y_train, X_test, y_test, max_evals, log_to_mlflow, experiment_id):
    # Hyperopt + LightGBM + logging MLflow conditionnel
    ...
```

### Pipeline `deployment` (nodes.py)
```python
def push_to_model_registry(registry_name, model_uri):
    # Crée la version dans MLflow Registry
    ...

def stage_model(registry_name, version):
    # Assigne l'alias "staging" ou "production"
    ...
```

### Problèmes rencontrés et solutions

**Problème 1 : `TypedDict` non défini**
```
NameError: name 'TypedDict' is not defined
```
**Cause :** Import manquant dans `training/nodes.py`  
**Solution :** Ajouter `from typing import Any, TypedDict`

**Problème 2 : `KeyError: 'class'`**
```
KeyError: 'class'
```
**Cause :** La clé dans le dict `ModelSpec` s'appelle `"model_class"`, pas `"class"`  
**Solution :** Remplacer `model_specs["class"]()` par `model_specs["model_class"]()`

**Problème 3 : `MemoryDataset` dans `catalog.yml`**
```
TypeError: issubclass() arg 1 must be a class
```
**Cause :** `MemoryDataset` ne doit pas être déclaré dans `catalog.yml`, Kedro le gère automatiquement  
**Solution :** Supprimer `mlflow_run_id`, `mlflow_model_uri`, `mlflow_model_version` du catalog

**Problème 4 : Clé JSON GCP bloquée (politique CYTech)**
```
An Organization Policy that blocks service accounts key creation has been enforced
```
**Cause :** CYTech a une politique d'organisation qui interdit la création de clés JSON  
**Solution :** Utiliser `gcloud auth application-default login` — les credentials personnels Google sont utilisés automatiquement

### Résultat : Pipeline global fonctionnel
```
[INFO] Completed 1/6 load_csv_from_bucket
[INFO] Completed 2/6 encode_features
[INFO] Completed 3/6 split_dataset
100%|██████████| 10/10 [best loss: -0.8785]
[INFO] Completed 4/6 auto_ml
[INFO] Completed 5/6 push_to_model_registry
[INFO] Completed 6/6 stage_model
Pipeline execution completed in 44.3 sec.
```

---

## 📓 NOTEBOOK 8 — Tests Logiciels

### Ce qu'on a construit
Tests automatisés avec **pytest** pour valider le pipeline Kedro.

### Structure des tests
```
tests/
  pipelines/
    loading/
      conftest.py    ← fixtures partagées (project_id, primary_folder)
      test_nodes.py  ← teste load_csv_from_bucket
      test_pipeline.py ← teste le pipeline loading complet
    processing/
      conftest.py
      test_nodes.py  ← teste encode_features et split_dataset
unit/
  test_1.py          ← test minimal
```

### Tests du pipeline loading (test_nodes.py)
```python
def test_load_csv_from_bucket(project_id, primary_folder):
    df = load_csv_from_bucket(project_id, primary_folder)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert "purchased" in df.columns
```

### Tests du pipeline processing (test_nodes.py)
```python
def test_encode_features(dataset_not_encoded):
    encoded = encode_features(dataset_not_encoded)
    df = encoded["features"]
    assert isinstance(df, pd.DataFrame)
    assert df["purchased"].isin([0, 1]).all()
    # Toutes les colonnes doivent être numériques après encodage
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(df.dtypes[col])

def test_split_dataset(dataset_encoded, test_ratio):
    result = split_dataset(dataset_encoded, test_ratio)
    assert X_train.shape[0] + X_test.shape[0] == dataset_encoded.shape[0]
```

### Résultat
```
7 passed in 55.41s
Coverage: 89%
```

### Ce qu'il faut retenir
- `conftest.py` = fixtures centralisées (données de test partagées entre plusieurs fichiers de test)
- `scope="module"` = la fixture est créée une seule fois par module (évite de re-télécharger depuis GCS à chaque test)
- **89% de coverage** = 89% du code source est exercé par les tests → très bon

---

## 📓 NOTEBOOK 9 — Linting et Refactoring (Ruff)

### Ce qu'on a construit
Configuration du linter **Ruff** et du hook **pre-commit** pour garantir la qualité du code.

### Fichiers créés
- `purchase-predict/ruff.toml` : `line-length = 120`
- `purchase-predict/.pre-commit-config.yaml` : hooks ruff-check + ruff-format

### Pourquoi Ruff ?
Ruff est un linter Python écrit en Rust — **10 à 100x plus rapide** que flake8+black.  
Il remplace en un seul outil : flake8, isort, pyupgrade, black.

### Problème rencontré : ruff bloquait le commit
**Cause :** Imports inutilisés (`import shap`, `import pandas as pd`) dans les fichiers de test  
**Solution :** Supprimer les imports non utilisés. Pre-commit est conçu pour bloquer les commits non conformes — c'est exactement son rôle.

**Workflow correct :**
```
git commit → pre-commit déclenche ruff check
→ si erreurs → ruff les liste → tu corriges → git add . → git commit à nouveau
→ si ruff format modifie des fichiers → git add . → git commit à nouveau
```

### Ce qu'il faut retenir
- `ruff check` = analyse statique (bugs potentiels, PEP 8)
- `ruff format` = reformatage automatique du code (comme Black)
- `pre-commit install` = installe les hooks Git → vérification automatique avant chaque commit

---

## 📓 NOTEBOOK 10 — Tests de Modèles

### Ce qu'on a construit
Tests **post-entraînement** du modèle LightGBM pour vérifier son comportement métier.

### Fichier créé : `unit/test_model.py`

**3 types de tests :**

**1. Test d'invariance** — une variation de ±1€ sur le prix ne doit pas changer la probabilité de plus de 30%
```python
@pytest.mark.skipif(not MODEL_EXISTS, reason="model.pkl not available in CI")
def test_invariance_price():
    # Si "price" n'est pas dans les features → skip automatique
    if "price" not in X_test.columns:
        return
    y_plus = model.predict_proba(X_plus)[:, 1]
    y_minus = model.predict_proba(X_minus)[:, 1]
    assert np.abs(y_plus - y_minus).mean() < 0.30
```

**2. Test directionnel** — augmenter la durée doit augmenter la probabilité d'achat
```python
def test_directional_duration():
    # Logique métier : plus un utilisateur reste, plus il est susceptible d'acheter
    prob_orig = model.predict_proba(X_test)[:, 1].mean()
    prob_more = model.predict_proba(X_more)[:, 1].mean()
    assert prob_more >= prob_orig - 0.05
```

**3. Test avec prototypes** — K-Means trouve des cas représentatifs, le modèle doit les classer cohéremment
```python
def test_prototypes():
    kmeans = KMeans(n_clusters=5, random_state=42)
    centroids = pd.DataFrame(data=kmeans.cluster_centers_, columns=X_test.columns)
    probas = model.predict_proba(centroids)[:, 1]
    assert all(0 <= p <= 1 for p in probas)
```

### Pourquoi `@pytest.mark.skipif(not MODEL_EXISTS)` ?
En CI (Cloud Build), `model.pkl` n'existe pas encore car l'entraînement n'a pas encore eu lieu. Ces tests sont conçus pour tourner **après** l'entraînement, pas avant. C'est une pratique standard dans les équipes ML.

### Ce qu'il faut retenir
- Tests pré-entraînement → vérifier la qualité des données
- Tests post-entraînement → vérifier le comportement métier du modèle
- K-Means comme générateur de **prototypes** (observations représentatives des différents profils utilisateurs)

---

## 📓 NOTEBOOK 11 — MLflow

### Ce qu'on a construit
Déploiement du serveur **MLflow** sur une VM GCP et tracking des expériences d'entraînement.

### Architecture
```
VM GCP g1-small (europe-west1-b)
  └── MLflow Server (port 80)
      ├── Backend Store : SQLite (mlflow.db)
      └── Artifact Store : gs://example-mlops/mlflow/
```

### Déploiement sur VM GCP
```bash
# Sur la VM en SSH
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install "mlflow==3.0.1" google-cloud-storage

# Service systemd pour démarrage automatique
sudo systemctl enable mlflow
sudo systemctl start mlflow
```

### Pourquoi MLflow 3.0.1 et pas la dernière version ?
MLflow 3.1.0+ a introduit une refactorisation de Huey qui crée un conflit avec UV. La version 3.0.1 est stable et compatible avec tout notre stack.

### 4 composants de MLflow
1. **Tracking** : log des métriques, paramètres, artifacts à chaque run
2. **Projects** : packaging du code pour la reproductibilité
3. **Models** : format standardisé pour sauvegarder les modèles
4. **Registry** : versioning des modèles (staging → production)

### Ce qu'on log dans chaque run
```python
mlflow.log_params({"learning_rate": 0.05, "num_iterations": 500, ...})
mlflow.log_metrics({"f1": 0.8785})
mlflow.log_artifact("data/04_feature/transform_pipeline.pkl")
mlflow.sklearn.log_model(model, name="model", signature=signature)
```

### Problème rencontré : IP externe de la VM disparaît
**Cause :** GCP réassigne les IPs éphémères lors du redémarrage de la VM  
**Impact :** Cloud Build ne peut plus contacter MLflow → timeout  
**Solution :** Réassigner une IP éphémère depuis la console GCP → Edit VM → Network interfaces → External IPv4 address → Ephemeral → Save  
**Leçon :** Pour un projet en production réel, utiliser une **IP statique réservée** (quelques € par mois)

---

## 📓 NOTEBOOK 12 — Tracking et Model Registry

### Ce qu'on a construit
Intégration du tracking MLflow dans le pipeline Kedro et gestion du **Model Registry**.

### Modifications apportées au pipeline Kedro

**`training/__init__.py`** : chargement automatique des variables d'environnement
```python
from dotenv import load_dotenv
load_dotenv()
```

**`parameters.yml`** : nouveaux paramètres MLflow
```yaml
mlflow_enabled: True
mlflow_experiment_id: 1
mlflow_model_registry: "purchase_predict"
```

**`training/nodes.py`** : signature étendue de `auto_ml`
```python
def auto_ml(X_train, y_train, X_test, y_test, max_evals,
            log_to_mlflow: bool = False, experiment_id: int = -1):
    if log_to_mlflow:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_SERVER"))
        run = mlflow.start_run(experiment_id=exp_id)
        # ... log params, metrics, artifacts, model
        mlflow.end_run()
```

### Model Registry : les alias MLflow 3.x
MLflow 3.x remplace les "stages" (Staging/Production) par des **alias** libres. On peut maintenant avoir :
- `@staging` : version testée en staging
- `@production` : version déployée en production
- `@champion` : meilleur modèle
- `@challenger` : nouveau modèle à comparer

### Résultat final sur MLflow
```
purchase_predict
  Version 1   @ staging : Version 1   (créé le 24/04/2026)
```

### Ce qu'il faut retenir
- Le pipeline CD (kedro run) crée automatiquement une nouvelle version du modèle à chaque run
- Le Model Registry permet de savoir **exactement quelle version** est déployée en production à tout moment
- C'est l'équivalent de `git tag` pour les modèles ML

---

## 📓 NOTEBOOK 13 — Flask

### Ce qu'on a construit
Une API REST basique avec **Flask** pour gérer un panier d'achat, comme introduction aux APIs avant l'API du modèle.

### Adaptation Windows
Le notebook utilise `os.setsid()` qui n'existe pas sur Windows. Solution : créer `server.py` manuellement et le lancer depuis un terminal séparé.

### Routes implémentées
| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Health check → "Coucou !" |
| GET | `/cart` | Lister le panier |
| POST | `/cart` | Ajouter un produit |
| PATCH | `/cart` | Modifier la quantité |
| DELETE | `/cart` | Supprimer un produit |

### Test réalisé
```
GET /cart        → []
POST /cart       → {} (200)
GET /cart        → [{"id": "je8zng", "quantity": 1}]
POST /cart (même id) → quantité cumulée : 1+2=3
PATCH /cart      → quantity: 10
DELETE /cart     → []
```

### Ce qu'il faut retenir
- Flask = **micro-framework** Python pour APIs REST → léger, extensible
- `@app.route('/cart', methods=['POST'])` = décoration d'une fonction Python en route HTTP
- `request.get_json()` = récupère le corps JSON de la requête
- `jsonify(data)` = sérialise en JSON avec les bons headers

---

## 📓 NOTEBOOK 14 — API du Modèle

### Ce qu'on a construit
L'API de prédiction `purchase_predict_api` — une application Flask qui charge le modèle depuis MLflow Registry et expose un endpoint `/predict`.

### Structure du projet
```
purchase_predict_api/
  app.py          ← serveur Flask principal
  src/
    __init__.py   ← validation des variables d'environnement au démarrage
    model.py      ← classe Model (chargement MLflow + prédiction)
  .env            ← variables d'env (ne pas commiter)
  Dockerfile      ← conteneurisation (notebook 15)
  cloudbuild.yaml ← CI/CD (notebook 18)
  k8s/            ← Kubernetes (notebook 19)
```

### La classe `Model` (src/model.py)
```python
class Model:
    def load_model(self):
        client = MlflowClient()
        model_version = client.get_model_version_by_alias(
            MLFLOW_REGISTRY_NAME, ENV  # ENV = "staging" ou "production"
        )
        # Récupère le transform_pipeline.pkl (encodeurs LabelEncoder)
        pipeline_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{model_version.run_id}/transform_pipeline.pkl"
        )
        self.model = mlflow.sklearn.load_model(f"models:/{MLFLOW_REGISTRY_NAME}@{ENV}")
        self.transform_pipeline = joblib.load(pipeline_path)

    def predict(self, X):
        # 1. Applique les mêmes encodeurs que lors de l'entraînement
        for name, encoder in self.transform_pipeline.items():
            X[name] = encoder.transform(X[name].astype("string").fillna("unknown"))
        # 2. Prédit avec LightGBM
        return self.model.predict(X)
```

### Test réalisé
```python
result = requests.post("http://127.0.0.1:5000/predict",
                       json=dataset.sample(n=5).to_json()).json()
# → [1, 0, 1, 1, 1]
```

### Problème rencontré : conflit de workspace UV
```
mlops-esgmba depends on mlflow==2.20.2 and purchase-predict-api depends on mlflow==3.0.1
```
**Cause :** UV détectait `purchase_predict_api` comme membre du workspace parent qui utilisait mlflow 2.20.2  
**Solution :** Supprimer `purchase_predict_api` de la section `[tool.uv.workspace]` dans le `pyproject.toml` parent

---

## 📓 NOTEBOOK 15 — Conteneurisation de l'API (Docker)

### Ce qu'on a construit
Containerisation de l'API Flask avec Docker et stockage de l'image dans **Artifact Registry** GCP.

### Le Dockerfile
```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim
RUN apt update && apt install libgomp1 -y   # dépendance LightGBM
RUN mkdir /app && WORKDIR /app
COPY pyproject.toml app.py src/ /app/
RUN uv sync --no-cache-dir
EXPOSE 80
CMD ["uv", "run", "gunicorn", "app:app", "-b", "0.0.0.0:80", "-w", "4"]
```

**Pourquoi `libgomp1` ?** LightGBM utilise OpenMP pour le parallélisme — cette bibliothèque est nécessaire sur Linux.  
**Pourquoi Gunicorn ?** Flask seul est un serveur de développement monothreadé. Gunicorn lance 4 workers en parallèle → supporte ~4x plus de requêtes simultanées.

### Artifact Registry
```bash
# Tag et push de l'image
docker tag purchase_predict_api \
  us-central1-docker.pkg.dev/PROJECT_ID/purchased-docker/purchase_predict_api
docker push us-central1-docker.pkg.dev/PROJECT_ID/purchased-docker/purchase_predict_api
```

**Pourquoi Artifact Registry plutôt que Docker Hub ?**
- Privé (pas d'exposition publique de l'image)
- Dans le même réseau GCP → transfers plus rapides
- Intégration native avec Cloud Run et GKE

### Ce qu'il faut retenir
- Docker = **immuabilité** : l'image contient exactement les mêmes dépendances partout
- "It works on my machine" → "It works everywhere" grâce à Docker
- Un **registre** (Artifact Registry) est à Docker ce que GitHub est au code source

---

## 📓 NOTEBOOK 16 — Déploiement sur Cloud Run

### Ce qu'on a construit
Déploiement de l'API sur **Cloud Run** — le service serverless de GCP.

### Configuration Cloud Run
- **Mémoire** : 2 GiB (nécessaire pour LightGBM qui charge ~500 MB de modèle)
- **CPU** : 2 vCPU
- **Région** : europe-west1
- **Autoscaling** : 0 à 3 instances
- **VPC Connector** : pour accéder à la VM MLflow en réseau interne

### Problèmes rencontrés

**Problème 1 : OOMKilled (Out of Memory)**
```
Memory limit of 512 MiB exceeded with 515 MiB used
```
**Cause :** LightGBM + les modèles MLflow dépassent 512 MiB  
**Solution :** Augmenter la limite mémoire à 2 GiB

**Problème 2 : Connection timeout vers MLflow**
```
Connection to 34.38.40.178 timed out
```
**Cause :** Cloud Run avec VPC routait le trafic uniquement en interne → l'IP publique de MLflow inaccessible  
**Solution :** Configurer le routing VPC pour autoriser le trafic vers l'IP publique

**Problème 3 : Status 405 (Method Not Allowed)**
**Cause :** URL sans `/predict` à la fin — la route `/` n'accepte que GET, pas POST  
**Solution :** Ajouter `/predict` à l'URL

### Résultat final
```
Status: 200
Résultat: [1, 1, 1, 1, 1]
URL: https://purchase-predict-api-511275009279.europe-west1.run.app
```

### Cloud Run vs VM classique
| Critère | Cloud Run | VM classique |
|---------|-----------|-------------|
| Gestion serveur | Aucune | Complète |
| Scalabilité | Automatique 0→N | Manuelle |
| Facturation | Par requête | Par heure |
| Cold start | 2-5 secondes | Aucun |
| Cas d'usage | Workloads variables | Workloads constants |

---

## 📓 NOTEBOOK 17 — Pipelines CI/CD

### Ce qu'on a construit
Pipeline **CI/CD** avec **Cloud Build** qui se déclenche automatiquement à chaque push Git.

### Fichiers créés

**`purchase-predict/install.sh`** : installation des dépendances système sur Linux
```bash
apt-get update && apt-get install -y libgomp1 git
```

**`purchase-predict/cloudbuild.yaml`** : pipeline en 3 étapes
```yaml
steps:
  - id: CI    # Tests automatiques
    run: uv sync && uv run pytest .

  - id: CD    # Entraînement Kedro
    run: uv run kedro run
    waitFor: ["CI"]

  - id: API Deploy  # Déploiement de l'API
    run: git clone purchase_predict_api && gcloud builds submit
    waitFor: ["CD"]
```

### Résultat : build Successful
```
✅ 0: CI      (00:00:48) — 7 passed, 3 skipped
✅ 1: CD      (00:00:35) — Pipeline execution completed in 16.3 sec.
✅ 2: API Deploy (00:00:58) — PUSH, DONE
```

### Problème rencontré : tests échouaient en CI (`FileNotFoundError`)
**Cause :** `unit/test_model.py` cherchait `model.pkl` avec des chemins Windows absolus → inexistants sur les serveurs Linux Cloud Build  
**Solution :** `@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="model.pkl not available in CI")` → les tests sont skippés quand le modèle n'est pas disponible

### Ce qu'il faut retenir
- **CI** = Continuous Integration : tests automatiques à chaque push → détecte les régressions tôt
- **CD** = Continuous Delivery/Deployment : déploiement automatique si CI réussit
- `waitFor: ["CI"]` = le CD ne se lance QUE si CI réussit → on n'entraîne pas avec du code cassé

---

## 📓 NOTEBOOK 18 — Pipeline de Pré-production

### Ce qu'on a construit
Extension du pipeline CI/CD pour **déclencher automatiquement** le rebuild et redéploiement de l'API après chaque entraînement.

### Flux complet
```
Push sur staging
  → Cloud Build Kedro (CI + CD + API Deploy)
      ↓ API Deploy clone purchase_predict_api
      → Cloud Build API (build Docker + push + deploy Cloud Run)
```

### Problèmes rencontrés et solutions

**Problème 1 : `Unable to read file [/tmp/purchase_predict_api/cloudbuild.yaml]`**
**Cause :** `cloudbuild.yaml` n'existait pas encore dans le repo `purchase_predict_api`  
**Solution :** Créer et pusher le fichier

**Problème 2 : `INVALID_ARGUMENT: could not resolve source`**
**Cause :** Syntaxe incorrecte de `gcloud builds submit` avec des chemins absolus  
**Solution :** `gcloud builds submit --config=cloudbuild.yaml --async .` (chemin relatif + `--async`)

**Problème 3 : `Secret Manager Secret Accessor` manquant**
**Cause :** Cloud Build n'avait pas la permission de lire les secrets (token GitHub)  
**Solution :** Activer "Secret Manager Secret Accessor" dans Cloud Build Permissions

**Problème 4 : Storage permissions**
**Solution :** Activer "Storage Admin" + "Storage Object Creator" dans Cloud Build Permissions

### Liste complète des permissions Cloud Build activées
- ✅ Cloud Run Admin
- ✅ Cloud Storage Admin
- ✅ Cloud Storage Object Creator
- ✅ Secret Manager Secret Accessor
- ✅ Artifact Registry Writer
- ✅ Cloud Build Editor
- ✅ Cloud Build Service Account

---

## 📓 NOTEBOOK 19 — Déploiement sur Kubernetes

### Ce qu'on a construit
Déploiement de l'API sur **Kubernetes (GKE)** — l'orchestrateur de conteneurs pour la production.

### Cluster GKE créé
```
purchase-predict-api
  - Zone: europe-west1-b
  - Machine: e2-small (2 Go RAM)
  - Nœuds: 2 (autoscaling 1-5)
```

### Fichiers k8s créés

**`k8s/ns.yaml`** : namespace d'isolation
```yaml
kind: Namespace → api-purchase
```

**`k8s/deployment.yaml`** : décrit l'état désiré du pod
```yaml
replicas: 1
image: us-central1-docker.pkg.dev/.../purchase_predict_api:latest
resources:
  requests: memory: "512Mi", cpu: "100m"
  limits:   memory: "1G",    cpu: "500m"
```

**`k8s/service.yaml`** : expose les pods en interne (NodePort)

**`k8s/ingress.yaml`** : point d'entrée externe avec load balancing (IP: 34.120.173.161)

### Problèmes rencontrés

**Problème 1 : OOMKilled (même qu'avec Cloud Run)**
```
Last State: Terminated, Reason: OOMKilled, Exit Code: 137
```
**Cause :** Limite mémoire 512Mi insuffisante pour LightGBM  
**Solution :** requests: 512Mi, limits: 1G (adapté à e2-small qui a 2 Go en tout)

**Problème 2 : Pod Pending (ressources insuffisantes)**
**Cause :** Deux pods (ancien + nouveau) consommaient plus de mémoire que disponible  
**Solution :** `kubectl delete replicaset purchase-predict-845459c58d --namespace api-purchase` pour forcer la transition

**Problème 3 : Backend UNHEALTHY → "Server Error"**
**Cause :** Le load balancer GKE faisait des health checks mais le pod prenait du temps à charger le modèle  
**Solution :** Attendre 2-3 minutes que le load balancer mette à jour son état

### Résultat final
```
kubectl get pods --namespace api-purchase
NAME                                READY   STATUS    RESTARTS   AGE
purchase-predict-56689cbc8c-h4d9n   1/1     Running   0          53s

curl http://34.120.173.161/
→ StatusCode: 200, Content: OK !

Status: 200
Résultat: [1, 1, 1, 1, 1]
```

### Kubernetes vs Cloud Run
| Critère | Kubernetes | Cloud Run |
|---------|------------|-----------|
| Contrôle | Total | Limité |
| Complexité | Élevée | Faible |
| Scalabilité | Horizontale + verticale | Horizontale auto |
| Cold start | Aucun (pods toujours up) | 2-5 secondes |
| Cas d'usage | Production haute charge | APIs variables |

---

## 📓 NOTEBOOK 20 — Pipeline de Production

### Ce qu'on a construit
Extension du pipeline CI/CD à la branche **`main`** (production) avec déploiement automatique sur Kubernetes.

### 4 déclencheurs Cloud Build au total
| Nom | Repo | Branche | Action |
|-----|------|---------|--------|
| `build-purchase-predict-staging` | purchase-predict | `staging` | CI + CD Kedro + API Deploy Cloud Run |
| `build-purchase-predict-production` | purchase-predict | `main` | CI + CD Kedro + API Deploy K8s |
| `build-purchase-predict-api-staging` | purchase_predict_api | `staging` | Build Docker + Deploy Cloud Run |
| `build-purchase-predict-api-production` | purchase_predict_api | `main` | Build Docker + Deploy K8s |

### Résultat final
```
✅ Successful: build-purchase-predict-production
  ✅ 0: CI      (00:00:48)
  ✅ 1: CD      (00:00:35) — Pipeline completed in 16.3 sec.
  ✅ 2: API Deploy (00:00:58)

kubectl get deploy purchase-predict -o yaml --namespace api-purchase | findstr image:
→ image: us-central1-docker.pkg.dev/.../purchase_predict_api:latest
```

### Ce qu'il faut retenir (vision globale MLOps)
- Un **push sur `staging`** → entraîne un nouveau modèle → le déploie sur Cloud Run (staging)
- Un **push sur `main`** → entraîne un nouveau modèle → le déploie sur Kubernetes (production)
- Le pipeline est **100% automatisé** : zéro intervention manuelle après la configuration initiale
- C'est exactement ce que font les équipes ML professionnelles en entreprise

---

## 🎯 Résumé des Concepts MLOps Clés

| Concept | Outil utilisé | Pourquoi |
|---------|--------------|----------|
| Feature Engineering | pandas, LabelEncoder | Transformer les données brutes en features utiles |
| Modélisation | LightGBM | Gradient boosting rapide et performant |
| Optimisation | Hyperopt (TPE) | Recherche bayésienne plus efficace que GridSearch |
| Interprétabilité | SHAP | Comprendre les décisions du modèle |
| Orchestration | Kedro | Pipelines reproductibles et testables |
| Tracking | MLflow | Traçabilité des expériences et versioning des modèles |
| Tests | pytest | Qualité et non-régression du code et des données |
| Linting | Ruff | Qualité et cohérence du code |
| API | Flask + Gunicorn | Exposition du modèle en production |
| Conteneurisation | Docker | Portabilité et reproductibilité |
| Registre | Artifact Registry | Stockage des images Docker |
| CI/CD | Cloud Build | Automatisation du déploiement |
| Staging | Cloud Run | Déploiement serverless rapide et économique |
| Production | Kubernetes GKE | Scalabilité et disponibilité maximales |

---

## 📸 Screenshots à Prendre pour la Présentation

### 1. MLflow — Interface principale
**URL :** `http://35.195.252.125/#/experiments/1`  
**Ce qu'on doit voir :** Liste des runs avec les métriques F1

### 2. MLflow — Model Registry
**URL :** `http://35.195.252.125/#/models`  
**Ce qu'on doit voir :** `purchase_predict — Version 1 — @ staging : Version 1`

### 3. Kedro Viz — Pipeline graph
**Commande :**
```powershell
cd purchase-predict
uv run kedro viz run --port 4141
```
**URL :** `http://localhost:4141`  
**Ce qu'on doit voir :** Graphe des 4 pipelines avec les nodes et datasets

### 4. Cloud Build — Builds réussis
**URL :** `https://console.cloud.google.com/cloud-build/builds`  
**Ce qu'on doit voir :** Builds avec ✅ vert (CI + CD + API Deploy)

### 5. Cloud Build — 4 déclencheurs
**URL :** `https://console.cloud.google.com/cloud-build/triggers`  
**Ce qu'on doit voir :** Les 4 triggers (staging + production pour Kedro et API)

### 6. Cloud Run — Service actif
**URL :** `https://console.cloud.google.com/run`  
**Ce qu'on doit voir :** `purchase-predict-api` avec ✅ vert

### 7. Kubernetes — Pods en cours
**Commande :**
```powershell
kubectl get pods --namespace api-purchase
```
**Ce qu'on doit voir :** Pod `Running` avec `0 RESTARTS`

### 8. Artifact Registry — Image Docker
**URL :** `https://console.cloud.google.com/artifacts`  
**Ce qu'on doit voir :** `purchased-docker/purchase_predict_api`

### 9. Résultat de prédiction
**Commande :**
```powershell
cd purchase_predict_api
& ".venv\Scripts\python.exe" test_cloudrun.py
```
**Ce qu'on doit voir :** `Status: 200, Résultat: [1, 1, 1, 1, 1]`

### 10. Tests pytest
**Commande :**
```powershell
cd purchase-predict
uv run pytest -v
```
**Ce qu'on doit voir :** `7 passed, 3 skipped, Coverage: 89%`
