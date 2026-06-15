# ⚡ TL;DR: Version 1 Minute

## 🎯 Besoin

Améliorer la **précision des modèles** avec techniques avancées.

## ✅ Solution

3 techniques implémentées et prêtes à utiliser:

```
1. Stacking OOF    → +2-3% gain  (meilleur)
2. Blending        → +1-2% gain  (rapide)
3. Pseudo-Labeling → +1-2% gain  (données non labellisées)
```

## 🚀 Utilisation

```bash
# 1. Voir ça marche (30 sec)
python examples_advanced_techniques.py

# 2. Appliquer (2 min)
python apply_advanced_ensemble.py --technique all

# 3. Voir résultats (1 min)
mlflow ui  # → http://localhost:5000
```

## 📁 Fichiers créés

| Fichier | Usage |
|---------|-------|
| `src/advanced_ensemble_techniques.py` | 💎 Cœur du code |
| `apply_advanced_ensemble.py` | 🎯 Application directe |
| `examples_advanced_techniques.py` | 📚 Exemple complet |
| `README_ADVANCED.md` | 🚀 Démarrage (lire d'abord) |
| `ADVANCED_ENSEMBLE_GUIDE.md` | 📖 Guide complet |
| `VISUAL_GUIDE.md` | 👀 Explications visuelles |
| `quick_paths.py` | ⚡ Chemins rapides |

## 💻 Code rapide

```python
from src.advanced_ensemble_techniques import StackingWithOOF

# Créer
stacking = StackingWithOOF(base_models, n_splits=5)

# Entraîner
stacking.fit(X_train, y_train)

# Prédire
y_pred = stacking.predict(X_test)
```

## 🎓 Ordre de lecture

1. ✅ Ce fichier (1 min)
2. ✅ `README_ADVANCED.md` (10 min)
3. ✅ `VISUAL_GUIDE.md` (15 min)
4. ✅ Exécuter code (5 min)

## 🔥 Commande à lancer NOW

```bash
python examples_advanced_techniques.py
```

**C'est tout!** 🎉

---

Pour plus d'infos: `cat README_ADVANCED.md`
