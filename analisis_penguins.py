# %% [markdown]
# # Análisis de clasificación supervisada - Palmer Penguins
# Pipeline completo: limpieza de nulos, exploración, balanceo, PCA, Bayes ingenuo,
# LDA, QDA y SVM, con matrices de confusión, accuracy y fronteras de decisión.
# Los gráficos se guardan en `graficos/` y los resultados numéricos en `resultados/`.

# %% Librerías y configuración
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DIR_GRAFICOS = BASE / "graficos"
DIR_RESULTADOS = BASE / "resultados"
DIR_GRAFICOS.mkdir(exist_ok=True)
DIR_RESULTADOS.mkdir(exist_ok=True)

SEMILLA = 42
VARIABLES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
NOMBRES_CORTOS = ["Largo pico", "Prof. pico", "Largo aleta", "Masa corporal"]
OBJETIVO = "species"
COLORES = ["#FF8C00", "#A034F0", "#159090"]  # Adelie, Chinstrap, Gentoo

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

pasos = []


def registrar(texto):
    pasos.append(texto)
    print(f"[Paso {len(pasos)}] {texto}")


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(DIR_GRAFICOS / nombre, bbox_inches="tight")
    plt.close(fig)


# %% Paso 1 - Carga de datos
datos_crudos = pd.read_csv(BASE / "db" / "penguins.csv")
registrar(f"Carga de 'penguins.csv': {datos_crudos.shape[0]} filas x {datos_crudos.shape[1]} columnas.")
print(datos_crudos.head())

# %% Paso 2 - Eliminación de valores nulos
nulos = datos_crudos.isnull().sum()
print("Nulos por columna:\n", nulos)

fig, ax = plt.subplots(figsize=(8, 4))
nulos.plot.bar(ax=ax, color="#C0392B")
ax.set_title("Valores nulos por columna (antes de limpiar)")
ax.set_ylabel("Cantidad de nulos")
for i, v in enumerate(nulos):
    ax.text(i, v + 0.2, str(v), ha="center")
guardar(fig, "01_valores_nulos.png")

datos = datos_crudos.dropna().reset_index(drop=True)
registrar(
    f"Eliminación de nulos: se quitan {len(datos_crudos) - len(datos)} filas "
    f"({nulos.sum()} celdas vacías). Quedan {len(datos)} filas."
)

# %% Paso 3 - Selección de variables
X = datos[VARIABLES].to_numpy()
codificador = LabelEncoder()
y = codificador.fit_transform(datos[OBJETIVO])
CLASES = list(codificador.classes_)
registrar(
    f"Selección de variables: {len(VARIABLES)} predictoras numéricas {VARIABLES}; "
    f"objetivo '{OBJETIVO}' con clases {CLASES}. Se descartan 'id', 'year', 'island' y 'sex'."
)

# %% Paso 4 - Exploración
conteo_original = pd.Series(y).value_counts().sort_index()
print("Distribución de clases:\n", dict(zip(CLASES, conteo_original.tolist())))

# Histogramas por especie
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, var, nombre in zip(axes.ravel(), VARIABLES, NOMBRES_CORTOS):
    for k, clase in enumerate(CLASES):
        ax.hist(datos.loc[y == k, var], bins=20, alpha=0.55, color=COLORES[k], label=clase)
    ax.set_title(nombre)
axes[0, 0].legend()
fig.suptitle("Distribución de cada variable por especie")
guardar(fig, "03_histogramas_por_especie.png")

# Matriz de dispersión
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        for k, clase in enumerate(CLASES):
            if i == j:
                ax.hist(X[y == k, i], bins=15, alpha=0.55, color=COLORES[k])
            else:
                ax.scatter(X[y == k, j], X[y == k, i], s=8, alpha=0.6, color=COLORES[k], label=clase)
        if i == 3:
            ax.set_xlabel(NOMBRES_CORTOS[j])
        if j == 0:
            ax.set_ylabel(NOMBRES_CORTOS[i])
axes[0, 1].legend(fontsize=8)
fig.suptitle("Matriz de dispersión de las variables")
guardar(fig, "04_matriz_dispersion.png")

# Correlación
corr = np.corrcoef(X, rowvar=False)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(4), NOMBRES_CORTOS, rotation=30)
ax.set_yticks(range(4), NOMBRES_CORTOS)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center")
ax.grid(False)
fig.colorbar(im)
ax.set_title("Matriz de correlación")
guardar(fig, "05_correlacion.png")
registrar("Exploración: histogramas, matriz de dispersión y matriz de correlación generados.")

# %% Paso 5 - División entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEMILLA
)
registrar(
    f"División estratificada 70/30: {len(X_train)} filas de entrenamiento y {len(X_test)} de prueba."
)


# %% Paso 6 - Balanceo (sobremuestreo aleatorio, sólo sobre entrenamiento)
def balancear(X_in, y_in, semilla=SEMILLA):
    """Sobremuestreo aleatorio de las clases minoritarias hasta igualar a la mayoritaria."""
    n_max = np.bincount(y_in).max()
    X_out, y_out = [], []
    for k in np.unique(y_in):
        Xk = X_in[y_in == k]
        if len(Xk) < n_max:
            Xk = resample(Xk, replace=True, n_samples=n_max, random_state=semilla)
        X_out.append(Xk)
        y_out.append(np.full(len(Xk), k))
    return np.vstack(X_out), np.concatenate(y_out)


X_train_bal, y_train_bal = balancear(X_train, y_train)
antes = np.bincount(y_train).tolist()
despues = np.bincount(y_train_bal).tolist()

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, conteo, titulo in zip(axes, [antes, despues], ["Antes del balanceo", "Después del balanceo"]):
    ax.bar(CLASES, conteo, color=COLORES)
    for i, v in enumerate(conteo):
        ax.text(i, v + 1, str(v), ha="center")
    ax.set_title(f"{titulo} (entrenamiento)")
axes[0].set_ylabel("Cantidad de pingüinos")
guardar(fig, "02_balanceo_clases.png")
registrar(
    f"Balanceo por sobremuestreo: entrenamiento pasa de {dict(zip(CLASES, antes))} "
    f"a {dict(zip(CLASES, despues))}. El conjunto de prueba NO se modifica."
)

# %% Paso 7 - Estandarización
escalador = StandardScaler().fit(X_train_bal)
X_train_esc = escalador.transform(X_train_bal)
X_test_esc = escalador.transform(X_test)
registrar("Estandarización: media 0 y desvío 1, calculados sólo con el entrenamiento balanceado.")

# %% Paso 8 - PCA
pca = PCA().fit(X_train_esc)
var_exp = pca.explained_variance_ratio_
cargas = pd.DataFrame(pca.components_.T, index=NOMBRES_CORTOS,
                      columns=[f"PC{i + 1}" for i in range(len(VARIABLES))])
print("Varianza explicada:", np.round(var_exp, 4))
print("Cargas:\n", cargas.round(3))

fig, ax = plt.subplots(figsize=(7, 4))
pcs = [f"PC{i + 1}" for i in range(len(var_exp))]
ax.bar(pcs, var_exp * 100, color="#2E86C1", label="Individual")
ax.plot(pcs, np.cumsum(var_exp) * 100, "o-", color="#C0392B", label="Acumulada")
for i, v in enumerate(var_exp):
    ax.text(i, v * 100 + 2, f"{v:.1%}", ha="center")
ax.set_ylabel("% de varianza explicada")
ax.set_title("PCA: varianza explicada por componente")
ax.legend()
guardar(fig, "06_pca_varianza.png")

Z_train = pca.transform(X_train_esc)
Z_test = pca.transform(X_test_esc)
fig, ax = plt.subplots(figsize=(8, 7))
for k, clase in enumerate(CLASES):
    ax.scatter(Z_train[y_train_bal == k, 0], Z_train[y_train_bal == k, 1],
               s=18, alpha=0.6, color=COLORES[k], label=clase)
desfase_texto = [0, 0, 0.18, -0.18]  # largo de aleta y masa corporal casi se superponen
for i, nombre in enumerate(NOMBRES_CORTOS):
    ax.arrow(0, 0, cargas.iloc[i, 0] * 2.5, cargas.iloc[i, 1] * 2.5, color="k",
             width=0.02, head_width=0.1)
    ax.text(cargas.iloc[i, 0] * 2.9, cargas.iloc[i, 1] * 2.9 + desfase_texto[i], nombre,
            fontsize=11, ha="center", fontweight="bold")
ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})")
ax.set_ylabel(f"PC2 ({var_exp[1]:.1%})")
ax.set_title("Biplot PCA: especies y aporte de cada variable")
ax.legend()
guardar(fig, "07_pca_biplot.png")
cargas.round(4).to_csv(DIR_RESULTADOS / "pca_cargas.csv")
registrar(
    f"PCA: PC1 explica {var_exp[0]:.1%} y PC2 {var_exp[1]:.1%} "
    f"(acumulado {var_exp[:2].sum():.1%}). Se usan las 2 primeras para visualizar fronteras."
)

# %% Paso 9 - Entrenamiento de los modelos
MODELOS = {
    "Bayes ingenuo": GaussianNB(),
    "PCA + Bayes ingenuo": make_pipeline(PCA(), GaussianNB()),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "SVM lineal": SVC(kernel="linear", C=1.0),
    "SVM RBF": SVC(kernel="rbf", C=1.0, gamma="scale"),
}

entrenados = {}
predicciones = {}
for nombre, modelo in MODELOS.items():
    m = clone(modelo).fit(X_train_esc, y_train_bal)
    entrenados[nombre] = m
    predicciones[nombre] = m.predict(X_test_esc)
registrar(f"Entrenamiento de {len(MODELOS)} modelos sobre el entrenamiento balanceado y estandarizado.")


# %% Paso 10 - Accuracy (prueba + validación cruzada)
def validacion_cruzada(modelo, X_in, y_in):
    """Validación cruzada estratificada; el balanceo y el escalado se hacen dentro de cada fold."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEMILLA)
    puntajes = []
    for idx_tr, idx_va in cv.split(X_in, y_in):
        Xb, yb = balancear(X_in[idx_tr], y_in[idx_tr])
        pipe = make_pipeline(StandardScaler(), clone(modelo)).fit(Xb, yb)
        puntajes.append(accuracy_score(y_in[idx_va], pipe.predict(X_in[idx_va])))
    return np.array(puntajes)


filas = []
for nombre, modelo in MODELOS.items():
    cv = validacion_cruzada(modelo, X, y)
    filas.append({
        "Modelo": nombre,
        "Accuracy prueba": accuracy_score(y_test, predicciones[nombre]),
        "Accuracy CV (media)": cv.mean(),
        "Accuracy CV (desvío)": cv.std(),
    })
metricas = pd.DataFrame(filas).set_index("Modelo")
print(metricas.round(4))
metricas.round(4).to_csv(DIR_RESULTADOS / "metricas_accuracy.csv")

fig, ax = plt.subplots(figsize=(10, 5))
pos = np.arange(len(metricas))
ax.bar(pos - 0.2, metricas["Accuracy prueba"] * 100, 0.4, label="Prueba (30%)", color="#2E86C1")
ax.bar(pos + 0.2, metricas["Accuracy CV (media)"] * 100, 0.4,
       yerr=metricas["Accuracy CV (desvío)"] * 100, capsize=4,
       label="Validación cruzada 5x3", color="#27AE60")
for p, (a, b, s) in zip(pos, metricas.to_numpy()):
    ax.text(p - 0.2, a * 100 + 0.4, f"{a:.1%}", ha="center", fontsize=8)
    ax.text(p + 0.2, (b + s) * 100 + 0.4, f"{b:.1%}", ha="center", fontsize=8)
ax.set_xticks(pos, metricas.index, rotation=15)
ax.set_ylim(80, 102)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Comparación de accuracy entre modelos")
ax.legend(loc="lower right")
guardar(fig, "09_comparacion_accuracy.png")
registrar("Accuracy calculada sobre el conjunto de prueba y con validación cruzada estratificada 5x3.")


# %% Paso 11 - Matrices de confusión y descripción de errores
def describir_errores(nombre, cm):
    total, aciertos = cm.sum(), np.trace(cm)
    lineas = [f"{nombre}: {aciertos}/{total} aciertos ({aciertos / total:.1%}), {total - aciertos} error(es)."]
    for i, real in enumerate(CLASES):
        for j, pred in enumerate(CLASES):
            if i != j and cm[i, j] > 0:
                lineas.append(f"  - {cm[i, j]} {real} clasificado(s) erróneamente como {pred}.")
    if total == aciertos:
        lineas.append("  - Sin errores en el conjunto de prueba.")
    return "\n".join(lineas)


fig, axes = plt.subplots(2, 3, figsize=(15, 9))
textos = []
for ax, nombre in zip(axes.ravel(), MODELOS):
    cm = confusion_matrix(y_test, predicciones[nombre])
    ConfusionMatrixDisplay(cm, display_labels=CLASES).plot(ax=ax, cmap="Blues", colorbar=False,
                                                           values_format="d")
    ax.set_title(f"{nombre}\naccuracy = {accuracy_score(y_test, predicciones[nombre]):.1%}")
    ax.grid(False)
    textos.append(describir_errores(nombre, cm))
    textos.append(classification_report(y_test, predicciones[nombre], target_names=CLASES, digits=3))
fig.suptitle("Matrices de confusión sobre el conjunto de prueba (filas = real, columnas = predicho)")
guardar(fig, "08_matrices_confusion.png")
(DIR_RESULTADOS / "errores_matriz_confusion.txt").write_text("\n".join(textos), encoding="utf-8")
print("\n".join(textos))
registrar("Matrices de confusión generadas y errores descriptos en resultados/errores_matriz_confusion.txt.")

# %% Paso 12 - Cómo "ve" los datos cada algoritmo
# Bayes ingenuo: una campana de Gauss por variable y por clase
nb = entrenados["Bayes ingenuo"]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for v, (ax, nombre) in enumerate(zip(axes.ravel(), NOMBRES_CORTOS)):
    grilla = np.linspace(X_train_esc[:, v].min() - 0.5, X_train_esc[:, v].max() + 0.5, 300)
    for k, clase in enumerate(CLASES):
        mu, var = nb.theta_[k, v], nb.var_[k, v]
        ax.hist(X_train_esc[y_train_bal == k, v], bins=20, density=True, alpha=0.25, color=COLORES[k])
        ax.plot(grilla, np.exp(-(grilla - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var),
                color=COLORES[k], lw=2, label=clase)
    ax.set_title(f"{nombre} (estandarizada)")
axes[0, 0].legend()
fig.suptitle("Bayes ingenuo: distribución normal estimada por variable y especie")
guardar(fig, "10_bayes_densidades.png")
print("Bayes - probabilidades a priori:", dict(zip(CLASES, nb.class_prior_.round(3).tolist())))

# LDA: proyección a los 2 ejes discriminantes
lda = entrenados["LDA"]
L_train = lda.transform(X_train_esc)
L_test = lda.transform(X_test_esc)
fig, ax = plt.subplots(figsize=(8, 6))
for k, clase in enumerate(CLASES):
    ax.scatter(L_train[y_train_bal == k, 0], L_train[y_train_bal == k, 1], s=15, alpha=0.35,
               color=COLORES[k], label=f"{clase} (entrenamiento)")
    ax.scatter(L_test[y_test == k, 0], L_test[y_test == k, 1], s=45, marker="^",
               edgecolor="k", color=COLORES[k], label=f"{clase} (prueba)")
ax.set_xlabel(f"LD1 ({lda.explained_variance_ratio_[0]:.1%} de la separación)")
ax.set_ylabel(f"LD2 ({lda.explained_variance_ratio_[1]:.1%} de la separación)")
ax.set_title("LDA: proyección sobre los ejes discriminantes")
ax.legend(fontsize=8)
guardar(fig, "11_lda_proyeccion.png")

svm_rbf = entrenados["SVM RBF"]
svm_lin = entrenados["SVM lineal"]
print("SVM lineal - vectores de soporte por clase:", dict(zip(CLASES, svm_lin.n_support_.tolist())))
print("SVM RBF - vectores de soporte por clase:", dict(zip(CLASES, svm_rbf.n_support_.tolist())))
registrar(
    "Interpretación interna: medias/varianzas de Bayes, ejes discriminantes de LDA "
    f"(LD1={lda.explained_variance_ratio_[0]:.1%}) y vectores de soporte de SVM "
    f"(lineal={svm_lin.n_support_.sum()}, RBF={svm_rbf.n_support_.sum()})."
)


# %% Paso 13 - Fronteras de decisión (plano PC1-PC2)
def graficar_frontera(ax, modelo, Z_tr, y_tr, Z_te, y_te, titulo, vectores_soporte=False):
    x_min, x_max = Z_tr[:, 0].min() - 1, Z_tr[:, 0].max() + 1
    y_min, y_max = Z_tr[:, 1].min() - 1, Z_tr[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    zona = modelo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zona, levels=[-0.5, 0.5, 1.5, 2.5], colors=COLORES, alpha=0.2)
    ax.contour(xx, yy, zona, levels=[0.5, 1.5], colors="k", linewidths=1)
    for k, clase in enumerate(CLASES):
        ax.scatter(Z_te[y_te == k, 0], Z_te[y_te == k, 1], s=30, color=COLORES[k],
                   edgecolor="k", label=clase)
    if vectores_soporte:
        sv = modelo.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=110, facecolors="none", edgecolors="k",
                   linewidths=1.2, label="Vector de soporte")
    acc = accuracy_score(y_te, modelo.predict(Z_te))
    ax.set_title(f"{titulo}\naccuracy prueba (2D) = {acc:.1%}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return acc


Z_train2, Z_test2 = Z_train[:, :2], Z_test[:, :2]
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
acc_2d = {}
for ax, (nombre, modelo) in zip(axes.ravel(), MODELOS.items()):
    m2 = clone(modelo).fit(Z_train2, y_train_bal)
    acc_2d[nombre] = graficar_frontera(ax, m2, Z_train2, y_train_bal, Z_test2, y_test, nombre)
axes[0, 0].legend(loc="lower left", fontsize=8)
fig.suptitle("Fronteras de decisión en el plano de las 2 primeras componentes principales "
             "(puntos = conjunto de prueba)")
guardar(fig, "12_fronteras_decision.png")
pd.Series(acc_2d, name="Accuracy prueba 2D").round(4).to_csv(DIR_RESULTADOS / "accuracy_fronteras_2d.csv")
registrar("Fronteras de decisión: cada modelo se reentrena sólo con PC1 y PC2 para poder dibujarlas.")

# %% Paso 14 - Máquina de soporte vectorial
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, kernel in zip(axes, ["linear", "rbf"]):
    svm2 = SVC(kernel=kernel, C=1.0, gamma="scale").fit(Z_train2, y_train_bal)
    graficar_frontera(ax, svm2, Z_train2, y_train_bal, Z_test2, y_test,
                      f"SVM kernel {kernel} - {len(svm2.support_vectors_)} vectores de soporte",
                      vectores_soporte=True)
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle("Máquina de soporte vectorial multiclase (PC1-PC2)")
guardar(fig, "13_svm_multiclase.png")

# Caso binario más difícil (Adelie vs Chinstrap) para visualizar hiperplano y margen.
# Se usan los datos sin sobremuestreo (evita puntos duplicados) y el balanceo se hace con class_weight.
binario = np.isin(y_train, [0, 1])
Xb = escalador.transform(X_train[binario])[:, :2]  # largo y profundidad del pico
yb = y_train[binario]
svm_bin = SVC(kernel="linear", C=1.0, class_weight="balanced").fit(Xb, yb)
fig, ax = plt.subplots(figsize=(8, 7))
xx, yy = np.meshgrid(np.linspace(Xb[:, 0].min() - 0.5, Xb[:, 0].max() + 0.5, 400),
                     np.linspace(Xb[:, 1].min() - 0.5, Xb[:, 1].max() + 0.5, 400))
dec = svm_bin.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, dec > 0, levels=[-0.5, 0.5, 1.5], colors=COLORES[:2], alpha=0.15)
ax.contour(xx, yy, dec, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])
for k in [0, 1]:
    ax.scatter(Xb[yb == k, 0], Xb[yb == k, 1], s=20, color=COLORES[k], alpha=0.7, label=CLASES[k])
ax.scatter(svm_bin.support_vectors_[:, 0], svm_bin.support_vectors_[:, 1], s=120,
           facecolors="none", edgecolors="k", label="Vectores de soporte")
w = svm_bin.coef_[0]
ax.set_title(f"SVM lineal Adelie vs Chinstrap\nhiperplano (línea llena) y margen (punteadas), "
             f"ancho del margen = {2 / np.linalg.norm(w):.2f}")
ax.set_xlabel("Largo del pico (estandarizado)")
ax.set_ylabel("Profundidad del pico (estandarizada)")
ax.legend()
guardar(fig, "14_svm_margen_binario.png")
registrar(
    f"SVM: frontera multiclase (lineal y RBF) y ejemplo binario Adelie vs Chinstrap con "
    f"{len(svm_bin.support_vectors_)} vectores de soporte."
)

# %% Registro final del paso a paso
(DIR_RESULTADOS / "paso_a_paso.txt").write_text(
    "\n".join(f"{i}. {p}" for i, p in enumerate(pasos, 1)), encoding="utf-8"
)
print("\nListo. Gráficos en", DIR_GRAFICOS, "y resultados en", DIR_RESULTADOS)
