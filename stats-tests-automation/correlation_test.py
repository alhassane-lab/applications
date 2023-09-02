from scipy.stats import chi2_contingency
import numpy as np
import seaborn as sns
from scipy.stats import shapiro
import scipy.stats as st
from matplotlib import pyplot as plt
import researchpy


def shapiro_test(x, alpha=0.05):
    x1, pval1 = shapiro(x)
    print("=" * 100, "\n")
    print("\t\t\t\t\t TEST DE LA NORMALITE (TEST DE SHAPIRO) \n")
    print("=" * 100, "\n")
    print("""
    \t##### \033[1m0. Hypothèse du test\033[0m #####\n
    H0 : \033[1m{0}\033[0m suit une loi normale \n
    H1 : \033[1m{0}\033[0m ne suit pas une loi normale \n

    \t##### \033[1m1. Paramètre du test de Shapiro\033[0m #####\n
    Variable aléatoire étudiée : \033[1m{0}\033[0m\n
    Indice de confiance : \033[1m{1}\033[0m\n
    Taille de l'échantillon : \033[1m{2}\033[0m\n

    \t #### \033[1m2. Résultat du test\033[0m ####\n
    p-value de shapiro : \033[1m{3}\033[0m\n
    coefficient de shapiro : \033[1m{4}\033[0m\n 

    \t #### \033[1m3. Conclusion du test\033[0m ####\n""".format(x.name, alpha, x.shape[0], pval1, x1))
    if pval1 < alpha:
        print("L'hypothèse nulle est rejetée \t ==> \033[1m{}\033[0m ne suit pas une loi normale".format(x.name))
    else:
        print("On ne peut pas rejeter l'hypothèse nulle H0 (\033[1m{}\033[0m suit une loi normale)".format(x.name))
    print()
    print("=" * 100, "\n")


def spearman_test(x, y, alpha=0.05):
    pvalue = st.spearmanr(x, y)[1]
    rs = st.spearmanr(x, y)[0]
    print("=" * 100, "\n")
    print("\t\t\t\t\t  \033[1mTEST D'INDEPENDANCE DE SPEARMAN \033[0m \n")
    print("=" * 100, "\n")
    print("""\t##### \033[1m0. Hypothèse du test\033[0m #####\n
    H0 : Les variables {0} sont indépendantes\n
    H1 : Les variables {0} sont corrélées\n
    \t##### \033[1m1. Paramètre du test\033[0m #####\n
    Variables aléatoires étudiées : \033[1m{0}\033[0m\n
    Indice de confiance : \033[1m{1}\033[0m\n
    Taille de l'échantillon : \033[1m{2}\033[0m\n

    \t #### \033[1m2. Résultat du test\033[0m ####\n
    coefficient de Spearman : \033[1m{3}\033[0m\n
    p-value associée au test de Spearman : \033[1m{4}\033[0m\n
    \t #### \033[1m3. Conclusion du test\033[0m ####\n""".format((x.name, y.name), alpha, x.shape[0], rs, pvalue))

    if abs(rs) < .10:
        qual = 'négligeable (ou nulle)'
    elif abs(rs) < .20:
        qual = 'faible'
    elif abs(rs) < .40:
        qual = 'modérée'
    elif abs(rs) < .60:
        qual = 'plutôt forte'
    elif abs(rs) < .80:
        qual = 'forte'
    else:
        qual = 'très forte'
    print()
    if rs == 0:
        print(" --> On ne peut pas rejeter l'hypothèse nulle H0 (Les variables sont indépendantes)")
    elif rs < 0:
        if pvalue < alpha:
            print(
                """ --> \033[1m{}\033[0m présentent \033[1msignificativement\033[0m une \033[1m{}\033[0m corrélation négative.""".format(
                    (x.name, y.name), qual))
        else:
            print(" --> \033[1m{}\033[0m présentent une corrélation négative \033[1m{} peu significative\033[0m".format(
                (x.name, y.name), qual))
    elif rs > 0:
        if pvalue < alpha:
            print(
                " --> \033[1m{}\033[0m présentent \033[1msignificativement\033[0m une \033[1m{}\033[0m corrélation positive.".format(
                    (x.name, y.name), qual))
        else:
            print(" --> \033[1m{}\033[0m présentent une corrélation {} positive \033[1mpeu significative\033[0m".format(
                (x.name, y.name), qual))


def chi2test(data, x, y, alpha=0.05):
    # Tableau de contingence
    cont = data[[x, y]].pivot_table(index=x, columns=y, aggfunc=len, margins=False)

    # Test du chi-2
    chi2, p, dof, expected = chi2_contingency(cont, alpha)

    print("=" * 100, "\n")
    print("""\t\t\t\t\t \033[1mTEST D'INDEPENDANCE DU CHI-2\033[0m \n""")
    print("=" * 100, "\n")

    f, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cont, annot=True, fmt="d", linewidths=2, linecolor="k", ax=ax[0])
    sns.heatmap(np.int64(expected), annot=True, fmt="d", linewidths=2, ax=ax[1], linecolor="k")
    ax[0].set_title("Fréquence observées")
    ax[1].set_title("Fréquence théoriques")
    plt.show()
    print("""

    \t\t ##### \033[1m0. Hypothèse du test \033[0m ##### \n
    H0 : \033[1m{0}\033[0m et \033[1m{1}\033[0m sont indépendantes \n
    H1 : \033[1m{0}\033[0m et \033[1m{1}\033[0m sont corrélées \n 
    \n\t\t ##### \033[1m1. Paramètre du test\033[0m ##### 

    Variables aléatoires étudiées : \033[1m{0} et {1}\033[0m\n
    Indice de confiance alpha : \033[1m{2}\033[0m \n
    Degré de liberté : \033[1m{3}\033[0m\n
    \n\t\t ##### \033[1m2. Résultat du test du Qui-2 \033[0m ##### \n
    Coefficient du qui-2 : \033[1m{4}\033[0m\n
    p-value calculée : \033[1m{5}\033[0m\n

    \n\t\t ##### \033[1m3. Conclusion du test \033[0m ##### 

    """.format(x, y, alpha, dof, chi2, p))

    if p < alpha:
        print("H0 est rejetée : \033[1m{}\033[0m et \033[1m{}\033[0m sont corrélées significativement".format(x, y))
    else:
        print("H1 est rejetée : \033[1m{}\033[0m et \033[1m{}\033[0m ne sont pas corrélées entre elles".format(x, y))

    # Test de V Cramer
    print("\n\n")
    print("=" * 100, "\n")
    print("""\t\t\t\t\t \033[1mTEST DE SIGNIFICATIVITE DE V CRAMER\033[0m \n""")
    print("=" * 100, "\n")

    crosstab, res = researchpy.crosstab(data[x], data[y], test='chi-square')
    coeff_cramer = res.iloc[2, 1]
    if abs(coeff_cramer) < .10:
        qual = "L'intensité du lien entre les variables est \033[1mquasiement nulle\033[0m"
    elif abs(coeff_cramer) < .20:
        qual = "L'intensité du lien entre les variables est \033[1mfaible\033[0m"
    elif abs(coeff_cramer) < .30:
        qual = "L'intensité du lien entre les variables est \033[1mmoyen\033[0m"
    else:
        qual = "L'intensité du lien entre les variables est \033[1mforte\033[0m"
    print("""Le coefficient de Cramer est de : \033[1m{}\033[0m \n
    {}""".format(coeff_cramer, qual))
    print("\n")
    print("=" * 100, "\n")
    


def pearson_test(x, y, alpha=0.05):
    pvalue = st.pearsonr(x, y)[1]
    rs = st.pearsonr(x, y)[0]
    print("=" * 100, "\n")
    print("\t\t\t\t\t  \033[1mTEST D'INDEPENDANCE DE PEARSON \033[0m \n")
    print("=" * 100, "\n")
    print("""\t##### \033[1m0. Hypothèse du test\033[0m #####\n
    H0 : Les variables {0} sont indépendantes\n
    H1 : Les variables {0} sont corrélées\n
    \t##### \033[1m1. Paramètre du test \033[0m #####\n
    Variables aléatoires étudiées : \033[1m{0}\033[0m\n
    Indice de confiance : \033[1m{1}\033[0m\n
    Taille de l'échantillon : \033[1m{2}\033[0m\n

    \t #### \033[1m2. Résultat du test\033[0m ####\n
    coefficient de Spearman : \033[1m{3}\033[0m\n
    p-value associée au test de Spearman : \033[1m{4}\033[0m\n
    \t #### \033[1m3. Conclusion du test\033[0m ####\n""".format((x.name, y.name), alpha, x.shape[0], rs, pvalue))
    
    if abs(rs) < .10:
        qual = 'négligeable (ou nulle)'
    elif abs(rs) < .20:
        qual = 'faible'
    elif abs(rs) < .40:
        qual = 'modérée'
    elif abs(rs) < .60:
        qual = 'plutôt forte'
    elif abs(rs) < .80:
        qual = 'forte'
    else:
        qual = 'très forte'
    print()
    if rs == 0:
        print(" --> On ne peut pas rejeter l'hypothèse nulle H0 (Les variables sont indépendantes)")
    elif rs < 0:
        if pvalue < alpha:
            print(
                """ --> \033[1m{}\033[0m présentent \033[1msignificativement\033[0m une \033[1m{}\033[0m corrélation négative.""".format(
                    (x.name, y.name), qual))
        else:
            print(" --> \033[1m{}\033[0m présentent une corrélation négative \033[1m{} peu significative\033[0m".format(
                (x.name, y.name), qual))
    elif rs > 0:
        if pvalue < alpha:
            print(
                " --> \033[1m{}\033[0m présentent \033[1msignificativement\033[0m une \033[1m{}\033[0m corrélation positive.".format(
                    (x.name, y.name), qual))
        else:
            print(" --> \033[1m{}\033[0m présentent une corrélation {} positive \033[1mpeu significative\033[0m".format(
                (x.name, y.name), qual))
