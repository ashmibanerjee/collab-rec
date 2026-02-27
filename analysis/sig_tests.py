import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, ttest_ind
from statsmodels.sandbox.stats.multicomp import MultiComparison


def anova_one_way(mami, masi, sasi):
    if min(len(mami), len(masi), len(sasi)) < 2:
        print("ANOVA skipped: each group needs at least 2 samples.")
        return
    # Step 1: One-way ANOVA
    f_stat, p_anova = f_oneway(mami, masi, sasi)
    print("ANOVA F-statistic:", f_stat)
    print("ANOVA p-value:", p_anova)


def get_comparison_data(sasi: np.array,
                        masi: np.array,
                        mami: np.array):
    data = pd.DataFrame({
        'value': list(sasi) + list(masi) + list(mami),
        'group': ['sasi'] * len(sasi) + ['masi'] * len(masi) + ['mami'] * len(mami)
    })
    return data


def multicomparison(sasi, masi, mami):
    if min(len(sasi), len(masi), len(mami)) == 0:
        print("Multi-comparison skipped: at least one group is empty.")
        return None
    sasi = np.array(sasi)
    masi = np.array(masi)
    mami = np.array(mami)
    data = get_comparison_data(sasi, masi, mami)
    mc = MultiComparison(data['value'], data['group'])

    # Perform all pairwise independent t-tests with Bonferroni correction
    result = mc.allpairtest(ttest_ind, method='bonf')

    # Print summary
    print(result[0])
    print("\n")
    return result


def perform_significance_tests(mami: list[float], masi: list[float], sasi: list[float]):
    if min(len(mami), len(masi), len(sasi)) < 2:
        print("Significance tests skipped: each group needs at least 2 samples.")
        return
    # Distribution diagnostics
    def _shapiro_p(x):
        if len(x) < 3 or len(x) > 5000:
            return None
        return stats.shapiro(x).pvalue

    def _cohens_d(a, b):
        a = np.array(a)
        b = np.array(b)
        if len(a) < 2 or len(b) < 2:
            return None
        dof = len(a) + len(b) - 2
        pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / dof)
        if pooled == 0:
            return 0.0
        return (np.mean(a) - np.mean(b)) / pooled

    def _cliffs_delta(a, b):
        a = np.array(a)
        b = np.array(b)
        if len(a) == 0 or len(b) == 0:
            return None
        gt = 0
        lt = 0
        for x in a:
            gt += np.sum(x > b)
            lt += np.sum(x < b)
        return (gt - lt) / (len(a) * len(b))

    shapiro_mami = _shapiro_p(mami)
    shapiro_masi = _shapiro_p(masi)
    shapiro_sasi = _shapiro_p(sasi)
    levene_p = stats.levene(mami, masi, sasi).pvalue
    print(f"Shapiro p-values (mami/masi/sasi): {shapiro_mami}, {shapiro_masi}, {shapiro_sasi}")
    print(f"Levene p-value (equal variances): {levene_p:.4f}")

    # One-way ANOVA (Source [9])
    f_stat, p_anova = stats.f_oneway(mami, masi, sasi)
    print(f"ANOVA p-value: {p_anova:.4f}")

    if p_anova < 0.05:
        # Pairwise tests with Bonferroni correction alpha = 0.017 (Source [5, 10])
        alpha_corrected = 0.017
        pairs = [("MAMI", "MASI", mami, masi),
                 ("MAMI", "SASI", mami, sasi),
                 ("MASI", "SASI", masi, sasi)]

        for name1, name2, grp1, grp2 in pairs:
            t_stat, p_val = stats.ttest_ind(grp1, grp2)
            is_significant = p_val < alpha_corrected
            d = _cohens_d(grp1, grp2)
            delta = _cliffs_delta(grp1, grp2)
            d_str = f"{d:.3f}" if d is not None else "n/a"
            delta_str = f"{delta:.3f}" if delta is not None else "n/a"
            print(f"{name1} vs {name2}: p={p_val:.4f}, Significant: {is_significant}, d={d_str}, delta={delta_str}")


def perform_multicomparison(mami: list[float], masi: list[float], sasi: list[float]):
    return multicomparison(sasi, masi, mami)


def perform_nonparametric_tests(mami: list[float], masi: list[float], sasi: list[float]):
    if min(len(mami), len(masi), len(sasi)) < 2:
        print("Nonparametric tests skipped: each group needs at least 2 samples.")
        return

    h_stat, p_kw = stats.kruskal(mami, masi, sasi)
    print(f"Kruskal-Wallis p-value: {p_kw:.4f}")

    if p_kw < 0.05:
        alpha_corrected = 0.017
        pairs = [("MAMI", "MASI", mami, masi),
                 ("MAMI", "SASI", mami, sasi),
                 ("MASI", "SASI", masi, sasi)]

        for name1, name2, grp1, grp2 in pairs:
            u_stat, p_val = stats.mannwhitneyu(grp1, grp2, alternative="two-sided")
            is_significant = p_val < alpha_corrected
            print(f"{name1} vs {name2}: p={p_val:.4f}, Significant: {is_significant}")


def test(mami: list[float], masi: list[float]):
    if len(mami) != len(masi):
        print(f"Fisher test skipped: length mismatch (mami={len(mami)}, masi={len(masi)})")
        return

    mami_success = 0
    masi_success = 0
    ties = 0
    for m_score, s_score in zip(mami, masi):
        if m_score > s_score:
            mami_success += 1
        elif s_score > m_score:
            masi_success += 1
        else:
            ties += 1

    mami_fail = masi_success
    masi_fail = mami_success

    table = [[mami_success, mami_fail], [masi_success, masi_fail]]
    odds_ratio, p_val = stats.fisher_exact(table, alternative="two-sided")
    print(f"Fisher's Exact Test (MAMI > MASI): p={p_val:.4f}, odds_ratio={odds_ratio:.4f}, ties={ties}")
