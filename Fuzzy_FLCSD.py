def trapezoidal(x, a, b, c, d):
    if x <= a:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1
    elif c < x <= d:
        return (d - x) / (d - c)
    else:
        return 0


def triangular(x, a, b, c):
    if x <= a:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return (c - x) / (c - b)
    else:
        return 0


def residual_energy(re, E_max):
    return {"l": trapezoidal(re, 0, 0, 0.3 * E_max, 0.6 * E_max),
            "m": triangular(re, 0.3 * E_max, 0.6 * E_max, 0.9 * E_max),
            "h": trapezoidal(re, 0.6 * E_max, 0.9 * E_max, E_max, E_max)}


def distance_to_mc(d, d_max):
    return {"n": trapezoidal(d, 0, 0, 0.25 * d_max, 0.35 * d_max),
            "a": trapezoidal(d, 0.25 * d_max, 0.35 * d_max, 0.65 * d_max, 0.75 * d_max),
            "f": trapezoidal(d, 0.65 * d_max, 0.75 * d_max, d_max, d_max)}


def critical_node_density(cn, cn_max):
    return {"f": trapezoidal(cn, 0, 0, 0.25 * cn_max, 0.35 * cn_max),
            "a": trapezoidal(cn, 0.25 * cn_max, 0.35 * cn_max, 0.65 * cn_max, 0.75 * cn_max),
            "m": trapezoidal(cn, 0.65 * cn_max, 0.75 * cn_max, cn_max, cn_max)}


def energy_consumption_rate(ecr, ecr_max):
    return {"l": trapezoidal(ecr, 0, 0, 0.3 * ecr_max, 0.6 * ecr_max),
            "m": triangular(ecr, 0.3 * ecr_max, 0.6 * ecr_max, 0.9 * ecr_max),
            "h": trapezoidal(ecr, 0.6 * ecr_max, 0.9 * ecr_max, ecr_max, ecr_max)}


def out_crisp(fuzzy_set):
    if fuzzy_set == "vl":
        return 0.1
    elif fuzzy_set == "l":
        return 0.35
    elif fuzzy_set == "m":
        return 0.6
    elif fuzzy_set == "h":
        return 0.85
    else:
        return 1


def estimate(re, E_max, d, d_max, cn, cn_max, ecr, ecr_max):
    out_rule = ['M', 'M', 'VH', 'M', 'H', 'VH', 'H', 'VH', 'VH', 'L', 'L', 'M', 'L', 'M', 'H', 'M', 'H', 'VH', 'L', 'L',
                'M', 'L', 'L', 'H', 'L', 'M', 'H', 'L', 'M', 'H', 'M', 'H', 'H', 'M', 'H', 'VH', 'L', 'L', 'H', 'M',
                'M', 'H', 'M', 'M', 'H', 'L', 'L', 'M', 'M', 'M', 'H', 'M', 'M', 'H', 'VL', 'VL', 'L', 'VL', 'L', 'M',
                'L', 'M', 'H', 'VL', 'VL', 'L', 'VL', 'L', 'M', 'L', 'L', 'M', 'VL', 'VL', 'L', 'VL', 'L', 'L', 'L',
                'L', 'M']
    out_rule = [item.lower() for item in out_rule]
    re_fuzzy = residual_energy(re, E_max)
    d_fuzzy = distance_to_mc(d, d_max)
    cn_fuzzy = critical_node_density(cn, cn_max)
    ecr_fuzzy = energy_consumption_rate(ecr, ecr_max)
    out_membership = []
    for re_index in ["l", "m", "h"]:
        for d_index in ["n", "a", "f"]:
            for cn_index in ["f", "a", "m"]:
                for ecr_index in ["l", "m", "h"]:
                    out_membership.append(
                        min(re_fuzzy[re_index], d_fuzzy[d_index], cn_fuzzy[cn_index], ecr_fuzzy[ecr_index]))
    return sum([out_membership[index] * out_crisp(out_rule[index]) for index, _ in enumerate(out_rule)]) / sum(
        [out_membership[index] for index, _ in enumerate(out_rule)])
