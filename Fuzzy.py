import csv
import operator


def min_E(avg):
    membership = {"l": 0, "m": 0, "h": 0}
    if avg < 3:
        membership["l"] = 1
    elif avg < 5:
        membership["l"] = 0.5 * (5 - avg)

    if 5 >= avg > 3:
        membership["m"] = 0.5 * (avg - 3)
    elif 7 >= avg > 5:
        membership["m"] = 0.5 * (7 - avg)

    if avg > 7:
        membership["h"] = 1
    elif 5 <= avg <= 7:
        membership["h"] = 0.5 * (avg - 5)

    return membership


def p_e(pe):
    membership = {"l": 0, "m": 0, "h": 0}
    if pe < 0.1:
        membership["l"] = 1
    elif pe < 0.3:
        membership["l"] = 5 * (0.3 - pe)

    if pe > 0.5:
        membership["h"] = 1
    elif pe > 0.3:
        membership["h"] = 5 * (pe - 0.3)

    if 0.1 <= pe < 0.3:
        membership["m"] = 5 * (pe - 0.1)
    elif 0.3 <= pe <= 0.5:
        membership["m"] = 5 * (0.5 - pe)

    return membership


def len_E(std):
    membership = {"l": 0, "m": 0, "h": 0}
    if std <= 1:
        membership["l"] = 1
    elif std <= 3:
        membership["l"] = 0.5 * (3 - std)

    if std >= 6:
        membership["h"] = 1
    elif std >= 4:
        membership["h"] = 0.5 * (std - 4)

    if 1 <= std <= 3:
        membership["m"] = 0.5 * (std - 1)
    elif 3 <= std <= 4:
        membership["m"] = 1
    elif 4 <= std <= 6:
        membership["m"] = 0.5 * (6 - std)

    return membership


def get_value(str1):
    if str1 == "l":
        out = -1
    elif str1 == "m":
        out = 0
    else:
        out = 1
    return out


def rule(avg, std, pe):
    out = get_value(avg) - get_value(std) + get_value(pe)
    if out == -3 or out == -2:
        temp = "vl"
    elif out == -1:
        temp = "l"
    elif out == 0:
        temp = "m"
    elif out == 1:
        temp = "h"
    else:
        temp = "vh"
    return temp


def get_output(avg, std, pe):
    temp_avg = min_E(avg)
    temp_len = len_E(std)
    temp_pe = p_e(pe)
    temp = dict()
    for key_avg, value_avg in temp_avg.items():
        for key_std, value_std in temp_len.items():
            for key_pe, value_pe in temp_pe.items():
                out = rule(key_avg, key_std, key_pe)
                out_value = min(value_avg, value_std, value_pe)
                temp[out] = max(temp.get(out, -1), out_value)
    r = max(temp.items(), key=operator.itemgetter(1))[0]
    if r == "vl":
        output = -0.1
    elif r == "l":
        output = 0.0
    elif r == "m":
        output = 0.1
    elif r == "h":
        output = 0.2
    elif r == "vh":
        output = 0.3
    return output


# f = open("log/energy_info.csv", "r")
# g = open("log/fuzzy_number.csv", "w")
# reader = csv.DictReader(f)
# writer = csv.DictWriter(g, fieldnames=["avg", "len", "p/e", "output"])
# for row in reader:
#     print(row)
#     if row["min E"] == "":
#         continue
#     avg = float(row["min E"])
#     std = float(row["len E"])
#     pe = float(row["p/e"])
#     output1 = get_output(avg, std, pe)
#     writer.writerow({"avg": min_E(avg), "len": len_E(std), "p/e": p_e(pe), "output": output1})
# f.close()
# g.close()
