import itertools
import pprint
from tqdm import tqdm

from output_tex import output_tex

def calc_2_numbers(num_0, num_1, sign_0):

    if sign_0 is "+":
        ans = int(num_0) + int(num_1)

    elif sign_0 is "-":
        ans = int(num_0) - int(num_1)

    elif sign_0 is "*":
        ans = int(num_0) * int(num_1)

    elif sign_0 is "/":
        if int(num_1) == 0:
            return False

        elif int(num_0) % int(num_1) != 0:
            return False

        else:
            ans = int(num_0) // int(num_1)

    return ans

def calc(eqn):

    num_0, sign_0, num_1, sign_1, num_2 = eqn

    if sign_0 in ["+", "-"]:
        sign_0_type = 0

    else:
        sign_0_type = 1

    if sign_1 in ["+", "-"]:
        sign_1_type = 0

    else:
        sign_1_type = 1

    # ((n0 s0 n1) s1 n2) パターン
    if (sign_0_type == sign_1_type) or ((sign_0_type != sign_1_type) and (sign_1_type == 0)):
        ans_0 = calc_2_numbers(num_0, num_1, sign_0)

        if ans_0 is False:
            return False

        ans = calc_2_numbers(ans_0, num_2, sign_1)

    # (n0 s0 (n1 s1 n2)) パターン
    else:
        ans_1 = calc_2_numbers(num_1, num_2, sign_1)

        if ans_1 is False:
            return False

        ans = calc_2_numbers(num_0, ans_1, sign_0)

    return ans

def trans_tex(eqn):

    num_0, sign_0, num_1, sign_1, num_2 = eqn
    
    if sign_0 is "*":
        sign_0 = r"\times"

    elif sign_0 is "/":
        sign_0 = "\div"

    if sign_1 is "*":
        sign_1 = r"\times"

    elif sign_1 is "/":
        sign_1 = "\div"

    text = "$" + num_0 + " " + sign_0 + " " + num_1 + " " + sign_1 + " " + num_2 + "$"

    return text

def make_dataset():

    # 使用する数字と符号
    numbers = ["0", "1", "2", "3", "4"]
    signs = ["+", "-", "*", "/"]

    # 数式生成（num sign num sign num）
    eqns = list(itertools.product(numbers, signs, numbers, signs, numbers))

    # pprint.pprint(eqns)
    print(len(eqns))

    answers = []
    eqns_by_tex = []

    for eqn in eqns:

        # 答えを出す（答えが整数にならないものは除外）
        ans = calc(eqn)

        if ans is not False:
            answers.append(ans)

            # texに変換
            eqns_by_tex.append(trans_tex(eqn))

    print(len(answers), len(eqns_by_tex))
    # print(eqns_by_tex)

    # 出力
    root_path = "./dataset/"
    for i, (text, ans) in enumerate(zip(tqdm(eqns_by_tex), answers)):
        output_tex(text, root_path + str(i) + "_" + str(ans) + ".png")

    # with open(path_w, mode='w') as f:
    #     f.write('\n'.join(l))    


if __name__ == "__main__":
    make_dataset()