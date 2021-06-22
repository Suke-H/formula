import itertools
import pprint
from tqdm import tqdm

from output_tex import output_tex

def calc(eqn):

    # num_0, sign_0, num_1 = eqn
    num_0, sign_0, num_1, sign_1, num_2 = eqn

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

    # if isinstance(ans, int):
    #     return ans

    # else:
    #     return False

def trans_tex(eqn):

    # num_0, sign_0, num_1 = eqn
    num_0, sign_0, num_1, sign_1, num_2 = eqn
    
    if sign_0 is "*":
        sign_0 = r"\times"

    elif sign_0 is "/":
        sign_0 = "\div"

    if sign_1 is "*":
        sign_1 = r"\times"

    elif sign_1 is "/":
        sign_1 = "\div"

    # text = "$" + num_0 + " " + sign_0 + " " + num_1 + "$"
    text = "$" + num_0 + " " + sign_0 + " " + num_1 + " " + sign_1 + " " + num_2 + "$"

    return text

def make_dataset():

    # 使用する数字と符号
    numbers = ["0", "1", "2", "3", "4"]
    signs = ["+", "-", "*", "/"]

    # 数式生成（num sign num sign num）
    # eqns = list(itertools.product(numbers, signs, numbers))
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

    print(eqns_by_tex)

    # 出力
    root_path = "./dataset/"
    for i, (text, ans) in enumerate(zip(tqdm(eqns_by_tex), answers)):
        output_tex(text, root_path + str(i) + "_" + str(ans) + ".png")


if __name__ == "__main__":
    make_dataset()