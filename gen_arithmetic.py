import json
import random

random.seed(42)
OUT = '/home/zwb/ai/data/math/arithmetic.jsonl'
N = 50000

QT = {
    'add': ['{a}+{b}等于几？', '{a}加{b}等于多少？', '计算：{a}+{b}', '口算 {a}+{b}。', '请计算 {a}+{b} 的结果。'],
    'sub': ['{a}-{b}等于几？', '{a}减{b}等于多少？', '计算：{a}-{b}', '口算 {a}-{b}。', '请计算 {a}-{b} 的结果。'],
    'mul': ['{a}×{b}等于几？', '{a}乘以{b}等于多少？', '计算：{a}×{b}', '口算 {a}×{b}。', '请计算 {a}×{b} 的结果。'],
    'div': ['{a}÷{b}等于几？', '{a}除以{b}等于多少？', '计算：{a}÷{b}', '口算 {a}÷{b}。', '请计算 {a}÷{b} 的结果。'],
}


def q(kind, a, b):
    return random.choice(QT[kind]).format(a=a, b=b)


def add_steps(a, b):
    """按位分解（仅无进位时调用）"""
    parts = []
    for pos, name in [(2, '个位'), (1, '十位'), (0, '百位')]:
        x, y = int(str(a).zfill(3)[pos]), int(str(b).zfill(3)[pos])
        parts.append(f'{name}{x}+{y}={x + y}')
    return '，'.join(parts) + f'，所以 {a}+{b}={a + b}。'


def no_carry(a, b):
    return all(int(x) + int(y) < 10 for x, y in zip(str(a).zfill(3), str(b).zfill(3)))


def gen_add():
    if random.random() < 0.4:  # 三位数，30%给步骤
        for _ in range(50):
            a, b = random.randint(100, 899), random.randint(100, 899)
            if no_carry(a, b) and random.random() < 0.3:
                return q('add', a, b), add_steps(a, b)
        return q('add', a, b), f'{a}+{b}={a + b}。'
    a = random.randint(2, 99)
    b = random.randint(2, 999 if random.random() < 0.3 else 99)
    return q('add', a, b), f'{a}+{b}={a + b}。'


def gen_sub():
    a = random.randint(20, 999)
    b = random.randint(2, a - 1)
    return q('sub', a, b), f'{a}-{b}={a - b}。'


def gen_mul():
    r = random.random()
    if r < 0.3:
        a, b = random.randint(2, 9), random.randint(2, 9)
    elif r < 0.85:
        a, b = random.randint(10, 99), random.randint(2, 9)
    else:
        a, b = random.randint(100, 999), random.randint(2, 9)
    return q('mul', a, b), f'{a}×{b}={a * b}。'


def gen_div():
    b = random.randint(2, 9)
    c = random.randint(2, 99)
    a = b * c
    return q('div', a, b), f'{a}÷{b}={c}。'


def gen_mixed():
    r = random.random()
    if r < 0.4:
        a, b, c = random.randint(2, 99), random.randint(2, 9), random.randint(2, 9)
        question = f'{a}+{b}×{c}等于几？'
        ans = f'先算乘法：{b}×{c}={b * c}，再算加法：{a}+{b * c}={a + b * c}。所以 {a}+{b}×{c}={a + b * c}。'
    elif r < 0.7:
        a, b, c = random.randint(2, 50), random.randint(2, 50), random.randint(2, 9)
        question = f'({a}+{b})×{c}等于几？'
        ans = f'先算括号里：{a}+{b}={a + b}，再算乘法：({a}+{b})×{c}={(a + b) * c}。'
    else:
        a, b, c = random.randint(10, 99), random.randint(2, 9), random.randint(2, 99)
        question = f'{a}×{b}-{c}等于几？'
        ans = f'先算乘法：{a}×{b}={a * b}，再算减法：{a * b}-{c}={a * b - c}。所以 {a}×{b}-{c}={a * b - c}。'
    return question, ans


def gen_cmp():
    a, b = random.randint(1, 999), random.randint(1, 999)
    if a == b:
        b += 1
    hi, lo = max(a, b), min(a, b)
    question = random.choice([f'{a}和{b}哪个数大？', f'比较 {a} 和 {b} 的大小。', f'{a}和{b}，哪个更大？'])
    ans = f'{hi}比{lo}大。' if random.random() < 0.5 else f'{hi}>{lo}，所以{hi}更大。'
    return question, ans


def gen_fill():
    r = random.random()
    if r < 0.5:
        a, x = random.randint(2, 99), random.randint(2, 99)
        question = f'{a}加多少等于{a + x}？'
        ans = f'{a + x}-{a}={x}，所以加{x}。'
    elif r < 0.8:
        b, c = random.randint(2, 9), random.randint(2, 99)
        question = f'多少乘{b}等于{b * c}？'
        ans = f'{b * c}÷{b}={c}，所以乘{c}。'
    else:
        a, c = random.randint(10, 99), random.randint(1, 9)
        question = f'多少减{a}等于{c}？'
        ans = f'{c}+{a}={c + a}，所以是{c + a}。'
    return question, ans


def gen_word():
    r = random.random()
    if r < 0.2:
        a, b = random.randint(3, 40), random.randint(2, 20)
        return f'小明有{a}个苹果，妈妈又给了他{b}个，现在一共有几个苹果？', f'{a}+{b}={a + b}（个），现在一共有{a + b}个苹果。'
    if r < 0.4:
        a, b = random.randint(10, 60), random.randint(2, 9)
        return f'树上有{a}只鸟，飞走了{b}只，还剩几只？', f'{a}-{b}={a - b}（只），还剩{a - b}只。'
    if r < 0.6:
        a, b = random.randint(10, 90), random.randint(2, 8)
        return f'一辆汽车每小时行驶{a}千米，{b}小时能行驶多少千米？', f'{a}×{b}={a * b}（千米），{b}小时能行驶{a * b}千米。'
    if r < 0.8:
        b, c = random.randint(2, 9), random.randint(2, 12)
        return f'把{b * c}颗糖平均分给{b}个小朋友，每人分到几颗？', f'{b * c}÷{b}={c}（颗），每人分到{c}颗。'
    a, b = random.randint(2, 15), random.randint(2, 9)
    return f'一支铅笔{a}元，买{b}支需要多少钱？', f'{a}×{b}={a * b}（元），需要{a * b}元。'


GENS = [(gen_add, 0.22), (gen_sub, 0.18), (gen_mul, 0.15), (gen_div, 0.10),
        (gen_mixed, 0.15), (gen_cmp, 0.07), (gen_fill, 0.06), (gen_word, 0.07)]

with open(OUT, 'w', encoding='utf-8') as f:
    for _ in range(N):
        g = random.choices([g for g, _ in GENS], weights=[w for _, w in GENS])[0]
        question, ans = g()
        sample = {'conversations': [
            {'from': 'human', 'value': question},
            {'from': 'gpt', 'value': ans},
        ]}
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f'done: {N} samples -> {OUT}')
