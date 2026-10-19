"""Shared, bounded calculator protocol for local and API model evaluation."""
import ast
from decimal import Decimal, localcontext
import re

SYSTEM_PROMPT = '你是一个准确、简洁的数学助手。请认真计算，并按要求给出最终答案。'
CALCULATOR_PROMPT = '''
You may use a calculator. To call it, output ONLY <calculator>expression</calculator>
and wait for the result. Supports decimal numbers, parentheses, +, -, *, /, //, %, ** (integer exponent -8 to 8).
Example call: <calculator>(18 + 27) / 3</calculator>
The next user message will contain the calculator result. Then continue solving or
provide the final answer in the requested format. Do not invent tool results.
你可以调用计算器：单独输出 <calculator>算式</calculator>，等待返回结果后继续作答。
'''
CALL_RE = re.compile(r'<calculator>(.*?)</calculator>', re.S)


def calculate(expression):
    """Interpret only a small arithmetic AST, never execute model-generated code."""
    if len(expression) > 256:
        raise ValueError('expression too long')
    tree = ast.parse(expression.strip(), mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError('expression too complex')

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = Decimal(ast.get_source_segment(expression.strip(), node))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            result = value if isinstance(node.op, ast.UAdd) else -value
        elif isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): result = left + right
            elif isinstance(node.op, ast.Sub): result = left - right
            elif isinstance(node.op, ast.Mult): result = left * right
            elif isinstance(node.op, ast.Div): result = left / right
            elif isinstance(node.op, ast.FloorDiv): result = (left / right).__floor__()
            elif isinstance(node.op, ast.Mod): result = left - Decimal((left / right).__floor__()) * right
            elif isinstance(node.op, ast.Pow):
                if right != right.to_integral_value() or abs(right) > 8:
                    raise ValueError('exponent must be an integer between -8 and 8')
                result = left ** int(right)
            else: raise ValueError('unsupported operator')
            result = Decimal(result)
        else:
            raise ValueError('only arithmetic expressions are supported')
        if (not result.is_finite() or abs(result) > Decimal('1e100')
                or (result != 0 and abs(result) < Decimal('1e-100'))):
            raise ValueError('result out of range')
        return result

    with localcontext() as context:
        context.prec = 40
        result = visit(tree.body)
        return format(result, 'f').rstrip('0').rstrip('.') if '.' in format(result, 'f') else format(result, 'f')


def add_calculator_args(parser):
    parser.add_argument('--calculator', action='store_true', help='Enable shared calculator protocol')
    parser.add_argument('--max-tool-calls', type=int, default=4)


def calculator_settings(args):
    return {'enabled': args.calculator, 'protocol': 'calculator_xml_v1',
            'max_tool_calls': args.max_tool_calls,
            'total_completion_token_budget': args.max_new_tokens}


def run_dialogue(generate_turn, prompt, token_budget, enabled=False, max_calls=4):
    """generate_turn(messages, remaining_tokens) returns (text, generated_token_count)."""
    if token_budget < 1 or max_calls < 0:
        raise ValueError('token budget must be positive; max calls must be nonnegative')
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT + (CALCULATOR_PROMPT if enabled else '')},
                {'role': 'user', 'content': prompt}]
    trace, used, calls = [], 0, 0
    while used < token_budget:
        answer, count = generate_turn(messages, token_budget - used)
        if not isinstance(count, int) or count < 0 or count > token_budget - used:
            raise ValueError('invalid completion token usage')
        used += count
        trace.append({'role': 'assistant', 'content': answer, 'completion_tokens': count})
        if not enabled or '<calculator>' not in answer:
            return answer, {'trace': trace, 'completion_tokens': used, 'tool_calls': calls, 'stop_reason': 'final'}
        match = CALL_RE.search(answer)
        if not match:
            reason = 'invalid_tool_call'
            break
        if calls >= max_calls:
            reason = 'tool_limit'
            break
        calls += 1
        expression = match.group(1)
        try:
            result = calculate(expression)
            feedback = f'Calculator result: {result}'
            error = False
        except (ValueError, SyntaxError, ArithmeticError, RecursionError) as exc:
            result = str(exc)
            feedback = f'Calculator error: {result}'
            error = True
        trace.append({'role': 'calculator', 'expression': expression, 'result': result, 'error': error,
                      'discarded_after_call': answer[match.end():]})
        messages.extend([{'role': 'assistant', 'content': answer[:match.end()]}, {'role': 'user', 'content': feedback}])
        if count == 0:
            reason = 'no_token_progress'
            break
    else:
        reason = 'token_limit'
    # A tool request or tool result is never scored as the model's final answer.
    return '', {'trace': trace, 'completion_tokens': used, 'tool_calls': calls, 'stop_reason': reason}
