import unittest
from eval_calculator import calculate, run_dialogue


class CalculatorTests(unittest.TestCase):
    def test_arithmetic(self):
        for expression, expected in [('91-36', '55'), ('0.1+0.2', '0.3'),
                                     ('(18+27)/3', '15'), ('-7//3', '-3'),
                                     ('-7%3', '2'), ('2*0', '0')]:
            with self.subTest(expression=expression):
                self.assertEqual(calculate(expression), expected)

    def test_rejects_code_and_unbounded_work(self):
        for expression in ['__import__("os").getcwd()', '2**1000000', '[1]*100',
                           'True+1', '1/0', '9'*257, '1e101', '1e-999999']:
            with self.subTest(expression=expression):
                with self.assertRaises((ValueError, ArithmeticError)):
                    calculate(expression)

    def test_round_trip_and_shared_budget(self):
        budgets = []
        def generate(messages, budget):
            budgets.append(budget)
            if len(budgets) == 1:
                return '<calculator>91-36</calculator>', 8
            self.assertEqual(messages[-1]['content'], 'Calculator result: 55')
            return '最终答案：55', 6
        answer, log = run_dialogue(generate, '91减36', 20, True)
        self.assertEqual(answer, '最终答案：55')
        self.assertEqual(budgets, [20, 12])
        self.assertEqual(log['tool_calls'], 1)
        self.assertEqual(log['completion_tokens'], 14)

    def test_errors_returned_to_model(self):
        turns = iter([('<calculator>1/0</calculator>', 5), ('Final answer: 0', 4)])
        answer, log = run_dialogue(lambda *_: next(turns), 'test', 20, True)
        self.assertTrue(log['trace'][1]['error'])
        self.assertEqual(answer, 'Final answer: 0')

    def test_exhausted_or_malformed_calls_not_scored(self):
        for text, budget, calls, reason in [
            ('<calculator>1+1</calculator>', 5, 4, 'token_limit'),
            ('<calculator>1+1</calculator>', 20, 0, 'tool_limit'),
            ('<calculator>1+1', 20, 4, 'invalid_tool_call')]:
            answer, log = run_dialogue(lambda *_: (text, 5), 'test', budget, True, calls)
            self.assertEqual(answer, '')
            self.assertEqual(log['stop_reason'], reason)

    def test_premature_answer_is_discarded_and_tool_really_runs(self):
        turns = iter([('<calculator>2+2</calculator>\nFinal answer: 99', 12),
                      ('Final answer: 4', 5)])
        def generate(messages, _):
            if len(messages) > 2:
                self.assertNotIn('99', messages[-2]['content'])
                self.assertEqual(messages[-1]['content'], 'Calculator result: 4')
            return next(turns)
        answer, log = run_dialogue(generate, 'test', 30, True)
        self.assertEqual(answer, 'Final answer: 4')
        self.assertEqual(log['tool_calls'], 1)
        self.assertIn('99', log['trace'][1]['discarded_after_call'])

    def test_disabled_uses_original_system_prompt(self):
        def generate(messages, _):
            self.assertNotIn('<calculator>', messages[0]['content'])
            return 'Final answer: 2', 5
        answer, log = run_dialogue(generate, 'test', 20)
        self.assertEqual(answer, 'Final answer: 2')
        self.assertEqual(log['tool_calls'], 0)


if __name__ == '__main__':
    unittest.main()
