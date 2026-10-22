import unittest

from dataset.build_chinese_math_sft import evaluate_expression, validate_row


class ChineseMathSFTTests(unittest.TestCase):
    def test_evaluates_percent_and_multiplication(self):
        self.assertEqual(evaluate_expression("800 × 75%"), 600)

    def test_accepts_consistent_solution(self):
        row = {
            "question": "原价800元，打75折后多少钱？",
            "chain_of_thought": "折后价格：800 × 75% = 600元\n再优惠28元：600 - 28 = 572元",
            "final_answer": "最终价格是572元",
        }
        self.assertEqual(validate_row(row), (row["question"], row["chain_of_thought"], "572"))

    def test_rejects_incorrect_intermediate_arithmetic(self):
        row = {
            "question": "每天生产72个，做22天，共多少？",
            "chain_of_thought": "总量：72 × 22 = 1634个\n答案：1634个",
            "final_answer": "1634个",
        }
        with self.assertRaisesRegex(ValueError, "arithmetic_mismatch"):
            validate_row(row)

    def test_rejects_final_answer_that_disagrees_with_last_equation(self):
        row = {
            "question": "12个苹果送出3个，还剩多少？",
            "chain_of_thought": "剩下：12 - 3 = 9个",
            "final_answer": "还剩8个",
        }
        with self.assertRaisesRegex(ValueError, "final_answer_mismatch"):
            validate_row(row)


if __name__ == "__main__":
    unittest.main()
