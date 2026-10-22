import unittest
from decimal import Decimal

from dataset.build_gsm8k_zh_sft import parse_chinese_final, validate_row


class GSM8KChineseSFTTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "split": "train",
            "question": "How many apples?",
            "question_zh": "有多少个苹果？",
            "answer_only": "12",
            "answer_zh": "计算后，答案是：12个。",
        }
        self.official = {"howmanyapples": Decimal(12)}

    def test_accepts_matching_final_and_official_answer(self):
        self.assertEqual(validate_row(self.row, self.official), ("有多少个苹果？", "howmanyapples", "12"))

    def test_supports_chinese_large_number_suffixes(self):
        self.assertEqual(parse_chinese_final("最终答案：3.5万"), 35000)

    def test_rejects_test_split(self):
        with self.assertRaisesRegex(ValueError, "not_training_split"):
            validate_row(self.row | {"split": "test"}, self.official)

    def test_rejects_answer_mismatch(self):
        with self.assertRaisesRegex(ValueError, "chinese_final_mismatches"):
            validate_row(self.row | {"answer_zh": "最终答案：13"}, self.official)

    def test_rejects_clock_time_as_numeric_final(self):
        with self.assertRaisesRegex(ValueError, "clock_or_colon"):
            parse_chinese_final("最终答案：4:30")


if __name__ == "__main__":
    unittest.main()
