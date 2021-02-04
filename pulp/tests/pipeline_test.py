import unittest

from pulp.pipeline.QAPipeLine import QAPipeLine


class PipeLineTestCase(unittest.TestCase):
    def run_pipeline(self):
        pl = QAPipeLine()
        print(pl("1702.00505.txt","What is The goal of this exploration"))
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
