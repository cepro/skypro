import logging
import unittest
import subprocess


class TestBlah(unittest.TestCase):

    def test_blah(self):
        # print(subprocess.check_output(['pwd']))

        logging.info("Here")
        res = subprocess.run([
            'python3',
            './src/skypro/main.py',
            'simulate',
            '-y',
            '-c',
            './src/tests/integration/fixtures/config.json'
        ])
        logging.info("There")

        if res.returncode != 0:
            raise ValueError("Non zero exit code")