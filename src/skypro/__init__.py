def run_tests():
    import unittest

    loader = unittest.TestLoader()
    suite = loader.discover("./src", pattern="test*.py")
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise Exception("Tests failed")


def run_linter():
    import subprocess

    subprocess.run(["ruff", "check", "."], check=True)
