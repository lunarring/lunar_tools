import unittest
import os
import time
import sys
import string


class TestPackageImport(unittest.TestCase):
    def test_import(self):
        # Determine the path to the directory containing your package
        package_dir = os.path.join(os.getcwd())
        sys.path.insert(0, package_dir)

        try:
            import lunar_tools 
        except ImportError as e:
            self.fail(f"Failed to import package: {e}")

if __name__ == '__main__':
    unittest.main()
        



if __name__ == "__main__":
    unittest.main()

