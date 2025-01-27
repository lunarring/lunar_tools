import unittest
import os
import time
import sys
import string
import importlib
import re



class TestInitImports(unittest.TestCase):
    def parse_imports_from_init(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
    
        import_statements = []
        for line in lines:
            # Adjust the pattern if your import style is different
            match = re.match(r'from\s+\.(?P<module>\w+)\s+import\s+(?P<attr>\w+)', line)
            if match:
                import_statements.append((match.group('module'), match.group('attr')))
        return import_statements
    
    def test_dynamic_imports(self):
        # Path to your __init__.py
        init_path = os.path.join(os.getcwd(), 'lunar_tools', '__init__.py')
        imports_to_test = self.parse_imports_from_init(init_path)
    

        for module_name, attr_name in imports_to_test:
            with self.subTest(module=module_name, attr=attr_name):
                module = importlib.import_module(module_name, package='lunar_tools')
                self.assertTrue(hasattr(module, attr_name), f"{attr_name} not found in {module_name}")


if __name__ == '__main__':
    unittest.main()
        
