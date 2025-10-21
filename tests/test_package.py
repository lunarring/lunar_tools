import importlib
import os
import re
import unittest

from lunar_tools._optional import OptionalDependencyError


class TestInitImports(unittest.TestCase):
    def parse_imports_from_init(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        import_statements = []
        for line in lines:
            match = re.match(r"from\s+\.(?P<module>\w+)\s+import\s+(?P<attr>\w+)", line)
            if match:
                import_statements.append((match.group("module"), match.group("attr")))
        return import_statements

    def test_dynamic_imports(self):
        init_path = os.path.join(os.getcwd(), "lunar_tools", "__init__.py")
        imports_to_test = self.parse_imports_from_init(init_path)

        for module_name, attr_name in imports_to_test:
            with self.subTest(module=module_name, attr=attr_name):
                full_name = f"lunar_tools.{module_name}"
                try:
                    module = importlib.import_module(full_name)
                except OptionalDependencyError:
                    self.skipTest(f"Optional dependency not installed for {module_name}.{attr_name}")
                except ModuleNotFoundError:
                    self.skipTest(f"Module {full_name} not available")
                else:
                    self.assertTrue(hasattr(module, attr_name), f"{attr_name} not found in {full_name}")


if __name__ == "__main__":
    unittest.main()
