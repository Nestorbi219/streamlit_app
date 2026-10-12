import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class DashboardSmokeTest(unittest.TestCase):
    def test_demo_dashboard_renders_without_errors(self) -> None:
        app = AppTest.from_file(
            str(ROOT / "streamlit_app.py"),
            default_timeout=30,
        ).run()

        self.assertEqual([], list(app.exception))
        self.assertEqual(4, len(app.metric))
        self.assertGreaterEqual(len(app.get("plotly_chart")), 3)
        self.assertEqual(2, len(app.get("download_button")))


if __name__ == "__main__":
    unittest.main()
