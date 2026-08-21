import pandas as pd
import json
from django.test import TestCase, Client
from django.urls import reverse


class VisualizationDashboardTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "income": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
            "department": ["IT", "HR", "IT", "Sales", "HR", "Sales", "IT", "HR"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1]
        })
        self.df_json = self.df.to_json(orient="columns")

    def test_visualization_redirects_when_no_dataset(self):
        """Should redirect to upload_dataset if session contains no dataset."""
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("datasets", response.url)

    def test_dashboard_renders_dropdown_options(self):
        """Should render Chart Type dropdown with EXACTLY Histogram, Boxplot, Pie Chart, Pairplot."""
        session = self.client.session
        session["noise_free_dataset"] = self.df_json
        session.save()

        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visualization Dashboard")
        self.assertContains(response, "Histogram")
        self.assertContains(response, "Boxplot")
        self.assertContains(response, "Pie Chart")
        self.assertContains(response, "Pairplot")
        self.assertContains(response, "Continue to Feature Selection")

    def test_generate_histogram(self):
        """Should generate interactive Histogram plot for selected numerical column."""
        session = self.client.session
        session["noise_free_dataset"] = self.df_json
        session.save()

        response = self.client.post(reverse("dashboard"), {
            "chart_type": "histogram",
            "num_col": "income",
            "bins": "25"
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Histogram: income")
        self.assertContains(response, "Generated Chart")

    def test_generate_boxplot_with_grouping(self):
        """Should generate Boxplot for numerical feature grouped by categorical feature."""
        session = self.client.session
        session["noise_free_dataset"] = self.df_json
        session.save()

        response = self.client.post(reverse("dashboard"), {
            "chart_type": "boxplot",
            "num_col": "income",
            "cat_col": "department"
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Boxplot: income by department")

    def test_generate_pie_chart(self):
        """Should generate Pie Chart for selected categorical column."""
        session = self.client.session
        session["noise_free_dataset"] = self.df_json
        session.save()

        response = self.client.post(reverse("dashboard"), {
            "chart_type": "pie",
            "cat_col": "department"
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Distribution of department")

    def test_generate_pairplot(self):
        """Should generate Pairplot matrix for numerical columns."""
        session = self.client.session
        session["noise_free_dataset"] = self.df_json
        session.save()

        response = self.client.post(reverse("dashboard"), {
            "chart_type": "pairplot",
            "hue": "target"
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Pairplot: Relationships Between Numeric Variables")
