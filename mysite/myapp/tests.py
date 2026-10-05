import json
import os
import shutil
import tempfile
from io import BytesIO
from unittest import mock

import numpy as np
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import translation
from PIL import Image

from . import views
from .models import AnalysisCT, BloodAnalysis, Disease


def make_png(name="scan.png"):
    buf = BytesIO()
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(buf, format="PNG")
    return SimpleUploadedFile(name, buf.getvalue(), content_type="image/png")


class PublicPagesTests(TestCase):
    def test_root_serves_analysis_page(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_analysis_page_matches_root(self):
        self.assertEqual(self.client.get(reverse("analysis_page")).status_code, 200)

    def test_login_and_register_pages_render(self):
        self.assertEqual(self.client.get(reverse("login")).status_code, 200)
        self.assertEqual(self.client.get(reverse("register")).status_code, 200)

    def test_upload_forms_render(self):
        for name in ("analyze_image2", "analyze_skin_image", "upload_image"):
            with self.subTest(name=name):
                self.assertEqual(self.client.get(reverse(name)).status_code, 200)

    def test_unknown_url_is_404(self):
        self.assertEqual(self.client.get("/definitely-not-a-page/").status_code, 404)


class AuthTests(TestCase):
    def test_register_creates_user_and_logs_in(self):
        resp = self.client.post(reverse("register"), {"email": "a@b.com", "password": "pw12345!"})
        self.assertRedirects(resp, reverse("analysis_page"), fetch_redirect_response=False)
        self.assertTrue(User.objects.filter(username="a@b.com").exists())
        self.assertIn("_auth_user_id", self.client.session)

    def test_register_rejects_duplicate_email(self):
        User.objects.create_user("a@b.com", password="pw12345!")
        resp = self.client.post(reverse("register"), {"email": "a@b.com", "password": "other"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(User.objects.filter(username="a@b.com").count(), 1)

    def test_login_success_and_failure(self):
        User.objects.create_user("a@b.com", password="pw12345!")
        bad = self.client.post(reverse("login"), {"email": "a@b.com", "password": "wrong"})
        self.assertEqual(bad.status_code, 200)
        self.assertNotIn("_auth_user_id", self.client.session)
        good = self.client.post(reverse("login"), {"email": "a@b.com", "password": "pw12345!"})
        self.assertEqual(good.status_code, 302)
        self.assertIn("_auth_user_id", self.client.session)


class DashboardTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("u@x.com", password="pw12345!")
        self.other = User.objects.create_user("o@x.com", password="pw12345!")

    def test_dashboard_requires_login(self):
        resp = self.client.get(reverse("user_kab"))
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(resp.url.startswith("/login/"))

    def test_dashboard_only_shows_own_analyses(self):
        AnalysisCT.objects.create(user=self.user, result="mine: 90%")
        AnalysisCT.objects.create(user=self.other, result="theirs: 90%")
        self.client.force_login(self.user)
        resp = self.client.get(reverse("user_kab"), {"section": "ct"})
        self.assertEqual(resp.status_code, 200)
        results = [a.result for a in resp.context["analyses"]]
        self.assertEqual(results, ["mine: 90%"])

    def test_unknown_section_is_handled(self):
        self.client.force_login(self.user)
        resp = self.client.get(reverse("user_kab"), {"section": "nope"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(list(resp.context["analyses"]), [])

    def test_save_results_ct_redirects_to_dashboard(self):
        analysis = AnalysisCT.objects.create(user=self.user, result="normal: 99%")
        self.client.force_login(self.user)
        resp = self.client.post(reverse("save_results_ct"), {"analysis_id": analysis.id})
        self.assertRedirects(resp, reverse("user_kab"), fetch_redirect_response=False)

    def test_save_results_ct_unknown_id_does_not_crash(self):
        self.client.force_login(self.user)
        resp = self.client.post(reverse("save_results_ct"), {"analysis_id": 99999})
        self.assertEqual(resp.status_code, 302)

    def test_save_results_ct_ignores_other_users_analysis(self):
        analysis = AnalysisCT.objects.create(user=self.other, result="normal: 99%")
        self.client.force_login(self.user)
        resp = self.client.post(reverse("save_results_ct"), {"analysis_id": analysis.id})
        self.assertEqual(resp.status_code, 302)


class TreatmentLookupTests(TestCase):
    def test_known_condition_returns_treatment(self):
        Disease.objects.create(name="Flu", treatment="Rest and fluids")
        resp = self.client.post(reverse("treatment_view"), {"illness": "flu"})
        self.assertContains(resp, "Rest and fluids")

    def test_unknown_condition_shows_fallback(self):
        resp = self.client.post(reverse("treatment_view"), {"illness": "nonexistent"})
        self.assertEqual(resp.status_code, 200)
        self.assertNotContains(resp, "Rest and fluids")


class BloodAnalysisApiTests(TestCase):
    def test_details_returns_values(self):
        analysis = BloodAnalysis.objects.create(hemoglobin_level=13.5, leukocytes_level=6.1)
        resp = self.client.get(reverse("get_analysis_details", args=[analysis.id]))
        data = resp.json()
        self.assertEqual(data["hemoglobin_level"], 13.5)
        self.assertEqual(data["leukocytes_level"], 6.1)

    def test_details_404_for_missing(self):
        resp = self.client.get(reverse("get_analysis_details", args=[424242]))
        self.assertEqual(resp.status_code, 404)


class ChatbotTests(TestCase):
    url = "/chatbot_api/"

    def _post(self, message="hello"):
        return self.client.post(self.url, json.dumps({"message": message}), content_type="application/json")

    def test_get_not_allowed(self):
        self.assertEqual(self.client.get(self.url).status_code, 405)

    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    def test_missing_api_key_returns_500(self):
        self.assertEqual(self._post().status_code, 500)

    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_returns_model_answer_and_uses_page_language(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value.choices = [
            mock.Mock(message=mock.Mock(content="  Drink water.  "))
        ]
        with mock.patch.object(views, "_get_ai_client", return_value=fake_client):
            resp = self.client.post(
                self.url,
                json.dumps({"message": "У меня температура"}),
                content_type="application/json",
                HTTP_ACCEPT_LANGUAGE="ru",
            )
        self.assertEqual(resp.json(), {"answer": "Drink water."})
        messages = fake_client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], views.CHAT_SYSTEM_PROMPTS["ru"])
        self.assertEqual(messages[1]["content"], "У меня температура")


class AiLanguageTests(TestCase):
    def test_russian_and_english_detected(self):
        with translation.override("ru"):
            self.assertEqual(views._current_ai_language(), "ru")
        with translation.override("en"):
            self.assertEqual(views._current_ai_language(), "en")

    def test_prompts_exist_for_both_languages(self):
        for lang in ("ru", "en"):
            self.assertIn(lang, views.CHAT_SYSTEM_PROMPTS)
            self.assertIn(lang, views.RECOMMENDATION_KEYWORDS)


class CTLabelTests(TestCase):
    def test_labels_cover_all_four_classes_in_training_order(self):
        self.assertEqual(sorted(views.CT_CLASS_LABELS), [0, 1, 2, 3])
        self.assertEqual(views.CT_CLASS_LABELS[2], "normal")
        self.assertTrue(views.CT_CLASS_LABELS[0].startswith("adenocarcinoma"))
        self.assertTrue(views.CT_CLASS_LABELS[1].startswith("large.cell"))
        self.assertTrue(views.CT_CLASS_LABELS[3].startswith("squamous.cell"))


class CTUploadTests(TestCase):
    """Regression test: stored analysis_file must be relative to MEDIA_ROOT so the
    dashboard can find it (an absolute path used to show 'Analysis file missing')."""

    def setUp(self):
        self.media = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.media, ignore_errors=True)

    def test_upload_saves_prediction_with_relative_path(self):
        probs = np.array([[0.05, 0.05, 0.85, 0.05]])
        fake_model = mock.Mock()
        fake_model.predict.return_value = probs
        with override_settings(MEDIA_ROOT=self.media), mock.patch.object(views, "model", fake_model):
            resp = self.client.post(reverse("analyze_image2"), {"file": make_png()})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.context["predicted_class"], "normal")
            analysis = AnalysisCT.objects.get()
            self.assertTrue(analysis.result.startswith("normal: 85"))
            self.assertFalse(os.path.isabs(analysis.analysis_file.name))
            self.assertTrue(FileSystemStorage(location=self.media).exists(analysis.analysis_file.name))
