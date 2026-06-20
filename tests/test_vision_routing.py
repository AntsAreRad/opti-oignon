#!/usr/bin/env python3
"""
Tests S48 -- Vision Auto-Routing + Image Upload
================================================

Tests unitaires et d'integration pour :
- Detection de contenu image dans les messages
- Routage automatique vers les modeles vision
- Extension des profils avec capacite vision
- Upload d'images et conversion base64
- Gestion multimodale dans l'executor
- Retrocompatibilite (texte seul)
- Schemas API avec champ images

Target : ~30 tests
"""

import base64
import os
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# Ajouter le repertoire racine au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# FIXTURES
# =============================================================================

# Image minimale PNG 1x1 pixel (valide)
TINY_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
TINY_PNG_B64 = base64.b64encode(TINY_PNG_BYTES).decode("ascii")


@pytest.fixture
def mock_ollama_models():
    """Mock ollama.list pour simuler les modeles disponibles."""
    models = [
        {"model": "qwen3:32b"},
        {"model": "qwen3-coder:30b"},
        {"model": "qwen3-vl:32b"},
        {"model": "llava:13b"},
        {"model": "nemotron-3-nano:30b"},
    ]
    mock_resp = {"models": models}
    with patch("opti_oignon.router.ollama.list", return_value=mock_resp):
        yield models


@pytest.fixture
def router_with_profiles(mock_ollama_models):
    """Router avec profils charges et modeles vision disponibles."""
    from opti_oignon.router import ModelRouter
    r = ModelRouter()
    # Forcer le refresh du cache
    r._available_models = []
    r._last_check = 0
    return r


# =============================================================================
# 1. MODEL PROFILES -- Vision Capability
# =============================================================================

class TestModelProfilesVision:
    """Tests des profils de modeles avec capacite vision."""

    def test_profiles_yaml_has_vision_models(self):
        """Le fichier YAML doit contenir au moins un modele vision."""
        from opti_oignon.model_profiles import profile_manager
        profile_manager._ensure_loaded()
        # Chercher les profils avec capacite vision
        vision_models = [
            p for p in profile_manager.list_profiles()
            if p.has_capability("vision")
        ]
        assert len(vision_models) >= 1, "Au moins un modele vision doit etre dans les profils"

    def test_qwen3_vl_has_vision_capability(self):
        """qwen3-vl:32b doit avoir la capacite vision."""
        from opti_oignon.model_profiles import profile_manager
        profile_manager._ensure_loaded()
        profile = profile_manager.get_profile("qwen3-vl:32b")
        assert profile is not None, "qwen3-vl:32b doit exister"
        assert profile.has_capability("vision")

    def test_llava_has_vision_capability(self):
        """llava:13b doit avoir la capacite vision."""
        from opti_oignon.model_profiles import profile_manager
        profile_manager._ensure_loaded()
        profile = profile_manager.get_profile("llava:13b")
        assert profile is not None, "llava:13b doit exister"
        assert profile.has_capability("vision")

    def test_find_best_for_vision_task(self):
        """find_best_for_task('vision') doit retourner des modeles incluant des modeles vision."""
        from opti_oignon.model_profiles import profile_manager
        profile_manager._ensure_loaded()
        results = profile_manager.find_best_for_task("vision", limit=5)
        assert len(results) >= 1
        # Au moins un resultat doit avoir la capacite vision
        has_vision = any(
            "vision" in p.capabilities or "vision" in p.recommended_for
            for p in results
        )
        assert has_vision, "Au moins un modele retourne doit avoir la capacite vision"

    def test_non_vision_model_lacks_capability(self):
        """qwen3-coder:30b ne doit PAS avoir la capacite vision."""
        from opti_oignon.model_profiles import profile_manager
        profile_manager._ensure_loaded()
        profile = profile_manager.get_profile("qwen3-coder:30b")
        assert profile is not None
        assert not profile.has_capability("vision")

    def test_vision_models_have_recommended_for(self):
        """Les modeles vision doivent avoir 'vision' dans recommended_for."""
        from opti_oignon.model_profiles import profile_manager
        profile_manager._ensure_loaded()
        for p in profile_manager.list_profiles():
            if p.has_capability("vision"):
                assert "vision" in p.recommended_for, (
                    f"{p.name} a la capacite vision mais pas dans recommended_for"
                )


# =============================================================================
# 2. ROUTER -- Image Detection
# =============================================================================

class TestImageDetection:
    """Tests de la detection d'images dans les messages."""

    def test_detect_explicit_images_list(self):
        """Detection avec une liste d'images base64 explicite."""
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        assert r.detect_images_in_message("Describe this", images=[TINY_PNG_B64])

    def test_detect_empty_images_list(self):
        """Pas de detection avec une liste vide."""
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        assert not r.detect_images_in_message("Hello", images=[])

    def test_detect_base64_inline(self):
        """Detection de donnees base64 inline dans le message."""
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        msg = f"Look at this: data:image/png;base64,{TINY_PNG_B64}"
        assert r.detect_images_in_message(msg)

    def test_detect_image_file_reference_png(self):
        """Detection de reference a un fichier .png."""
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        assert r.detect_images_in_message("Analyse le fichier photo.png")

    def test_detect_image_file_reference_jpg(self):
        """Detection de reference a un fichier .jpg."""
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        assert r.detect_images_in_message("What is in /tmp/image.jpg?")

    def test_detect_image_file_reference_webp(self):
        """Detection de reference a un fichier .webp."""
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        assert r.detect_images_in_message("Check output.webp")

    def test_no_detection_text_only(self):
        """Pas de detection pour un message texte classique."""
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        assert not r.detect_images_in_message("How to calculate Shannon diversity in R?")

    def test_no_detection_code_with_image_word(self):
        """Le mot 'image' seul dans du code ne declenche pas la detection."""
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        # Le mot "image" sans extension ne declenche pas
        assert not r.detect_images_in_message("docker build -t my_image .")

    def test_detect_none_images_parameter(self):
        """None en parametre images ne declenche pas la detection."""
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        assert not r.detect_images_in_message("Hello", images=None)


# =============================================================================
# 3. ROUTER -- Vision Auto-Routing
# =============================================================================

class TestVisionAutoRouting:
    """Tests du routage automatique vers les modeles vision."""

    def test_route_with_images_selects_vision_model(self, mock_ollama_models):
        """Quand des images sont fournies, un modele vision est selectionne."""
        from opti_oignon.analyzer import analyze
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        r._available_models = []
        r._last_check = 0
        analysis = analyze("Describe this image")
        result = r.route(analysis, images=[TINY_PNG_B64], message="Describe this image")
        # Le modele selectionne doit etre un modele vision
        assert result.model in ("qwen3-vl:32b", "llava:13b"), (
            f"Expected vision model, got {result.model}"
        )
        assert result.vision_routed is True

    def test_route_without_images_uses_standard(self, mock_ollama_models):
        """Sans images, le routage standard est utilise."""
        from opti_oignon.analyzer import analyze
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        r._available_models = []
        r._last_check = 0
        analysis = analyze("Write a Python function")
        result = r.route(analysis, message="Write a Python function")
        assert result.vision_routed is False
        assert result.model != "llava:13b"

    def test_route_vision_preserves_images(self, mock_ollama_models):
        """Les images sont preservees dans le RoutingResult."""
        from opti_oignon.analyzer import analyze
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        r._available_models = []
        r._last_check = 0
        imgs = [TINY_PNG_B64]
        analysis = analyze("What is this?")
        result = r.route(analysis, images=imgs, message="What is this?")
        assert result.images == imgs

    def test_route_vision_with_force_model(self, mock_ollama_models):
        """force_model a priorite sur le routage vision."""
        from opti_oignon.analyzer import analyze
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        r._available_models = []
        r._last_check = 0
        analysis = analyze("Describe this")
        result = r.route(
            analysis,
            force_model="qwen3:32b",
            images=[TINY_PNG_B64],
            message="Describe this",
        )
        assert result.model == "qwen3:32b"
        # vision_routed reste False car force_model a priorite
        assert result.vision_routed is False

    def test_route_vision_explanation_mentions_vision(self, mock_ollama_models):
        """L'explication de routage mentionne le routage vision."""
        from opti_oignon.analyzer import analyze
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        r._available_models = []
        r._last_check = 0
        analysis = analyze("What is this?")
        result = r.route(analysis, images=[TINY_PNG_B64], message="What is this?")
        assert "vision" in result.explanation.lower()

    def test_route_vision_task_type_set_to_vision(self, mock_ollama_models):
        """Le task_type est mis a 'vision' quand des images sont detectees."""
        from opti_oignon.analyzer import analyze
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        r._available_models = []
        r._last_check = 0
        analysis = analyze("What is this?")
        result = r.route(analysis, images=[TINY_PNG_B64], message="What is this?")
        assert result.task_type == "vision"

    def test_find_best_vision_model_returns_available(self, mock_ollama_models):
        """find_best_vision_model retourne un modele disponible."""
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        r._available_models = []
        r._last_check = 0
        result = r.find_best_vision_model()
        assert result is not None
        model, reason, alternatives = result
        assert model in ("qwen3-vl:32b", "llava:13b")
        assert reason == "vision_auto"

    def test_find_best_vision_model_no_vision_available(self):
        """Retourne None si aucun modele vision n'est disponible."""
        models = [{"model": "qwen3:32b"}, {"model": "nemotron-3-nano:30b"}]
        mock_resp = {"models": models}
        with patch("opti_oignon.router.ollama.list", return_value=mock_resp):
            from opti_oignon.router import ModelRouter
            r = ModelRouter()
            r._available_models = []
            r._last_check = 0
            result = r.find_best_vision_model()
            # Aucun modele vision dans la liste -> None
            assert result is None

    def test_route_fallback_when_no_vision_model(self):
        """Fallback vers routage standard si aucun modele vision disponible."""
        models = [{"model": "qwen3:32b"}, {"model": "nemotron-3-nano:30b"}]
        mock_resp = {"models": models}
        with patch("opti_oignon.router.ollama.list", return_value=mock_resp):
            from opti_oignon.analyzer import analyze
            from opti_oignon.router import ModelRouter
            r = ModelRouter()
            r._available_models = []
            r._last_check = 0
            analysis = analyze("What is in this image?")
            result = r.route(analysis, images=[TINY_PNG_B64], message="What is in this image?")
            # Doit fallback, pas d'erreur
            assert result.model in ("qwen3:32b", "nemotron-3-nano:30b")
            assert result.vision_routed is False


# =============================================================================
# 4. ROUTING RESULT -- Serialization
# =============================================================================

class TestRoutingResultSerialization:
    """Tests de serialisation du RoutingResult avec champs vision."""

    def test_to_dict_includes_vision_routed(self):
        """to_dict inclut vision_routed."""
        from opti_oignon.router import RoutingResult
        result = RoutingResult(
            model="qwen3-vl:32b",
            temperature=0.5,
            task_type="vision",
            prompt_variant="standard",
            model_type="general",
            priority_used="vision_auto",
            explanation="Vision auto-routed",
            timeout=120,
            vision_routed=True,
            images=[TINY_PNG_B64],
        )
        d = result.to_dict()
        assert "vision_routed" in d
        assert d["vision_routed"] is True

    def test_to_dict_vision_routed_default_false(self):
        """vision_routed est False par defaut."""
        from opti_oignon.router import RoutingResult
        result = RoutingResult(
            model="qwen3:32b",
            temperature=0.5,
            task_type="general",
            prompt_variant="standard",
            model_type="general",
            priority_used="primary",
            explanation="Standard routing",
            timeout=120,
        )
        d = result.to_dict()
        assert d["vision_routed"] is False


# =============================================================================
# 5. SCHEMAS -- ChatRequest with images
# =============================================================================

class TestChatRequestSchema:
    """Tests du schema ChatRequest avec le champ images."""

    def test_chat_request_accepts_images(self):
        """ChatRequest accepte le champ images."""
        from opti_oignon.api.schemas import ChatRequest
        req = ChatRequest(
            message="Describe this image",
            images=[TINY_PNG_B64],
        )
        assert req.images is not None
        assert len(req.images) == 1
        assert req.images[0] == TINY_PNG_B64

    def test_chat_request_images_optional(self):
        """images est optionnel dans ChatRequest."""
        from opti_oignon.api.schemas import ChatRequest
        req = ChatRequest(message="Hello")
        assert req.images is None

    def test_chat_request_empty_images_list(self):
        """ChatRequest accepte une liste vide d'images."""
        from opti_oignon.api.schemas import ChatRequest
        req = ChatRequest(message="Test", images=[])
        assert req.images == []

    def test_chat_request_multiple_images(self):
        """ChatRequest accepte plusieurs images."""
        from opti_oignon.api.schemas import ChatRequest
        req = ChatRequest(
            message="Compare these",
            images=[TINY_PNG_B64, TINY_PNG_B64],
        )
        assert len(req.images) == 2


# =============================================================================
# 6. SCHEMAS -- ImageUploadResponse
# =============================================================================

class TestImageUploadSchema:
    """Tests du schema ImageUploadResponse."""

    def test_image_upload_response_fields(self):
        """ImageUploadResponse a les champs attendus."""
        from opti_oignon.api.schemas import ImageUploadResponse
        resp = ImageUploadResponse(
            filename="test.png",
            size_bytes=1234,
            base64_data=TINY_PNG_B64,
            mime_type="image/png",
        )
        assert resp.filename == "test.png"
        assert resp.size_bytes == 1234
        assert resp.base64_data == TINY_PNG_B64
        assert resp.mime_type == "image/png"
        assert resp.width is None
        assert resp.height is None

    def test_image_upload_response_with_dimensions(self):
        """ImageUploadResponse avec dimensions optionnelles."""
        from opti_oignon.api.schemas import ImageUploadResponse
        resp = ImageUploadResponse(
            filename="photo.jpg",
            size_bytes=50000,
            base64_data="abc123",
            mime_type="image/jpeg",
            width=640,
            height=480,
        )
        assert resp.width == 640
        assert resp.height == 480


# =============================================================================
# 7. BACKWARD COMPATIBILITY
# =============================================================================

class TestBackwardCompatibility:
    """Tests de retrocompatibilite."""

    def test_route_without_images_param(self, mock_ollama_models):
        """route() fonctionne sans le parametre images (backward compatible)."""
        from opti_oignon.analyzer import analyze
        from opti_oignon.router import ModelRouter
        r = ModelRouter()
        r._available_models = []
        r._last_check = 0
        analysis = analyze("How to compute diversity in R?")
        # Appel sans images ni message
        result = r.route(analysis)
        assert result is not None
        assert result.vision_routed is False
        assert result.images == []

    def test_chat_request_backward_compatible(self):
        """ChatRequest sans images est retrocompatible."""
        from opti_oignon.api.schemas import ChatRequest
        # Simule un ancien client qui n'envoie pas le champ images
        req = ChatRequest(**{
            "message": "Hello",
            "model": "qwen3:32b",
            "think": True,
        })
        assert req.images is None
        assert req.think is True

    def test_routing_result_backward_compatible(self):
        """RoutingResult sans champs vision est retrocompatible."""
        from opti_oignon.router import RoutingResult
        result = RoutingResult(
            model="qwen3:32b",
            temperature=0.5,
            task_type="general",
            prompt_variant="standard",
            model_type="general",
            priority_used="primary",
            explanation="Standard",
            timeout=120,
        )
        # Les champs vision ont des valeurs par defaut
        assert result.vision_routed is False
        assert result.images == []


# =============================================================================
# 8. IMAGE UPLOAD ROUTE (routes_files.py)
# =============================================================================

class TestImageUploadRoute:
    """Tests de l'endpoint d'upload d'image."""

    def test_allowed_image_extensions(self):
        """Les extensions d'image attendues sont autorisees."""
        from opti_oignon.api.routes_files import ALLOWED_IMAGE_EXTENSIONS
        expected = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
        assert expected.issubset(ALLOWED_IMAGE_EXTENSIONS)

    def test_max_image_size(self):
        """La taille maximale d'image est de 10 MB."""
        from opti_oignon.api.routes_files import MAX_IMAGE_SIZE
        assert MAX_IMAGE_SIZE == 10_000_000

    def test_image_mime_types_mapping(self):
        """Le mapping des MIME types est correct."""
        from opti_oignon.api.routes_files import _IMAGE_MIME_TYPES
        assert _IMAGE_MIME_TYPES[".png"] == "image/png"
        assert _IMAGE_MIME_TYPES[".jpg"] == "image/jpeg"
        assert _IMAGE_MIME_TYPES[".jpeg"] == "image/jpeg"
        assert _IMAGE_MIME_TYPES[".gif"] == "image/gif"
        assert _IMAGE_MIME_TYPES[".webp"] == "image/webp"


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
