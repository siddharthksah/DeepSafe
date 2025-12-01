import pytest
import os
from deepsafe_utils.media_handler import MediaHandler


@pytest.mark.integration
class TestDeepSafeIntegration:

    def test_api_health(self, api_client):
        """Verify the Main API is healthy."""
        health = api_client.check_main_api_health()
        assert health.get("overall_api_status") == "healthy"
        assert "media_type_details" in health

    def test_image_prediction_flow(self, api_client, config_manager, sample_image_path):
        """Test the full prediction flow for a single image."""
        media_handler = MediaHandler(config_manager)
        encoded_media = media_handler.encode_media_to_base64(sample_image_path)
        assert encoded_media is not None

        # Test with 'stacking' ensemble (default)
        result = api_client.test_with_main_api(
            media_path=sample_image_path,
            media_type="image",
            encoded_media=encoded_media,
            threshold=0.5,
            ensemble_method="stacking",
        )

        assert "error" not in result
        assert "verdict" in result
        assert result["verdict"] in ["real", "fake"]
        assert "ensemble_score_is_fake" in result
        assert 0.0 <= result["ensemble_score_is_fake"] <= 1.0

    def test_individual_model_prediction(
        self, api_client, config_manager, sample_image_path
    ):
        """Test a specific individual model (NPR)."""
        media_handler = MediaHandler(config_manager)
        encoded_media = media_handler.encode_media_to_base64(sample_image_path)

        # Check if NPR model is configured
        models = config_manager.get_model_endpoints("image")
        if "npr_deepfakedetection" not in models:
            pytest.skip("npr_deepfakedetection not configured")

        result = api_client.test_with_individual_model(
            model_name="npr_deepfakedetection",
            media_path=sample_image_path,
            encoded_media=encoded_media,
            threshold=0.5,
        )

        assert "error" not in result
        assert "probability" in result
        assert "class" in result
