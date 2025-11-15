import unittest
from unittest import mock

from ramjet.tasks.gptchat.tts.azure import tts


class TestAzureTTS(unittest.TestCase):
    @mock.patch("ramjet.tasks.gptchat.tts.azure.speechsdk.SpeechSynthesizer")
    @mock.patch("ramjet.tasks.gptchat.tts.azure.speechsdk.SpeechConfig")
    def test_synthesize_text_to_speech_with_gpt(self, mock_config, mock_synthesizer):
        fake_result = mock.Mock(audio_data=b"audio-bytes", audio_duration=1.23)
        synth_instance = mock_synthesizer.return_value
        synth_instance.speak_text.return_value = fake_result
        synth_instance.synthesizing.connect.return_value = None

        config_instance = mock_config.return_value
        config_instance.set_properties_by_name.return_value = None

        result = tts(text="hello, world")

        self.assertEqual(result.audio, b"audio-bytes")
        self.assertEqual(result.duration, 1.23)
        synth_instance.speak_text.assert_called_once_with(text="hello, world")


if __name__ == "__main__":
    unittest.main()
