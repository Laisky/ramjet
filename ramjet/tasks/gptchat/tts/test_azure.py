import unittest
from ramjet.tasks.gptchat.tts.azure import (
    tts,
)


class TestAzureTTS(unittest.TestCase):
    def test_synthesize_text_to_speech_with_gpt(self):
        text = "hello, world"
        result = tts(text=text)

        # write audio data
        with open("audio.wav", "wb") as f:
            f.write(result.audio)


if __name__ == "__main__":
    unittest.main()
