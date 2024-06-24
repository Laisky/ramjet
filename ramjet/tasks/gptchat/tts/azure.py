import azure.cognitiveservices.speech as speechsdk

from ramjet import settings

from .base import AudioResult


def tts(text: str) -> AudioResult:
    """
    Synthesizes the given text to speech using Azure Speech Service.

    Args:
        text (str): The text to synthesize.

    Returns:
        AudioResult: The audio result.
    """
    # Setup speech synthesizer with websocket v2 endpoint
    speech_config = speechsdk.SpeechConfig(
        endpoint=f"wss://{settings.AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/websocket/v2",
        subscription=settings.AZURE_SPEECH_KEY,
    )

    # Set a voice name
    speech_config.speech_synthesis_voice_name = "en-US-AvaMultilingualNeural"

    # Set timeout values to avoid SDK canceling the request due to GPT latency
    properties = {
        "SpeechSynthesis_FrameTimeoutInterval": "100000000",
        "SpeechSynthesis_RtfTimeoutThreshold": "10",
    }
    speech_config.set_properties_by_name(properties)

    # Create speech synthesizer
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    speech_synthesizer.synthesizing.connect(lambda evt: print("[audio]", end=""))

    result = speech_synthesizer.speak_text(text=text)
    return AudioResult(
        audio=result.audio_data,
        duration=result.audio_duration,
    )
