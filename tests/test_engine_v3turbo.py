from unittest.mock import patch

from vieneu.v3turbo import V3TurboVieNeuTTS


@patch.object(V3TurboVieNeuTTS, "_load_v3_voices")
@patch("vieneu._v3_turbo_engine.onnx_runtime_lite.OnnxV3LiteEngine")
@patch(
    "onnxruntime.get_available_providers",
    return_value=["CoreMLExecutionProvider", "CPUExecutionProvider"],
)
def test_v3_turbo_forwards_requested_onnx_providers(
    mock_available, mock_engine, mock_load_voices
):
    requested = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    V3TurboVieNeuTTS(
        backend="onnx",
        device="coreml",
        onnx_dir="dummy",
        onnx_providers=requested,
    )

    mock_available.assert_called_once_with()
    mock_engine.assert_called_once()
    assert mock_engine.call_args.kwargs["providers"] == requested
