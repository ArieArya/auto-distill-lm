from core.engines.hf_inference_engine import HFInferenceEngine
from core.engines.vllm_inference_engine import VLLMInferenceEngine
from core.engines.openai_inference_engine import OpenAIInferenceEngine
from core.interfaces.inference_engine import InferenceEngine

def create_engine(
    engine_type: str,
    model_id: str
) -> InferenceEngine:
    engine_type = engine_type.lower()

    if engine_type == "hf":
        return HFInferenceEngine(model_id)
    elif engine_type == "vllm":
        return VLLMInferenceEngine(model_id)
    elif engine_type == "openai":
        return OpenAIInferenceEngine(model_id)
    else:
        raise ValueError(f"Unsupported engine type: {engine_type}")
