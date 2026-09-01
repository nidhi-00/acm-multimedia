from demo.pipeline.qwen import QwenV7Runtime


def test_qwen_components_are_loaded_once_and_reused() -> None:
    calls = 0
    tokenizer = object()
    model = object()
    torch = object()

    def load_components() -> tuple[object, object, object]:
        nonlocal calls
        calls += 1
        return tokenizer, model, torch

    runtime = QwenV7Runtime(component_loader=load_components)

    assert runtime._load() == (tokenizer, model, torch)
    assert runtime._load() == (tokenizer, model, torch)
    assert calls == 1
