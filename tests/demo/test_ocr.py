from demo.pipeline.ocr import detections_to_text


def test_detections_are_sorted_into_reading_order() -> None:
    detections = [
        (
            [[100, 40], [150, 40], [150, 60], [100, 60]],
            "world",
            0.8,
        ),
        (
            [[10, 40], [80, 40], [80, 60], [10, 60]],
            "Hello",
            0.9,
        ),
        (
            [[10, 90], [80, 90], [80, 110], [10, 110]],
            "again",
            0.7,
        ),
    ]

    text, confidence = detections_to_text(detections)

    assert text == "Hello world again"
    assert confidence is not None
    assert round(confidence, 3) == 0.8


def test_empty_detections_return_no_text() -> None:
    assert detections_to_text([]) == (
        None,
        None,
    )
