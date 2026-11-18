from pathlib import Path

from assetgen import server, webui
from assetgen.imagegen import ImageSpec


def test_builds(tmp_path):
    import gradio as gr

    manager = server.JobManager(tmp_path, ["cpu"], ImageSpec(backend="sdcpp", url="http://127.0.0.1:9"))
    assert isinstance(webui.build(manager), gr.Blocks)


def test_best_glb_and_stage_images_follow_progress(tmp_path):
    out = tmp_path
    assert server.best_glb(out, "fox") is None and webui.stage_images(out, "fox") == []
    (out / "1_mesh").mkdir()
    (out / "1_mesh" / "mesh.glb").write_bytes(b"x")
    (out / "1_mesh" / "front.png").write_bytes(b"x")
    assert server.best_glb(out, "fox") == out / "1_mesh" / "mesh.glb"
    (out / "2_rig").mkdir()
    (out / "2_rig" / "rigged.glb").write_bytes(b"x")
    assert server.best_glb(out, "fox") == out / "2_rig" / "rigged.glb"
    (out / "fox.glb").write_bytes(b"x")
    assert server.best_glb(out, "fox") == out / "fox.glb"
    assert [Path(p).name for p, _ in webui.stage_images(out, "fox")] == ["front.png"]


def test_viewer_html(tmp_path):
    assert webui.viewer_html(None) == webui.EMPTY_VIEWER
    glb = tmp_path / "a.glb"
    glb.write_bytes(b"x")
    html = webui.viewer_html(glb, 'walks "fast"')
    assert "<model-viewer" in html and "/gradio_api/file=" in html
    assert 'animation-name="walks &quot;fast&quot;"' in html


def test_character_markdown():
    data = {"name": "fox", "model": {"height": 0.6, "triangles": 20000}, "skeleton": {"joints": [{}] * 13},
            "source": {"prompt": "a red fox"},
            "clips": [{"name": "walks_forward", "role": "walk", "loop": True, "duration": 3.0, "speed": 0.22},
                      {"name": "jumps", "role": "action", "loop": False, "duration": 1.9}],
            "locomotion": {"blend": [{"clip": "idle", "speed": 0.0}, {"clip": "walks_forward", "speed": 0.22}]}}
    text = webui.character_markdown(data)
    assert "0.60 m tall, 13 joints" in text and "*a red fox*" in text
    assert "| `walks_forward` | walk | yes | 3.00 s | 0.22 m/s |" in text and "| `jumps` | action | once |" in text
    assert "`idle` (0.00) → `walks_forward` (0.22)" in text


def test_slug():
    assert server.slug("my fox!") == "my_fox" and server.slug("") == "asset"
