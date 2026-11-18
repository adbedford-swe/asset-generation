from pathlib import Path

from assetgen import webui


def test_builds():
    import gradio as gr

    assert isinstance(webui.build("cpu"), gr.Blocks)


def test_best_glb_and_stage_images_follow_progress(tmp_path):
    out = tmp_path
    assert webui.best_glb(out, "fox") is None and webui.stage_images(out, "fox") == []
    (out / "1_mesh").mkdir()
    (out / "1_mesh" / "mesh.glb").write_bytes(b"x")
    (out / "1_mesh" / "front.png").write_bytes(b"x")
    assert webui.best_glb(out, "fox") == out / "1_mesh" / "mesh.glb"
    (out / "2_rig").mkdir()
    (out / "2_rig" / "rigged.glb").write_bytes(b"x")
    assert webui.best_glb(out, "fox") == out / "2_rig" / "rigged.glb"
    (out / "fox.glb").write_bytes(b"x")
    assert webui.best_glb(out, "fox") == out / "fox.glb"
    assert [Path(p).name for p, _ in webui.stage_images(out, "fox")] == ["front.png"]


def test_viewer_html(tmp_path):
    assert webui.viewer_html(None) == webui.EMPTY_VIEWER
    glb = tmp_path / "a.glb"
    glb.write_bytes(b"x")
    html = webui.viewer_html(glb, 'walks "fast"')
    assert "<model-viewer" in html and "/gradio_api/file=" in html
    assert 'animation-name="walks &quot;fast&quot;"' in html


def test_slug():
    assert webui._slug("my fox!") == "my_fox" and webui._slug("") == "asset"
