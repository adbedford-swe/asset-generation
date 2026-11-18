@tool
extends EditorScenePostImport
## Runs on an assetgen character's .glb as it imports (the .glb.import's import_script). Reads the
## .assetgen file beside it and makes the scene ready to play:
##
## - every clip loops, or plays once, as the character file says;
## - an AnimationTree: a state machine whose "Locomotion" state blends the idle, walk and run
##   clips by speed (metres per second), with a state for every clip; one-shots return to
##   Locomotion when they end;
## - the AssetgenCharacter script on the root, holding the clips' metadata.

const CharacterScript := preload("res://addons/assetgen/assetgen_character.gd")
const LOCOMOTION := &"Locomotion"
const FADE := 0.15


func _post_import(scene: Node) -> Object:
	var path := get_source_file().get_basename() + ".assetgen"
	if not FileAccess.file_exists(path):
		push_warning("assetgen: no character file at %s; imported as a plain model" % path)
		return scene
	var data = JSON.parse_string(FileAccess.get_file_as_string(path))
	if typeof(data) != TYPE_DICTIONARY or data.get("format") != "assetgen.character":
		push_error("assetgen: %s is not an assetgen character file" % path)
		return scene
	if int(data.get("version", 1)) > 1:
		push_error("assetgen: %s is a newer character format than this addon reads; update it" % path)
		return scene
	var player := _find_player(scene)
	if player == null:
		push_error("assetgen: %s has no AnimationPlayer; was it imported with animation off?" % get_source_file())
		return scene
	build(scene, player, data)
	return scene


static func build(root: Node, player: AnimationPlayer, data: Dictionary) -> void:
	var clips := {}
	for clip in data.get("clips", []):
		var animation := _animation(player, clip["name"])
		if animation == null:
			push_warning("assetgen: the model has no animation named '%s'" % clip["name"])
			continue
		animation.loop_mode = Animation.LOOP_LINEAR if clip.get("loop", false) else Animation.LOOP_NONE
		clips[clip["name"]] = {
			"role": clip.get("role", "loop"), "loop": clip.get("loop", false),
			"duration": float(clip.get("duration", animation.length)), "speed": float(clip.get("speed", 0.0)),
			"prompt": clip.get("prompt", ""),
		}

	var machine := AnimationNodeStateMachine.new()
	var blend_points: Array = data.get("locomotion", {}).get("blend", []).filter(func(p): return clips.has(p["clip"]))
	if blend_points.size() >= 2:
		var space := AnimationNodeBlendSpace1D.new()
		space.min_space = 0.0
		space.max_space = maxf(float(blend_points[-1]["speed"]) * 1.5, 0.1)
		space.snap = 0.01
		for point in blend_points:
			var node := AnimationNodeAnimation.new()
			node.animation = point["clip"]
			space.add_blend_point(node, float(point["speed"]), -1, StringName(point["clip"]))
		machine.add_node(LOCOMOTION, space, Vector2(300, 0))
	else:
		var node := AnimationNodeAnimation.new()
		node.animation = _idle(data, clips)
		machine.add_node(LOCOMOTION, node, Vector2(300, 0))
	var start := AnimationNodeStateMachineTransition.new()
	start.advance_mode = AnimationNodeStateMachineTransition.ADVANCE_MODE_AUTO
	machine.add_transition(&"Start", LOCOMOTION, start)

	var row := 0
	for clip_name in clips:
		if clip_name == LOCOMOTION:
			continue
		var node := AnimationNodeAnimation.new()
		node.animation = clip_name
		machine.add_node(clip_name, node, Vector2(650, 80 * row))
		row += 1
		var there := AnimationNodeStateMachineTransition.new()
		there.xfade_time = FADE
		machine.add_transition(LOCOMOTION, clip_name, there)
		var back := AnimationNodeStateMachineTransition.new()
		back.xfade_time = FADE
		if not clips[clip_name]["loop"]:
			back.switch_mode = AnimationNodeStateMachineTransition.SWITCH_MODE_AT_END
			back.advance_mode = AnimationNodeStateMachineTransition.ADVANCE_MODE_AUTO
		machine.add_transition(clip_name, LOCOMOTION, back)

	var tree := AnimationTree.new()
	tree.name = "AnimationTree"
	tree.tree_root = machine
	root.add_child(tree)
	tree.owner = root
	tree.anim_player = tree.get_path_to(player)
	tree.active = true

	var model: Dictionary = data.get("model", {})
	var source: Dictionary = data.get("source", {})
	root.set_script(CharacterScript)
	root.set("clips", clips)
	root.set("locomotion_max_speed", float(blend_points[-1]["speed"]) if blend_points.size() >= 2 else 0.0)
	root.set("character_height", float(model.get("height", 0.0)))
	root.set("prompt", str(source.get("prompt", "")))
	root.set_meta("assetgen", data)


static func _find_player(node: Node) -> AnimationPlayer:
	if node is AnimationPlayer:
		return node
	for child in node.get_children():
		var found := _find_player(child)
		if found != null:
			return found
	return null


static func _animation(player: AnimationPlayer, clip_name: String) -> Animation:
	for library_name in player.get_animation_library_list():
		var library := player.get_animation_library(library_name)
		if library.has_animation(clip_name):
			return library.get_animation(clip_name)
	return null


static func _idle(data: Dictionary, clips: Dictionary) -> StringName:
	var idle = data.get("locomotion", {}).get("idle")
	if idle != null and clips.has(idle):
		return idle
	for clip_name in clips:
		if clips[clip_name]["loop"]:
			return clip_name
	return clips.keys()[0] if clips.size() > 0 else &""
