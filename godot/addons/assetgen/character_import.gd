@tool
extends EditorScenePostImport
## Runs on an assetgen asset's .glb as it imports (the .glb.import's import_script). Reads the
## .assetgen file beside it and makes the scene ready to use. A character ("assetgen.character")
## gets:
##
## - every clip looping, or playing once, as the character file says;
## - an AnimationTree: a state machine whose "Locomotion" state blends the idle, walk and run
##   clips by speed (metres per second), with a state for every clip; one-shots return to
##   Locomotion when they end;
## - the AssetgenCharacter script on the root, holding the clips' metadata.
##
## A prop or diorama ("assetgen.prop") gets a StaticBody3D with the collision shape its file asks
## for, and the AssetgenProp script on the root, holding its size and footprint.

const CharacterScript := preload("res://addons/assetgen/assetgen_character.gd")
const PropScript := preload("res://addons/assetgen/assetgen_prop.gd")
const CHARACTER_FORMAT := "assetgen.character"
const PROP_FORMAT := "assetgen.prop"
const LOCOMOTION := &"Locomotion"
const FADE := 0.15


func _post_import(scene: Node) -> Object:
	var path := get_source_file().get_basename() + ".assetgen"
	if not FileAccess.file_exists(path):
		push_warning("assetgen: no assetgen file at %s; imported as a plain model" % path)
		return scene
	var data = JSON.parse_string(FileAccess.get_file_as_string(path))
	if typeof(data) != TYPE_DICTIONARY:
		push_error("assetgen: %s is not an assetgen file" % path)
		return scene
	var format = data.get("format")
	if format != CHARACTER_FORMAT and format != PROP_FORMAT:
		push_warning("assetgen: %s has format '%s', which this addon does not read; imported as a plain model" % [path, format])
		return scene
	var what := "prop" if format == PROP_FORMAT else "character"
	if int(data.get("version", 1)) > 1:
		push_error("assetgen: %s is a newer %s format than this addon reads; update it" % [path, what])
		return scene
	if format == PROP_FORMAT:
		build_prop(scene, data)
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


static func build_prop(root: Node, data: Dictionary) -> void:
	var collider: Dictionary = data.get("collider", {})
	var collider_type := str(collider.get("type", "convex"))
	if not collider_type in ["convex", "mesh", "box", "none"]:
		push_warning("assetgen: %s asks for a '%s' collider, which this addon does not know; using its box" % [root.name, collider_type])
		collider_type = "box"
	var shapes := _prop_shapes(root, collider_type, collider.get("box", {}))
	if collider_type != "none" and shapes.is_empty():
		push_warning("assetgen: %s has no mesh to build a %s collider from" % [root.name, collider_type])
	if not shapes.is_empty():
		var body := StaticBody3D.new()
		body.name = "StaticBody"
		root.add_child(body)
		body.owner = root
		for collision in shapes:
			body.add_child(collision)
			collision.owner = root

	var model: Dictionary = data.get("model", {})
	var footprint: Dictionary = data.get("footprint", {})
	var outline := PackedVector2Array()
	for point in footprint.get("outline", []):
		outline.append(Vector2(float(point[0]), float(point[1])))
	root.set_script(PropScript)
	root.set("kind", str(data.get("kind", "prop")))
	root.set("height", float(model.get("height", 0.0)))
	root.set("size", _vector3(model.get("size", [])))
	root.set("collider", collider_type)
	root.set("footprint_radius", float(footprint.get("radius", 0.0)))
	root.set("base_radius", float(footprint.get("base_radius", 0.0)))
	root.set("outline", outline)
	root.set("prompt", str(data.get("source", {}).get("prompt", "")))
	root.set_meta("assetgen", data)


## One CollisionShape3D per mesh for "convex" and "mesh", placed where the mesh is; a single one for "box".
static func _prop_shapes(root: Node, collider_type: String, box: Dictionary) -> Array[CollisionShape3D]:
	var shapes: Array[CollisionShape3D] = []
	if collider_type == "box":
		var collision := CollisionShape3D.new()
		collision.name = "CollisionShape"
		var shape := BoxShape3D.new()
		shape.size = _vector3(box.get("size", [])).max(Vector3.ONE * 0.001)
		collision.shape = shape
		collision.position = _vector3(box.get("center", []))
		shapes.append(collision)
		return shapes
	if collider_type == "none":
		return shapes
	for node in root.find_children("*", "MeshInstance3D", true, false):
		var mesh: Mesh = node.mesh
		if mesh == null:
			continue
		var collision := CollisionShape3D.new()
		collision.name = "CollisionShape" if shapes.is_empty() else "CollisionShape%d" % (shapes.size() + 1)
		collision.shape = mesh.create_convex_shape() if collider_type == "convex" else mesh.create_trimesh_shape()
		collision.transform = _transform_in(root, node)
		shapes.append(collision)
	return shapes


## The node's transform relative to root; the imported scene is not in a tree yet, so there is no global transform.
static func _transform_in(root: Node, node: Node3D) -> Transform3D:
	var transform := Transform3D.IDENTITY
	var current: Node = node
	while current != root and current is Node3D:
		transform = (current as Node3D).transform * transform
		current = current.get_parent()
	return transform


static func _vector3(values: Array) -> Vector3:
	return Vector3(float(values[0]), float(values[1]), float(values[2])) if values.size() == 3 else Vector3.ZERO
