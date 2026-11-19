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
##
## A scene ("assetgen.scene") gets a StaticBody3D under its ground node and under every placement
## node, with the shape its object asks for (placements of one object share their shapes), and the
## AssetgenScene script on the root, which maps placements to objects.

const CharacterScript := preload("res://addons/assetgen/assetgen_character.gd")
const PropScript := preload("res://addons/assetgen/assetgen_prop.gd")
const SceneScript := preload("res://addons/assetgen/assetgen_scene.gd")
const CHARACTER_FORMAT := "assetgen.character"
const PROP_FORMAT := "assetgen.prop"
const SCENE_FORMAT := "assetgen.scene"
const COLLIDERS := ["convex", "mesh", "box", "none"]
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
	if not format in [CHARACTER_FORMAT, PROP_FORMAT, SCENE_FORMAT]:
		push_warning("assetgen: %s has format '%s', which this addon does not read; imported as a plain model" % [path, format])
		return scene
	var what: String = {PROP_FORMAT: "prop", SCENE_FORMAT: "scene"}.get(format, "character")
	if int(data.get("version", 1)) > 1:
		push_error("assetgen: %s is a newer %s format than this addon reads; update it" % [path, what])
		return scene
	if format == PROP_FORMAT:
		build_prop(scene, data)
		return scene
	if format == SCENE_FORMAT:
		build_scene(scene, data)
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
	var collider_type := _collider_type(str(collider.get("type", "convex")), root.name)
	_add_body(root, root, collider_type, collider.get("box", {}), {})

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


static func build_scene(root: Node, data: Dictionary) -> void:
	var nodes := {}
	for node in root.find_children("*", "Node3D", true, false):
		if not nodes.has(String(node.name)):
			nodes[String(node.name)] = node
	var objects := {}
	for object in data.get("objects", []):
		objects[str(object.get("id"))] = object

	var placements := {}
	var ground_name := str(data.get("ground", {}).get("node", "ground"))
	if nodes.has(ground_name):
		_add_body(root, nodes[ground_name], "mesh", {}, {})
	else:
		push_warning("assetgen: %s has no ground node '%s'" % [root.name, ground_name])
	var cache := {}
	for instance in data.get("instances", []):
		var node_name := str(instance.get("node"))
		var object_id := str(instance.get("object"))
		if not nodes.has(node_name) or not objects.has(object_id):
			push_warning("assetgen: %s skips the placement '%s' of '%s': there is no such node or object" % [root.name, node_name, object_id])
			continue
		var object: Dictionary = objects[object_id]
		var collider_type := _collider_type(str(object.get("collider", "convex")), object_id)
		_add_body(root, nodes[node_name], collider_type, object.get("box", {}), cache)
		placements[String(root.get_path_to(nodes[node_name]))] = object_id

	root.set_script(SceneScript)
	root.set("size", _vector2(data.get("model", {}).get("size", [])))
	root.set("ground_path", root.get_path_to(nodes[ground_name]) if nodes.has(ground_name) else NodePath())
	root.set("placements", placements)
	root.set_meta("assetgen", data)


static func _collider_type(collider_type: String, what: String) -> String:
	if collider_type in COLLIDERS:
		return collider_type
	push_warning("assetgen: %s asks for a '%s' collider, which this addon does not know; using its box" % [what, collider_type])
	return "box"


## A StaticBody3D under node with the collision shapes collider_type asks for, owned by root so they are saved;
## nothing for "none". Shapes made from one mesh are shared through cache.
static func _add_body(root: Node, node: Node, collider_type: String, box: Dictionary, cache: Dictionary) -> void:
	var shapes := _shapes(node, collider_type, box, cache)
	if collider_type != "none" and shapes.is_empty():
		push_warning("assetgen: %s has no mesh to build a %s collider from" % [node.name, collider_type])
	if shapes.is_empty():
		return
	var body := StaticBody3D.new()
	body.name = "StaticBody"
	node.add_child(body)
	body.owner = root
	for collision in shapes:
		body.add_child(collision)
		collision.owner = root


## One CollisionShape3D per mesh (node's own, and its children's) for "convex" and "mesh", placed where the
## mesh is; a single one for "box".
static func _shapes(node: Node, collider_type: String, box: Dictionary, cache: Dictionary) -> Array[CollisionShape3D]:
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
	var meshes := node.find_children("*", "MeshInstance3D", true, false)
	if node is MeshInstance3D:
		meshes.push_front(node)
	for mesh_node in meshes:
		var mesh: Mesh = mesh_node.mesh
		if mesh == null:
			continue
		var key := [mesh, collider_type]
		if not cache.has(key):
			cache[key] = mesh.create_convex_shape() if collider_type == "convex" else mesh.create_trimesh_shape()
		var collision := CollisionShape3D.new()
		collision.name = "CollisionShape" if shapes.is_empty() else "CollisionShape%d" % (shapes.size() + 1)
		collision.shape = cache[key]
		collision.transform = _transform_in(node, mesh_node)
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


static func _vector2(values: Array) -> Vector2:
	return Vector2(float(values[0]), float(values[1])) if values.size() == 2 else Vector2.ZERO


static func _vector3(values: Array) -> Vector3:
	return Vector3(float(values[0]), float(values[1]), float(values[2])) if values.size() == 3 else Vector3.ZERO
