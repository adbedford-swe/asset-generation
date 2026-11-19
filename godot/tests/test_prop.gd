extends SceneTree
## Checks imported assetgen props and dioramas in a Godot project, headless:
##
##   godot --headless --path <project> --import
##   godot --headless --path <project> --script res://test_prop.gd -- res://characters/barrel/barrel.glb ...
##
## Each asset is checked against its own .assetgen file. Prints one line per check and exits 1 if
## any fails.

const SHAPES := {"convex": "ConvexPolygonShape3D", "mesh": "ConcavePolygonShape3D", "box": "BoxShape3D"}
const SPACING := 20.0

var failures := 0


func check(ok: bool, what: String) -> void:
	print(("PASS " if ok else "FAIL ") + what)
	if not ok:
		failures += 1


func _initialize() -> void:
	var paths := OS.get_cmdline_user_args()
	if paths.is_empty():
		paths = PackedStringArray(["res://characters/barrel/barrel.glb"])
	var props := []
	for i in paths.size():
		props.append(_place(paths[i], Vector3(SPACING * i, 0.0, 0.0)))
	await physics_frame
	await physics_frame
	for i in paths.size():
		if props[i] != null:
			_check_collision(props[i], Vector3(SPACING * i, 0.0, 0.0))
	print("%d failure(s)" % failures)
	quit(1 if failures else 0)


func _place(path: String, at: Vector3) -> Node3D:
	var packed: PackedScene = load(path)
	check(packed != null, "loads %s" % path)
	if packed == null:
		return null
	var prop = packed.instantiate()
	var label := String(prop.name)
	check(prop is AssetgenProp, "%s: the root is an AssetgenProp" % label)
	if not prop is AssetgenProp:
		prop.free()
		return null
	var data: Dictionary = prop.get_meta("assetgen", {})
	check(data.get("format") == "assetgen.prop", "%s: the prop file is in the root's metadata" % label)
	var model: Dictionary = data.get("model", {})
	check(prop.kind == data.get("kind") and is_equal_approx(prop.height, float(model.get("height", -1.0))),
		"%s: a %s, %.2f m tall" % [label, prop.kind, prop.height])
	check(prop.collider == data.get("collider", {}).get("type") and prop.footprint_radius > 0.0
		and prop.base_radius > 0.0 and prop.outline.size() >= 3,
		"%s: collider %s, footprint radius %.2f m, base radius %.2f m" % [label, prop.collider, prop.footprint_radius, prop.base_radius])

	var bounds := _bounds(prop)
	check(absf(bounds.position.y) < 1e-3, "%s sits on y = 0 (lowest point %.4f)" % [label, bounds.position.y])
	check(absf(bounds.size.y - prop.height) < 1e-3, "%s's mesh is as tall as its file says" % label)
	check(Vector2(bounds.get_center().x, bounds.get_center().z).length() < 1e-3, "%s is centred on its origin" % label)

	var bodies := prop.find_children("*", "StaticBody3D", true, false)
	if prop.collider == "none":
		check(bodies.is_empty(), "%s has no collider" % label)
	else:
		var body: StaticBody3D = prop.get_node_or_null(^"StaticBody")
		check(body != null and bodies.size() == 1 and body.owner == prop, "%s has one StaticBody, saved with the scene" % label)
		var shapes: Array = body.find_children("*", "CollisionShape3D", false, false) if body != null else []
		var right := not shapes.is_empty()
		for collision in shapes:
			right = right and collision.owner == prop and collision.shape != null \
				and collision.shape.get_class() == SHAPES[prop.collider]
		check(right, "%s's collision shape is a %s" % [label, SHAPES[prop.collider]])
		if prop.collider == "box" and not shapes.is_empty():
			var box: Dictionary = data["collider"]["box"]
			var size := Vector3(box["size"][0], box["size"][1], box["size"][2])
			var center := Vector3(box["center"][0], box["center"][1], box["center"][2])
			check(shapes[0].shape.size.is_equal_approx(size) and shapes[0].position.is_equal_approx(center),
				"%s's box is %s at %s" % [label, size, center])
	root.add_child(prop)
	prop.position = at
	return prop


## A ray from high above the origin lands on the top of a prop with a collider, and falls through one without.
func _check_collision(prop: AssetgenProp, at: Vector3) -> void:
	var space := prop.get_world_3d().direct_space_state
	var hit := space.intersect_ray(PhysicsRayQueryParameters3D.create(at + Vector3.UP * 50.0, at + Vector3.DOWN * 50.0))
	if prop.collider == "none":
		check(hit.is_empty(), "%s: nothing to hit" % prop.name)
	else:
		check(not hit.is_empty() and absf(hit["position"].y - prop.height) < 0.02,
			"%s: a ray from above lands on its top (%s)" % [prop.name, hit.get("position", "no hit")])


func _bounds(prop: Node3D) -> AABB:
	var bounds := AABB()
	var first := true
	for node in prop.find_children("*", "MeshInstance3D", true, false):
		var transform := Transform3D.IDENTITY
		var current: Node = node
		while current != prop:
			transform = current.transform * transform
			current = current.get_parent()
		var box: AABB = transform * node.get_aabb()
		bounds = box if first else bounds.merge(box)
		first = false
	return bounds
