extends SceneTree
## Checks an imported assetgen scene in a Godot project, headless:
##
##   godot --headless --path <project> --import
##   godot --headless --path <project> --script res://test_scene.gd -- res://environments/camp/camp.glb
##
## Checked against the scene's own .assetgen file. Prints one line per check and exits 1 if any fails.

const SHAPES := {"convex": "ConvexPolygonShape3D", "mesh": "ConcavePolygonShape3D", "box": "BoxShape3D"}

var failures := 0


func check(ok: bool, what: String) -> void:
	print(("PASS " if ok else "FAIL ") + what)
	if not ok:
		failures += 1


func _initialize() -> void:
	var path: String = OS.get_cmdline_user_args()[0] if OS.get_cmdline_user_args().size() > 0 else "res://environments/camp/camp.glb"
	var packed: PackedScene = load(path)
	check(packed != null, "loads %s" % path)
	if packed == null:
		quit(1)
		return
	var scene = packed.instantiate()
	check(scene is AssetgenScene, "the root is an AssetgenScene")
	if not scene is AssetgenScene:
		quit(1)
		return
	root.add_child(scene)
	var data: Dictionary = scene.get_meta("assetgen", {})
	check(data.get("format") == "assetgen.scene", "the scene file is in the root's metadata")

	var ground: Node3D = scene.ground()
	check(ground != null and String(ground.name) == data["ground"]["node"], "the ground is %s" % data["ground"]["node"])
	check(_shape_classes(ground) == ["ConcavePolygonShape3D"], "the ground has a trimesh shape")

	for object in data["objects"]:
		var id: String = object["id"]
		var nodes: Array[Node3D] = scene.instances_of(id)
		check(nodes.size() == int(object["count"]), "%s: %d placement(s)" % [id, nodes.size()])
		var collider: String = object["collider"]
		var right := not nodes.is_empty()
		var owned := true
		for node in nodes:
			right = right and scene.object_of(node).get("id") == id
			var want := [] if collider == "none" else [SHAPES[collider]]
			right = right and _shape_classes(node) == want
			for child in node.find_children("*", "", true, false):
				owned = owned and child.owner == scene
		check(right, "%s: every placement has %s" % [id, "no collider" if collider == "none" else "a " + SHAPES[collider]])
		check(owned, "%s: the bodies are saved with the scene" % id)
		if collider == "box" and not nodes.is_empty():
			var shape = nodes[0].get_node(^"StaticBody").get_child(0)
			var box: Dictionary = object["box"]
			check(shape.shape.size.is_equal_approx(Vector3(box["size"][0], box["size"][1], box["size"][2]))
				and shape.position.is_equal_approx(Vector3(box["center"][0], box["center"][1], box["center"][2])),
				"%s: the box is the object's own, scaled with its placement" % id)
		if not nodes.is_empty():
			var body := nodes[0].get_node_or_null(^"StaticBody")
			check(body == null or scene.object_of(body).get("id") == id, "%s: object_of finds it from its body" % id)

	await physics_frame
	await physics_frame
	var space: PhysicsDirectSpaceState3D = scene.get_world_3d().direct_space_state
	var hit := _ray(space, _clearing(scene))
	check(not hit.is_empty() and hit["collider"].get_parent() == ground, "a ray from above lands on the ground")

	for object in data["objects"]:
		if object["collider"] == "none":
			continue
		var landed := false
		for node in scene.instances_of(object["id"]):
			hit = _ray(space, node.global_position)
			if not hit.is_empty() and hit["collider"].get_parent() == node:
				var top: float = node.global_position.y + float(object["height"]) * node.global_basis.get_scale().y
				landed = hit["position"].y <= top + 0.05 and hit["position"].y > node.global_position.y
				break
		check(landed, "a ray from above lands on a %s" % object["id"])
	print("%d failure(s)" % failures)
	quit(1 if failures else 0)


func _ray(space: PhysicsDirectSpaceState3D, at: Vector3) -> Dictionary:
	return space.intersect_ray(PhysicsRayQueryParameters3D.create(Vector3(at.x, 100.0, at.z), Vector3(at.x, -100.0, at.z)))


## The point of the ground farthest from every placement.
func _clearing(scene: AssetgenScene) -> Vector3:
	var best := Vector3.ZERO
	var best_distance := -1.0
	for i in 21:
		for j in 21:
			var at := Vector3((i / 20.0 - 0.5) * scene.size.x * 0.9, 0.0, (j / 20.0 - 0.5) * scene.size.y * 0.9)
			var nearest := INF
			for path in scene.placements:
				var node: Node3D = scene.get_node(NodePath(path))
				nearest = minf(nearest, Vector2(at.x - node.global_position.x, at.z - node.global_position.z).length())
			if nearest > best_distance:
				best = at
				best_distance = nearest
	return best


func _shape_classes(node: Node) -> Array:
	var classes := []
	var body := node.get_node_or_null(^"StaticBody") if node != null else null
	if body is StaticBody3D:
		for collision in body.get_children():
			if collision is CollisionShape3D and collision.shape != null:
				classes.append(collision.shape.get_class())
	return classes
