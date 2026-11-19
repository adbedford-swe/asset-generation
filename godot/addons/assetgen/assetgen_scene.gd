@tool
class_name AssetgenScene
extends Node3D
## An assetgen scene, as imported: the ground and every placed object, each with its collision shape
## on a "StaticBody" child. Every object is also in the scene's assets/ folder as a prop of its own.
##
## The scene file is in the "assetgen" metadata; [method object_of] reads an object's entry from it
## (id, kind, height, collider, box, footprint, count).

## The ground's width (x) and depth (z) in metres, centred on the origin.
@export var size := Vector2.ZERO
@export var ground_path := NodePath()
## Each placement node's path from here -> its object's id.
@export var placements: Dictionary = {}


func ground() -> Node3D:
	return get_node_or_null(ground_path) as Node3D


## The placement nodes of an object, by its id.
func instances_of(object_id: String) -> Array[Node3D]:
	var found: Array[Node3D] = []
	for path in placements:
		if placements[path] == object_id:
			var node := get_node_or_null(NodePath(path)) as Node3D
			if node != null:
				found.append(node)
	return found


## The object a node belongs to (a placement node, or anything under one); empty when it is none.
func object_of(node: Node) -> Dictionary:
	var current := node
	while current != null and current != self:
		var object_id = placements.get(String(get_path_to(current)))
		if object_id != null:
			return object(object_id)
		current = current.get_parent()
	return {}


## An object's entry in the scene file, by its id; empty when there is none.
func object(object_id: String) -> Dictionary:
	for entry in get_meta("assetgen", {}).get("objects", []):
		if entry.get("id") == object_id:
			return entry
	return {}


func object_ids() -> Array[String]:
	var ids: Array[String] = []
	for entry in get_meta("assetgen", {}).get("objects", []):
		ids.append(str(entry.get("id")))
	return ids
