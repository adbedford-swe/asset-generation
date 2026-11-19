@tool
class_name AssetgenProp
extends Node3D
## An assetgen prop or diorama, as imported: its size and footprint, for game code and scene
## layouts that place assets without them overlapping. The collision shape, if any, is on the
## "StaticBody" child.
##
## The origin is the centre of the asset's base, on the ground; its front faces +Z
## ([constant Vector3.MODEL_FRONT]), like every glTF model. All lengths are in metres, at the scale
## the asset was imported at.

## "prop" (one object) or "diorama" (a whole set piece).
@export var kind := "prop"
@export var height := 0.0
## Width (x), height (y) and depth (z).
@export var size := Vector3.ZERO
## "convex", "mesh", "box" or "none".
@export var collider := "convex"
## The farthest the asset reaches from its origin, seen from above.
@export var footprint_radius := 0.0
## The same for its base (the bottom tenth of its height): a tree's trunk rather than its canopy.
@export var base_radius := 0.0
## The outline seen from above, as (x, z) points.
@export var outline := PackedVector2Array()
@export_multiline var prompt := ""


func is_diorama() -> bool:
	return kind == "diorama"
