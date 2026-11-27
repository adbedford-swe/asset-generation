@tool
class_name AssetgenCharacter
extends Node3D
## Plays an assetgen character. Set [member speed] (metres per second) and the locomotion blend
## picks idle, walk or run at the pace the clips were made for; call [method play] for anything
## else. One-shot clips return to locomotion when they end.
##
## Characters face +Z ([constant Vector3.MODEL_FRONT]), like every glTF model.
##
## Clips made from video carry their timing: [signal footstep] when part of the body comes to rest
## on the ground, [signal impact] when a one-shot's action lands (hit boxes, damage, camera shake),
## and [signal clip_event] for both. Their sounds play from a "Sound" child as the clip starts.

## Any event of the playing clips: type is "contact" or "impact"; joint names the bone on the ground.
signal clip_event(clip: StringName, type: StringName, joint: StringName)
## Part of the body came to rest on the ground, carried by this joint.
signal footstep(joint: StringName)
## A one-shot's action lands: see the clip's "phases" in [member clips] for its wind-up and recovery.
signal impact(clip: StringName)

## Ground speed in metres per second. Drives the "Locomotion" blend.
@export_range(0.0, 20.0, 0.01, "or_greater", "suffix:m/s") var speed := 0.0:
	set(value):
		speed = maxf(value, 0.0)

## Move forward at [member speed]. Handy for previews and simple NPCs; leave it off when a
## CharacterBody3D or NavigationAgent3D moves the character.
@export var move_forward := false

## What each clip is for: name -> {role, loop, duration, speed, prompt, source, events, phases, sound}.
## Filled on import.
@export var clips: Dictionary = {}
@export var locomotion_max_speed := 0.0
@export var character_height := 0.0
@export_multiline var prompt := ""

var _tree: AnimationTree
var _playback: AnimationNodeStateMachinePlayback


func _ready() -> void:
	_tree = get_node_or_null(^"AnimationTree")
	if _tree != null:
		_playback = _tree.get(&"parameters/playback")


func _process(delta: float) -> void:
	if Engine.is_editor_hint() or _tree == null:
		return
	if locomotion_max_speed > 0.0:
		_tree.set(&"parameters/Locomotion/blend_position", speed)
	if move_forward:
		position += global_basis.z.normalized() * speed * delta


## Cross-fade to a clip by name (see [member clips]), or to "Locomotion". Returns false when there is
## no such clip.
func play(clip: StringName) -> bool:
	if _playback == null or (clip != &"Locomotion" and not clips.has(clip)):
		push_warning("%s has no clip '%s'" % [name, clip])
		return false
	_playback.travel(clip)
	return true


## Back to the locomotion blend, at [param metres_per_second].
func locomote(metres_per_second: float) -> void:
	speed = metres_per_second
	play(&"Locomotion")


## The phase of a one-shot at [param time] seconds into it: "windup", "active" or "recovery"; "" for
## clips without phases.
func phase_of(clip: StringName, time: float) -> String:
	var phases: Dictionary = clips.get(clip, {}).get("phases", {})
	for phase in ["windup", "active", "recovery"]:
		var span: Array = phases.get(phase, [])
		if span.size() == 2 and time >= float(span[0]) and time <= float(span[1]):
			return phase
	return ""


func _assetgen_event(clip: String, type: String, joint: String) -> void:
	clip_event.emit(StringName(clip), StringName(type), StringName(joint))
	if type == "contact":
		footstep.emit(StringName(joint))
	elif type == "impact":
		impact.emit(StringName(clip))


func _assetgen_sound(clip: String) -> void:
	var sound := get_node_or_null(^"Sound") as AudioStreamPlayer3D
	var path: String = clips.get(clip, {}).get("sound", "")
	if sound == null or path == "" or not ResourceLoader.exists(path):
		return
	sound.stream = load(path)
	sound.play()


## The speed the clip with this role (walk, run) was made at; 0 without one.
func speed_of(role: String) -> float:
	for clip_name in clips:
		if clips[clip_name]["role"] == role:
			return clips[clip_name]["speed"]
	return 0.0
