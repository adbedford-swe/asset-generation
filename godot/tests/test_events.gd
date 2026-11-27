extends SceneTree
## Checks the timing of an imported assetgen character's clips made from video, headless:
##
##   godot --headless --path <project> --import
##   godot --headless --path <project> --script res://test_events.gd -- res://characters/creature/creature.glb
##
## The character needs a looping clip with contact events and a one-shot with an impact and a sound.
## Prints one line per check and exits 1 if any fails.

var failures := 0
var events: Array = []


func check(ok: bool, what: String) -> void:
	print(("PASS " if ok else "FAIL ") + what)
	if not ok:
		failures += 1


func _initialize() -> void:
	var path: String = OS.get_cmdline_user_args()[0]
	var character = (load(path) as PackedScene).instantiate()
	root.add_child(character)
	check(character is AssetgenCharacter, "the root is an AssetgenCharacter")
	character.clip_event.connect(func(clip, type, joint): events.append([clip, type, joint]))
	var impacts: Array = []
	character.impact.connect(func(clip): impacts.append(clip))
	var steps: Array = []
	character.footstep.connect(func(joint): steps.append(joint))

	var loop := ""
	var once := ""
	for clip_name in character.clips:
		var clip: Dictionary = character.clips[clip_name]
		check(clip["source"] == "video", "%s came from video" % clip_name)
		if clip["loop"] and not clip["events"].is_empty():
			loop = clip_name
		elif not clip["loop"]:
			once = clip_name
	check(loop != "" and once != "", "a looping clip with events and a one-shot")
	var sound := character.get_node_or_null(^"Sound")
	check(sound is AudioStreamPlayer3D, "has a Sound player")
	check(ResourceLoader.exists(character.clips[once]["sound"]), "the one-shot's sound is in the project: %s" % character.clips[once]["sound"])

	character.play(StringName(loop))
	await _seconds(character.clips[loop]["duration"] * 1.5)
	check(steps.size() > 0, "footstep while %s plays (%d)" % [loop, steps.size()])

	var phases: Dictionary = character.clips[once]["phases"]
	check(character.phase_of(StringName(once), float(phases["active"][0]) + 0.01) == "active", "phase_of finds the active phase")
	check(character.phase_of(StringName(once), 0.0) == "windup", "phase_of finds the wind-up")
	character.play(StringName(once))
	await _seconds(0.3)
	check((sound as AudioStreamPlayer3D).stream != null, "the one-shot's sound is loaded as it starts")
	await _seconds(character.clips[once]["duration"])
	check(impacts.has(StringName(once)), "impact from %s" % once)
	check(events.any(func(e): return e[1] == &"impact"), "clip_event carries the impact")

	print("%d failure(s)" % failures)
	quit(1 if failures else 0)


func _seconds(t: float) -> void:
	var end := Time.get_ticks_msec() + int(t * 1000.0)
	while Time.get_ticks_msec() < end:
		await process_frame
