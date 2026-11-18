extends SceneTree
## Checks an imported assetgen character in a Godot project, headless:
##
##   godot --headless --path <project> --import
##   godot --headless --path <project> --script res://test_character.gd -- res://characters/fox/fox.glb
##
## Prints one line per check and exits 1 if any fails.

var failures := 0


func check(ok: bool, what: String) -> void:
	print(("PASS " if ok else "FAIL ") + what)
	if not ok:
		failures += 1


func _initialize() -> void:
	var path: String = OS.get_cmdline_user_args()[0] if OS.get_cmdline_user_args().size() > 0 else "res://characters/fox/fox.glb"
	var packed: PackedScene = load(path)
	check(packed != null, "loads %s" % path)
	if packed == null:
		quit(1)
		return
	var character = packed.instantiate()
	root.add_child(character)
	check(character is AssetgenCharacter, "the root is an AssetgenCharacter")
	check(character.clips.size() > 0, "clips: %s" % ", ".join(character.clips.keys()))

	var player: AnimationPlayer = character.find_children("*", "AnimationPlayer", true, false)[0]
	for clip_name in character.clips:
		var animation := player.get_animation(clip_name)
		var want := Animation.LOOP_LINEAR if character.clips[clip_name]["loop"] else Animation.LOOP_NONE
		check(animation != null and animation.loop_mode == want,
			"%s %s" % [clip_name, "loops" if want == Animation.LOOP_LINEAR else "plays once"])

	var tree: AnimationTree = character.get_node_or_null(^"AnimationTree")
	check(tree != null and tree.active, "has an active AnimationTree")
	var machine := tree.tree_root as AnimationNodeStateMachine
	check(machine != null and machine.has_node(&"Locomotion"), "the state machine has Locomotion")
	for clip_name in character.clips:
		check(machine.has_node(clip_name), "the state machine has %s" % clip_name)

	var skeleton: Skeleton3D = character.find_children("*", "Skeleton3D", true, false)[0]
	var bone := skeleton.get_bone_count() - 1
	await process_frame
	var before := skeleton.get_bone_global_pose(bone).origin

	character.speed = character.speed_of("walk") if character.speed_of("walk") > 0.0 else 0.5
	await _seconds(0.5)
	check(is_equal_approx(tree.get(&"parameters/Locomotion/blend_position"), character.speed) or character.locomotion_max_speed == 0.0,
		"Speed drives the locomotion blend (%.2f m/s)" % character.speed)
	check(skeleton.get_bone_global_pose(bone).origin.distance_to(before) > 1e-4, "the skeleton moves while walking")

	var playback: AnimationNodeStateMachinePlayback = tree.get(&"parameters/playback")
	var one_shot := ""
	for clip_name in character.clips:
		if not character.clips[clip_name]["loop"]:
			one_shot = clip_name
	if one_shot != "":
		check(character.play(one_shot), "play(%s)" % one_shot)
		await _seconds(0.4)
		check(playback.get_current_node() == StringName(one_shot), "%s is playing" % one_shot)
		await _seconds(character.clips[one_shot]["duration"] + 0.6)
		check(playback.get_current_node() == &"Locomotion", "%s returns to Locomotion when it ends" % one_shot)
	check(not character.play(&"no_such_clip"), "an unknown clip is refused")

	print("%d failure(s)" % failures)
	quit(1 if failures else 0)


func _seconds(t: float) -> void:
	var end := Time.get_ticks_msec() + int(t * 1000.0)
	while Time.get_ticks_msec() < end:
		await process_frame
