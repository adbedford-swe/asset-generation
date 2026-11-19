@tool
extends EditorPlugin
## Makes sure every assetgen asset (a character, a prop or a diorama) imports through
## character_import.gd, including a .glb and .assetgen dropped into the project by hand
## (`assetgen export --engine godot` already writes the .glb.import that does this).

const IMPORT_SCRIPT := "res://addons/assetgen/character_import.gd"


func _enter_tree() -> void:
	var filesystem := EditorInterface.get_resource_filesystem()
	filesystem.resources_reimported.connect(_on_reimported)
	filesystem.filesystem_changed.connect(_scan)
	add_tool_menu_item("Assetgen: set up assets", _scan)
	_scan.call_deferred()


func _exit_tree() -> void:
	var filesystem := EditorInterface.get_resource_filesystem()
	if filesystem.resources_reimported.is_connected(_on_reimported):
		filesystem.resources_reimported.disconnect(_on_reimported)
	if filesystem.filesystem_changed.is_connected(_scan):
		filesystem.filesystem_changed.disconnect(_scan)
	remove_tool_menu_item("Assetgen: set up assets")


func _on_reimported(paths: PackedStringArray) -> void:
	_fix(Array(paths))


func _scan() -> void:
	_fix(_find(EditorInterface.get_resource_filesystem().get_filesystem()))


func _find(dir: EditorFileSystemDirectory) -> Array:
	var found := []
	for i in dir.get_file_count():
		var path := dir.get_file_path(i)
		if path.get_extension().to_lower() in ["glb", "gltf"]:
			found.append(path)
	for i in dir.get_subdir_count():
		found.append_array(_find(dir.get_subdir(i)))
	return found


## Point each assetgen .glb's import at the import script and reimport the ones changed.
func _fix(paths: Array) -> void:
	var changed := PackedStringArray()
	for path in paths:
		if not path.get_extension().to_lower() in ["glb", "gltf"]:
			continue
		if not FileAccess.file_exists(path.get_basename() + ".assetgen"):
			continue
		var config := ConfigFile.new()
		if config.load(path + ".import") != OK:
			continue
		if config.get_value("params", "import_script/path", "") == IMPORT_SCRIPT:
			continue
		config.set_value("params", "import_script/path", IMPORT_SCRIPT)
		config.save(path + ".import")
		changed.append(path)
	if not changed.is_empty():
		EditorInterface.get_resource_filesystem().reimport_files(changed)
