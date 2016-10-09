fileIn "max2unity_lib.ms" quiet:true

actionMan.executeAction 0 "40472"  -- MAX Script: MAXScript Listener

script_root = @"O:\scripts\MAX2Unity\"

input_folder = @"Z:\Asset_dump\max\"
output_folder =  @"O:\_sandbox\Auto_MAX2Unity\output\"

input_files = getFiles (input_folder + "*.max")

for f in input_files do
(
	loadMaxFile (script_root + "init.max") useFileUnits:true quiet:true
	mergeMaxFile f quiet:true

	delete helpers
	delete lights

	delete $cs_* -- delete cut shapes
	
	objects_to_process = geometry as array

	ungroup_all objects_to_process

	vray_to_standard objects_to_process

	reduce_polygons objects_to_process
	
	objects_to_process = geometry as array

	attach_all objects_to_process

	delete_invisible_faces $merged_mesh
	
	exportFile (output_folder + GetFileNameFile f + ".FBX") #noPrompt using:FBXEXP
)