-- start logging
logname = localTime
logname = substituteString logname ":" "_"
logname = substituteString logname "/" "_"
openLog (save_folder + "\\3dslightcalc_log_" + logname + ".txt")


format "%" save_folder

assets_path = base_path + @"\assets\"

filein (base_path + @"\maxscript_util.ms") quiet:true
filein (base_path + @"\project_init.ms")  quiet:true
filein (base_path + @"\room_handling.ms") quiet:true
filein (base_path + @"\luminaire_handling.ms") quiet:true
filein (base_path + @"\main.ms") quiet:true

save_folder = save_folder + @"\"

if all_luminaires == "" or all_luminaires == "true" then 
(
	all_luminaires = true;
)
else
(
	if all_luminaires == "false" then all_luminaires = false;
)

if (x_size == "" or y_size == "" ) then
(
	x_size = y_size = 128;
)
else
(
	x_size = x_size as integer;
	y_size = y_size as integer;
)

if WZ == "" then 
(
	WZ = 50;
)
else
(
	WZ = WZ as integer;
) 

rendering_pres = assets_path + @"mental_ray_light_analysis_fast_box.rps"
convertor_lib = (getfiles_ assets_path "xaml")[1]
ies_files_array = getfiles_ assets_path "ies" + getfiles_ (getFilenamePath room_file) "ies"
max_lib = getfiles_ (assets_path + "/max2011/") "max"

units.SystemType = #millimeters;
units.DisplayType = #metric;

format "Starting calculations...\n"
format "TIME: %\n" localTime;
main room_file save_folder rendering_pres ies_files_array xaml:convertor_lib func_data_file:pos_file furn_path_array:max_lib render_all:all_luminaires Nx:x_size Ny:y_size wall_zone:WZ
format "TIME: %\n" localTime;

flushLog();
closeLog();
