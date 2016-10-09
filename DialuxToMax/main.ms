fn main room_file_path save_folder rendering_preset ies_files_array xaml: func_data_file: furn_path_array:max_lib Nx:128 Ny:128 wall_zone:50 = 
(
	local standart_struct = #() -- STRUCTURES (SETS/GROUPS) OF LUMINAIRES IN THE ROOM OR LUMINAIRE FILE;
 	local file_lum_info = #()   -- INFORMATION OF THE LUMINAIRE INSIDE THE ROOM OR LUMINAIRE FILE;
	local room_data = #()		-- DATA OF THE ROOM FILE;
	local lum_info = #()		-- DATA DEFINING THE LUMINAIRE AND IT'S ASSOCIATION TO AN IES PROFILE;
	local nr_lum_struct			-- 
	local nr_lums   			--

	------ SETTING UP THE RENDERING ----------

	renderpresets.LoadAll 0 (rendering_preset)
	--cur_renderer = renderers.current
	--cur_renderer.UnifiedEnable = false
	--cur_renderer.MinimumSamples = 0
	--cur_renderer.MaximumSamples = 1
	reM = maxOps.GetCurRenderElementMgr()
	reM.AddRenderElement (Illuminance_HDR_Data elementname:"Illuminance_HDR_Data" enabledOn:true scaleFactor:1.0 filterOn:true)

	------- GRID FIX (UNFINISHED) ------------

	--if (Nx < 20 or Ny < 20) then (cur_renderer.MinimumSamples = 5; cur_renderer.MaximumSamples = 5) else (cur_renderer.MinimumSamples = 1; cur_renderer.MaximumSamples = 2);

	------ INTIALIZING THE PROJECT -----------

	project = project_builder()
	project.project_init room_file_path func_data_file &file_lum_info &room_data &nr_lum_struct &nr_lums &lum_info

	------ BUILDING THE ROOM -----------------

	room = room_builder()
	room.build_room room_data read_lut_stream:xaml furn_models:furn_path_array 
	room.create_camera wall_zone Nx Ny
	
	------ HANDLING THE LUMINAIRES -----------

	luminaires = lum_handling()
	luminaires.place_luminaire_dummies file_lum_info nr_lums nr_lum_struct &standart_struct
	luminaires.add_ies_lights ies_files_array lum_info &standart_struct ma_factor:room.MF

	-------	RENDERING LOOP -------------------

	if func_data_file != unsupplied then
	(
		m = 1;
		for structure in standart_struct do
		(
			for lum in structure do lum.on = true;
			base_filename = save_folder + (getFilenameFile room_file_path) + "_" + m as string;
			render_image  base_filename;
			for lum in structure do lum.on = false;
			m += 1;
		)
		freescenebitmaps();
		clearundobuffer();
		resetMaxFile #noPrompt;
	)
)