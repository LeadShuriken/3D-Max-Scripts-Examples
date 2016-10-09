/*
THIS IS A SET OF FUNCTIONS THAT ARE NOT FIT IN THE STRUCTURES AND CAN FUNCTION INDEPENDEDLY OF THE PROJETC
*/

fn file_to_array file = 
(
	data = #()
	if file != unsupplied then 
	(
		fin = openFile file mode:"r"
		while not(eof fin) do
		(
			append data (readline fin)
		)
		close fin
	)
	data
)

mapped fn getfiles_ folder type =
(
	if (folder != undefined) do
	(
		folder += "\\*." + type
		passed_array = getFiles folder
		sort passed_array
		format "In % the following files have been found: %\n" folder passed_array
		passed_array
	)
)

fn render_image filename = 
(
	filename_hdr = filename + ".hdr"
	
	/*
	filename_max = filename + ".max"
	filename_csv = filename + ".csv"
	filename_png = filename + ".png"
	
	save_filename = filename_max
	saveMaxFile save_filename quiet:true saveAsVersion:2013
	*/

	reM = maxOps.GetCurRenderElementMgr()
	rendFilename = filename_hdr
	reM.SetRenderElementFilename 0 rendFilename
	
	setsilentmode true
	outputHDRbitmap

	rendFieldOrder = 1
	rendFieldRender = false
	
	max vpt camera
	max quick render

	/*
	if(doesFileExist filename_csv) do ((dotNetClass "System.IO.File").WriteAllLines filename_csv #())
	render_El = #()
	render_El[1] = openBitmap filename_hdr
	fs = openFile filename_csv mode:"w"
	for r = 0 to (renderheight - 1) do
	(
		pixel_row_array = getPixels render_El[1] [0,r] renderWidth linear:false
		pixel_row_string = ""
		for col_pix in pixel_row_array do 
		(
			val = (col_pix.r + col_pix.g + col_pix.b)/(3*255)
			pixel_row_string = pixel_row_string + ((val as string) + " ,");
		)
		pixel_row_string = substring pixel_row_string 1 (pixel_row_string.count - 2)
		format (pixel_row_string + "\n") to:fs
	)
	close fs
	*/
)

mapped fn crop_array_input start_pattern end_pattern full_return_array =
(
	start_index = 0;
	end_index = 0;
	for i = 1 to full_return_array.count do
	(
		if matchpattern full_return_array[i] pattern:start_pattern do 
		(
			start_index = i;
			for m = i to full_return_array.count do
			(
				if end_pattern == "to_end" then 
				(
					end_index = full_return_array.count;
				)
				else 
				(
					if matchpattern full_return_array[m] pattern:end_pattern do end_index = m - 1;
				)
			)
		)
	)
	full_return_array = (for i = start_index to end_index collect full_return_array[i]);
)

fn sort_and_import_obj MAX_path LUT_file_lines =
(
	dia_name = #();
	max_name = #();

	max_files = getfiles(MAX_path + "\\" + "*.max");

	dia_temp = filterstring LUT_file_lines[2] "<>";
	dia_temp[dia_temp.count-1];

	for max_file in max_files do
	(
		for l = 2 to (LUT_file_lines.count - 1) do
		(
			max_temp = filterstring LUT_file_lines[l] "\"";
			max_temp = max_temp[max_temp.count-1];
			
			if ((findstring max_temp ((getfilenamefile max_file) + "_m3dName")) != undefined) do
			(
				dia_temp = filterstring LUT_file_lines[l] "<>";
				dia_temp = dia_temp[dia_temp.count-1];
				append dia_name dia_temp;
				append max_name max_file;
			)
		)
	)

	boxes = objects as array
	for b in boxes do
	(
		object_dianame = (filterstring b.name "_")[2];
		libraryindex = finditem dia_name object_dianame;
		if libraryindex != 0 do
		(
			mergemaxfile max_name[libraryindex] #select #AutoRenameDups #useSceneMtlDups #neverReparent quite:true
			merged = selection as array
			for obj in merged do merged.mat = Architectural() 
			if merged.count > 1 do
			(
				group merged name:b.name select:true
				merged = selection as array
				merged = for m in merged where classof m == Dummy collect m -- get the group dummy
				if (distance selection.max selection.min < 100) do -- scale object that have wrong units
				(
					scale merged[1] [1000, 1000, 1000]
				)
			)
			
			merged = merged[1]
			merged.name = b.name
				
			scale merged [unit/1000, unit/1000, unit/1000 ]
				
			merged.position = b.position
			in coordsys merged merged.rotation = b.rotation
			in coordsys local rotate merged (angleaxis 90 [0,0,1])
			delete b
		)
	)
)

fn clean_up_empty_layers = 
(
	for i=((layerManager.count)-1) to 0 by -1 do 
	(	
		current_layer = (LayerManager.getLayer i)
		if current_layer.name == "0" then (current_layer.current = true) else   
		(
			current_layer.nodes &theNodes
			if theNodes.count == 0 do LayerManager.deleteLayerByName current_layer.name
		)
	)
)

mapped fn perc_to_color percentage =
(
	(percentage)^(1/2.2)*255;
)

mapped fn get_value name_string string_array =
(
	(for i = 1 to string_array.count where (matchpattern string_array[i] pattern:name_string) collect (filterString string_array[i] "=")[2])[1] as float;
)

mapped fn get_index _pattern string_array = 
(
	for i = 1 to string_array.count where (matchpattern string_array[i] pattern:_pattern) do (return i);
)

fn diffArray a b =
(
	retArray = #()
	for element in a do
	(
		idx = finditem b element
		if idx == 0  then
		(
			append retArray element 
		)
	)
	return retArray
)

fn ies_type_fix ies_light ies_file_path unit_shift:1000 =
(
	ies_file_data = #()
	if ies_file_path != unsupplied then 
	(
		fin = openFile ies_file_path mode:"r"
		while not(eof fin) do
		(
			append ies_file_data (readline fin)
		)
		close fin
	)

	ies_dimentions_index = (get_index "TILT=*" ies_file_data) + 1;
	glo_val = (filterString ies_file_data[ies_dimentions_index] " ")

	if ((glo_val[glo_val.count - 2] as float) > 0 and (glo_val[glo_val.count - 1] as float) > 0 and (glo_val[glo_val.count] as float) == 0) then
	(
		ies_light.type = "Target_Rectangle"
		ies_light.light_Width = (glo_val[glo_val.count - 2] as float)*unit_shift;
		ies_light.light_length = (glo_val[glo_val.count - 1] as float)*unit_shift;
	)

	if ((glo_val[glo_val.count - 2] as float) < 0 and (glo_val[glo_val.count] as float) == 0 and (glo_val[glo_val.count - 2] as float) == (glo_val[glo_val.count - 1] as float)) then
	(
		ies_light.type = "Free_Disc"
		ies_light.light_Radius = ((glo_val[glo_val.count - 2] as float)*unit_shift*-1)/2;
	)
)

fn create_dummies first_index nr_objects string_array unit_shift:1000 =
(
	lumi_array = #()
	last_index = (nr_objects*3 + (first_index - 1))
	for i = first_index to last_index by 3 do
	(
		Lumi = point size:((unit_shift) as integer) Box:true axistripod:true centermarker:true
		Lumi.name = (((filterString string_array[i] "=")[2]) as string)

		Lumi.position.x = ((filterString string_array[i+1] "= ")[2] as float) * unit_shift
		Lumi.position.y = ((filterString string_array[i+1] "= ")[3] as float) * unit_shift
		Lumi.position.z = ((filterString string_array[i+1] "= ")[4] as float) * unit_shift

		LumiTarget = point size:(((unit_shift) as integer)/2) Box:true axistripod:true centermarker:true
		LumiTarget.name = string_array[1] + "_" + (((filterString string_array[i] "=")[2]) as string) + "_target"

		LumiTarget.pivot = LumiTarget.center
		
		LumiTarget.position.x = Lumi.position.x
		LumiTarget.position.y = Lumi.position.y
		LumiTarget.position.z = Lumi.position.z
		
		in coordsys local LumiTarget.position.z = LumiTarget.position.z - unit_shift*4

		Lumi.target = LumiTarget
		Lumi.wirecolor = red
		LumiTarget.wirecolor = green

		rotate Lumi (eulerAngles ((filterString string_array[i+2] "= ")[2] as float) ((filterString string_array[i+2] "= ")[3] as float) (((filterString string_array[i+2] "= ")[4] as float)))

		append lumi_array Lumi
	)
	lumi_array
)

fn SetObjectRotation obj rx ry rz =
(
    -- Reset the object's transformation matrix so that 
    -- it only includes position and scale information. 
    -- Doing this clears out any previous object rotation.
    local translateMat = transMatrix obj.transform.pos
    local scaleMat = scaleMatrix obj.transform.scale
    obj.transform = scaleMat * translateMat
   
    -- Perform each axis rotation individually
    rotate obj (angleaxis rx [1,0,0])
    rotate obj (angleaxis ry [0,1,0])
    rotate obj (angleaxis rz [0,0,1])
)