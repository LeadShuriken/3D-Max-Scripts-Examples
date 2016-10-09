struct room_builder
(
	private workingPlane,
	
	public roomHeight,
	public roomNrPoints,
	public MF,
   
	private fn replace_dummies dia_name max_name unit_shift:1000 =
	(
		passed_obj = #()
		boxes = for obj in objects where classof obj == Box collect obj
		for b in boxes do
		(
			object_dianame = (filterstring b.name "_")[2];
			libraryindex = finditem dia_name object_dianame;
			if libraryindex != 0 do
			(
				mergemaxfile max_name[libraryindex] #select #AutoRenameDups #useSceneMtlDups #neverReparent quite:true
				merged = selection as array
				if merged.count > 1 do
				(
					---MATERIAL OVERIDE---
					for obj in merged where superClassOf obj == GeometryClass do obj.mat = Arch___Design__mi()
					----------------------
					group merged name:b.name select:true
					merged = selection as array
					merged = for m in merged where classof m == Dummy collect m 	-- get the group dummy
					if (distance selection.max selection.min < 100) do 				-- scale object that have wrong units
					(
						scale merged[1] [1000, 1000, 1000]
					)
				)
				merged = merged[1]
				merged.name = b.name
				merged.mat = b.mat 

				scale merged [unit_shift/1000, unit_shift/1000, unit_shift/1000]
					
				merged.position = b.position
				in coordsys merged merged.rotation = b.rotation
				in coordsys local rotate merged (angleaxis 90 [0,0,1])
				delete b
				append passed_obj merged
			)
		)
		passed_obj
	),
	
	private fn change_material_props obj room_strArray =
	(
		firstPointIndex = get_index ("[" + (FilterString obj.name "_")[1] + "]") room_strArray;
		data = (filterstring room_strArray[firstPointIndex + 1] "= ")
		r = data[3] as integer
		g = data[4] as integer
		b = data[5] as integer
		reflectance = (data[2] as float)/100
		obj.mat = Arch___Design__mi name:obj.name diff_color:(color r g b) diff_weight:reflectance refl_weight: 0.0
	),

	public fn build_room room_strArray read_lut_stream: furn_models: unit_shift:1000 = 
	(
		local firstFurnitureIndex;
		local firstPointIndex;
		local firstWallIndex;
		local firstLumIndex;
		local room_obj = #();

		this.roomHeight = get_value "Height=*" room_strArray;
		this.roomNrPoints = get_value "NrPoints=*" room_strArray;

		firstPointIndex = get_index "Point1=*" room_strArray ;
		firstWallIndex = get_index "R_Wall1=*" room_strArray ;
		firstLumIndex = get_index "Lum1=*" room_strArray ;
		
		ceiling_ref = get_index "R_Ceiling=*" room_strArray;
		floor_ref = get_index "R_Floor=*" room_strArray;
		this.MF = get_index "MF=*" room_strArray;

		if classof ceiling_ref != OkClass then ceiling_ref = (filterString room_strArray[ceiling_ref] "=")[2] as float else ceiling_ref = 0.7
		if classof floor_ref != OkClass then floor_ref = (filterString room_strArray[floor_ref] "=")[2] as float else floor_ref = 0.2
		if classof this.MF != OkClass then this.MF = (filterString room_strArray[this.MF] "=")[2] as float else this.MF = 0.8

	    dummy_layer = LayerManager.newLayerFromName (room_strArray[1]  + "_BoxGeometry")
	    geometry_layer = LayerManager.newLayerFromName (room_strArray[1]  + "_RoomArchGeometry")

		---------------------------------------------------------------------------------------------------------

		ceiling_mat = Arch___Design__mi name:(room_strArray[1] + "_ceiling_mat") diff_color:(color 255 255 255) diff_weight:ceiling_ref refl_weight: 0.0
		surface_mat = Multimaterial name:(room_strArray[1] + "surface_mat") materialList: #(ceiling_mat)

		if classof firstWallIndex != OkClass then 
		(
			for i = firstWallIndex to ((firstWallIndex - 1) + this.roomNrPoints) do
			(	
				reflectance = (filterString room_strArray[i]  "=")[2] as float

				mate = Arch___Design__mi diff_color:(color 255 255 255) diff_weight:reflectance refl_weight: 0.0
				mate.name = room_strArray[1] + "_" + ((filterstring room_strArray[i] "=")[1])
				surface_mat.materialList[surface_mat.materialList.count + 1] = mate
			)
		)
		else 
		(
			reflectance = 0.50
			for i = 1 to this.roomNrPoints do
			(	
				mate = Arch___Design__mi diff_color:(color 255 255 255) diff_weight:reflectance refl_weight: 0.0
				mate.name = room_strArray[1] + "_" + "R_Wall" + i as string
				surface_mat.materialList[surface_mat.materialList.count + 1] = mate
			)
		)
		
		surface_mat.materialList[surface_mat.materialList.count + 1] = Arch___Design__mi name:(room_strArray[1] + "_floor_mat") diff_color:(color 255 255 255) diff_weight:floor_ref refl_weight: 0.0

		------------- Build Points -----------------------------------------------------------------------------

		room_shape = SplineShape pos:[0,0,0] name:(room_strArray[1] + "_room_shape")
		addNewSpline room_shape
		
		for i = firstPointIndex to ((firstPointIndex - 1) + this.roomNrPoints) do
		(
			a = filterstring (filterString room_strArray[i]  "=")[2] " "
			addKnot room_shape 1 #corner #line [(a[1] as float) * unit_shift, (a[2] as float) * unit_shift, 0]
			if (numknots room_shape) > 1 then (setMaterialID room_shape 1 (numknots room_shape - 1 ) (numknots room_shape))
		)
		updateShape room_shape
		floorspline = copy room_shape

		a = filterstring (filterString room_strArray[firstPointIndex] "=")[2] " " 
		addKnot room_shape 1 #corner #line [(a[1] as float) * unit_shift, (a[2] as float) * unit_shift, 0]
		setMaterialID room_shape 1 (numknots room_shape - 1 ) (numknots room_shape)
		updateShape room_shape

		------------ Add walls ---------------------------------------------------------------------------------

		walls = copy room_shape
		addmodifier walls (Extrude amount:(this.roomHeight*unit_shift) useShapeIDs:on)
		addmodifier walls (Vertex_Weld())
		addmodifier walls (Normalmodifier flip:on)
		addmodifier walls (Smooth autosmooth:off)
		walls.primaryvisibility = false

		walls.mat = surface_mat
		walls.name = room_strArray[1] + "_walls"
		geometry_layer.addNode walls

		----------- Add a floor --------------------------------------------------------------------------------

		close floorspline 1
		updateShape floorspline
		delete room_shape
		
		converttopoly floorspline
		polyOp.setFaceSmoothGroup floorspline 1 0
	
		floorspline.name = room_strArray[1] + "_floor"
		floorspline.mat = surface_mat.materialList[surface_mat.materialList.count]
		floorspline.primaryvisibility = false

		while floorspline.modifiers.count != 0 do
		(
			if floorspline.modifiers[1] != undefined then deleteModifier floorspline floorspline.modifiers[1]
		)
		geometry_layer.addNode floorspline

		----------- Create ceiling -----------------------------------------------------------------------------

		ceilingspline = copy floorspline 
		ceilingspline.name = room_strArray[1] + "_ceiling"
		ceilingspline.pos.z = this.roomHeight*unit_shift
		addmodifier ceilingspline (Normalmodifier flip:on)
		ceilingspline.mat = surface_mat[1]
		geometry_layer.addNode ceilingspline
		ceilingspline.primaryvisibility = false

		----------- Add workplane ------------------------------------------------------------------------------
		
		workingPlaneHeight = get_value "WorkingPlane=*" room_strArray
		this.workingPlane = copy floorspline
		this.workingPlane.name = room_strArray[1] + "_working_plane"
		in coordsys world move this.workingPlane [0, 0, workingPlaneHeight * unit_shift]

		----- Remove when concept finalized --------------------------------------------------------------------
		/*
		polyop.bevelFaces this.workingPlane #{1} 0 (-1*(wall_zone/100.0)*unit_shift)

		face_count = polyOp.getNumFaces this.workingPlane
		
		polyop.deleteFaces this.workingPlane #{2..face_count} delIsoVerts:true
		*/

		this.workingPlane.mat = Arch___Design__mi name: (room_strArray[1] + "_working_plane")
		this.workingPlane.pivot = this.workingPlane.center
		this.workingPlane.receiveshadows = false;
		this.workingPlane.castShadows = false;
		geometry_layer.addNode this.workingPlane
		
		----- PlaceBoxObjReplacements --------------------------------------------------------------------------

		roomNrFurns = get_index "NrFurns=*" room_strArray;
		if classof roomNrFurns != OkClass then roomNrFurns = (filterString room_strArray[roomNrFurns] "=")[2] as float else roomNrFurns = 0
		
		firstFurnitureIndex = get_index "Furn1=*" room_strArray
	
		if roomNrFurns != undefined and roomNrFurns != 0.0 do 
		(
			last_index = (firstFurnitureIndex + roomNrFurns*5) as integer
			for i = firstFurnitureIndex to last_index - 1 by 5 do
			(
				DummyOBJ = Box()
				DummyOBJ.mat = Arch___Design__mi()
				DummyOBJ.name = ((filterString room_strArray[i+1] "=")[2]) as string + "_" + ((filterString room_strArray[i] "=")[2]) as string
			
				in coordsys local DummyOBJ.rotation = eulertoquat(eulerAngles ((filterString room_strArray[i+2] "= ")[2] as float) ((filterString room_strArray[i+2] "= ")[3] as float) ((filterString room_strArray[i+2] "= ")[4] as float - 90))

				DummyOBJ.length = ((filterString room_strArray[i+4] "= ")[2] as float) * unit_shift
				DummyOBJ.width = ((filterString room_strArray[i+4] "= ")[3] as float) * unit_shift
				DummyOBJ.height= ((filterString room_strArray[i+4] "= ")[4] as float) * unit_shift
				
				DummyOBJ.pivot = DummyOBJ.center
				
				DummyOBJ.position.x = ((filterString room_strArray[i+3] "= ")[2] as float) * unit_shift
				DummyOBJ.position.y = ((filterString room_strArray[i+3] "= ")[3] as float) * unit_shift
				DummyOBJ.position.z = ((filterString room_strArray[i+3] "= ")[4] as float) * unit_shift

				append room_obj DummyOBJ
				DummyOBJ.wirecolor = green

				DummyOBJ.primaryvisibility = false;
	            dummy_layer.addNode DummyOBJ
			)
		)

		LUT_file_lines = #()

		if read_lut_stream != unsupplied and furn_models != unsupplied then
		(
			if furn_models.count != 0 do 
			(
				dia_name = #() 
				max_name = #()

				LUT_file_lines = file_to_array read_lut_stream

				dia_temp = filterstring LUT_file_lines[2] "<>"
				dia_temp[dia_temp.count-1]

				for max_file in furn_models do
				(
					for l = 2 to (LUT_file_lines.count - 1) do
					(
						max_temp = filterstring LUT_file_lines[l] "\""
						max_temp = max_temp[max_temp.count-1]
						
						if ((findstring max_temp ((getfilenamefile max_file) + "_m3dName")) != undefined) do
						(
							dia_temp = filterstring LUT_file_lines[l] "<>"
							dia_temp = dia_temp[dia_temp.count-1]
							append dia_name dia_temp
							append max_name max_file
						)
					)
				)
				furn = this.replace_dummies dia_name max_name
				for obj in furn do change_material_props obj room_strArray
			)
		)
	),

	public fn create_camera wall_zone Nx Ny unit_shift:1000 = 
	(
		bb = nodeGetBoundingBox this.workingPlane this.workingPlane.transform
		
		target_plane_length = bb[2].y - bb[1].y
		target_plane_width = bb[2].x - bb[1].x

		wall_zone = (wall_zone/100.0)*unit_shift; -- dialux term, how much margin to exclude around the border of the workplane
		resolution = [NX,Ny]; -- render resolution -- to change
		camera_distance = 0.1 * unit_shift; -- camera distance above workplane

		c = Freecamera pos:[0,0,0] isSelected:on targetDistance:camera_distance;
		c.rotation = this.workingPlane.rotation;
		c.pos = this.workingPlane.pos;
		in coordsys c move c [0,0,camera_distance];
		c.fov = 2.0d0 *atan((((target_plane_width as double) - (wall_zone as double)*2) / 2.0d0) / (camera_distance as double));
		c.orthoProjection = true;

		renderPixelAspect = (((target_plane_width as double) - (wall_zone as double))/((target_plane_length as double) - (wall_zone as double))) * ((Ny as double)/(Nx as double));
		renderWidth = resolution[1];
		renderHeight = resolution[2];

		c.parent = this.workingPlane

		renderSceneDialog.update();
		c
	)
)