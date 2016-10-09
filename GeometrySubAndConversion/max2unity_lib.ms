fn validate_map map = 
(
	if classof map == Bitmaptexture then
	(
		try 
		(
			map.bitmap -- test if we can reference the bitmap
			return map
		)
		catch
		(
			return undefined
		)
	)
	else
	(
		return undefined
	)
)

fn vray_to_standard obj_array =
(
-- This script replaces v-ray materials with standard material, and sets values accordingly, including maps.
-- it loops over all objects in the scene and updates materials, perhaps looping over material instances is easier...

	for o in obj_array do
	(
		format "%\n" o.material
		if classof o.material == VRayMtl then
		(
			format "Replacing % with standard material...\n" o.material.name
			tempMtl = standard()
			tempMtl.name = o.material.name + "_standard"
			tempMtl.diffuse = o.material.diffuse
			tempMtl.specular = o.material.reflection
			tempMtl.specularLevel = 100
			tempMtl.glossiness = o.material.reflection_glossiness * 100
			tempMtl.opacity = floor (100 * (255/255.0 - o.material.refraction.g/255.0))
			tempMtl.diffuseMap = validate_map o.material.texmap_diffuse
			tempMtl.reflectionMap = validate_map o.material.texmap_reflection
			tempMtl.refractionMap = validate_map o.material.texmap_refraction
			tempMtl.opacityMap = validate_map o.material.texmap_opacity
			tempMtl.bumpMap = validate_map o.material.texmap_bump
			o.material = tempMtl
		) 
		else if classof o.material == VRayOverrideMtl or classof o.material == VRayBlendMtl then
		(
			if classof o.material.baseMtl == VRayFastSSS2 then
			(
				format "Replacing % with standard material...\n" o.material.name
				tempMtl = standard()
				tempMtl.name = o.material.name + "_standard"
				tempMtl.diffuse = o.material.baseMtl.overall_color
				tempMtl.specular = o.material.baseMtl.specular_color
				tempMtl.specularLevel = o.material.baseMtl.specular_amount * 100
				tempMtl.glossiness = o.material.baseMtl.specular_glossiness * 100
				tempMtl.useSelfIllumColor = true
				tempMtl.selfIllumColor = o.material.baseMtl.overall_color
				-- a call to the Simplygon SDK interface for creation of the emission channel
				--sgsdk_MaterialColor tempMtl.name "Emissive" o.material.baseMtl.overall_color.r o.material.baseMtl.overall_color.g o.material.baseMtl.overall_color.b o.material.baseMtl.overall_color.a
				--tempMtl.opacity = 80
				if o.material.baseMtl.texmap_sss_color != undefined then
				(
					tempMtl.selfillumMap = validate_map o.material.baseMtl.texmap_sss_color
					tempMtl.diffuseMap = validate_map o.material.baseMtl.texmap_sss_color
				)
				else if o.material.baseMtl.texmap_diffuse_color != undefined then
				(
					tempMtl.selfillumMap = validate_map o.material.baseMtl.texmap_diffuse_color
					tempMtl.diffuseMap = validate_map o.material.baseMtl.texmap_diffuse_color
				)
				else if o.material.baseMtl.texmap_overall_color != undefined do
				(
					tempMtl.selfillumMap = validate_map o.material.baseMtl.texmap_overall_color
					tempMtl.diffuseMap = validate_map o.material.baseMtl.texmap_overall_color		
				)
				o.material = tempMtl
			) 
			else if classof o.material.baseMtl == VRayMtl do
			(
				format "Replacing % with standard material...\n" o.material.name
				tempMtl = standard()
				tempMtl.name = o.material.name + "_standard"
				tempMtl.diffuse = o.material.baseMtl.diffuse
				tempMtl.specular = o.material.baseMtl.reflection
				tempMtl.specularLevel = 100
				tempMtl.glossiness = o.material.baseMtl.reflection_glossiness * 100
				tempMtl.opacity = floor (100 * (255/255.0 - o.material.baseMtl.refraction.g/255.0))
				tempMtl.diffuseMap = validate_map o.material.baseMtl.texmap_diffuse
				tempMtl.reflectionMap = validate_map o.material.baseMtl.texmap_reflection
				tempMtl.refractionMap = validate_map o.material.baseMtl.texmap_refraction
				tempMtl.opacityMap = validate_map o.material.baseMtl.texmap_opacity
				tempMtl.bumpMap = validate_map o.material.baseMtl.texmap_bump
				o.material = tempMtl
			)
		)
		else if classof o.material == Multimaterial do
		(
			for i = 1 to o.material.count do -- do the same replacements as above but now for all sub materials
			(
				if classof o.material[i] == VRayMtl then
				(
					format "Replacing % with standard material...\n" o.material[i].name
					tempMtl = standard()
					tempMtl.name = o.material[i].name + "_standard"
					tempMtl.diffuse = o.material[i].diffuse
					tempMtl.specular = o.material[i].reflection
					tempMtl.specularLevel = 100
					tempMtl.glossiness = o.material[i].reflection_glossiness * 100
					tempMtl.opacity = floor (100 * (255/255.0 - o.material[i].refraction.g/255.0))
					tempMtl.diffuseMap = validate_map o.material[i].texmap_diffuse
					tempMtl.reflectionMap = validate_map o.material[i].texmap_reflection
					tempMtl.refractionMap = validate_map o.material[i].texmap_refraction
					tempMtl.opacityMap = validate_map o.material[i].texmap_opacity
					tempMtl.bumpMap = validate_map o.material[i].texmap_bump
					o.material[i] = tempMtl
				) 
				else if classof o.material[i] == VRayOverrideMtl or classof o.material[i] == VRayBlendMtl do
				(
					if classof o.material[i].baseMtl == VRayFastSSS2 then
					(
						format "Replacing % with standard material...\n" o.material[i].name
						tempMtl = standard()
						tempMtl.name = o.material[i].name + "_standard"
						tempMtl.diffuse = o.material[i].baseMtl.overall_color
						tempMtl.specular = o.material[i].baseMtl.specular_color
						tempMtl.specularLevel = o.material[i].baseMtl.specular_amount * 100
						tempMtl.glossiness = o.material[i].baseMtl.specular_glossiness * 100
						tempMtl.useSelfIllumColor = true
						tempMtl.selfIllumColor = o.material[i].baseMtl.overall_color
						-- a call to the Simplygon SDK interface for creation of the emission channel
						--sgsdk_MaterialColor tempMtl.name "Emissive" o.material.baseMtl.overall_color.r o.material.baseMtl.overall_color.g o.material.baseMtl.overall_color.b o.material.baseMtl.overall_color.a
						--tempMtl.opacity = 80
						if o.material[i].baseMtl.texmap_sss_color != undefined then
						(
							tempMtl.selfillumMap = validate_map o.material[i].baseMtl.texmap_sss_color
							tempMtl.diffuseMap = validate_map o.material[i].baseMtl.texmap_sss_color
						)
						else if o.material[i].baseMtl.texmap_diffuse_color != undefined then
						(
							tempMtl.selfillumMap = validate_map o.material[i].baseMtl.texmap_diffuse_color
							tempMtl.diffuseMap = validate_map o.material[i].baseMtl.texmap_diffuse_color
						)
						else if o.material[i].baseMtl.texmap_overall_color != undefined do
						(
							tempMtl.selfillumMap = validate_map o.material[i].baseMtl.texmap_overall_color
							tempMtl.diffuseMap = validate_map o.material[i].baseMtl.texmap_overall_color		
						)
						o.material[i] = tempMtl
					) 
					else if classof o.material[i].baseMtl == VRayMtl do
					(
						format "Replacing % with standard material...\n" o.material[i].name
						tempMtl = standard()
						tempMtl.name = o.material[i].name + "_standard"
						tempMtl.diffuse = o.material[i].baseMtl.diffuse
						tempMtl.specular = o.material[i].baseMtl.reflection
						tempMtl.specularLevel = 100
						tempMtl.glossiness = o.material[i].baseMtl.reflection_glossiness * 100
						tempMtl.opacity = floor (100 * (255/255.0 - o.material[i].baseMtl.refraction.g/255.0))
						tempMtl.diffuseMap = validate_map o.material[i].baseMtl.texmap_diffuse
						tempMtl.reflectionMap = validate_map o.material[i].baseMtl.texmap_reflection
						tempMtl.refractionMap = validate_map o.material[i].baseMtl.texmap_refraction
						tempMtl.opacityMap = validate_map o.material[i].baseMtl.texmap_opacity
						tempMtl.bumpMap = validate_map o.material[i].baseMtl.texmap_bump
						o.material[i] = tempMtl
					)
				)
			)
		)
	)
)

fn reduce_polygons to_be_reduced =
(
-- Applies reduction to selected body objects and mesh/poly objects.
	
	select to_be_reduced

	-- make sure everything is ungrouped, otherwise we get into trouble
	actionMan.executeAction 0 "40141"  -- Groups: Ungroup
	actionMan.executeAction 0 "40141"  -- Groups: Ungroup
	actionMan.executeAction 0 "40141"  -- Groups: Ungroup

	-- make all objects unique, de-instance
	InstanceMgr.MakeObjectsUnique &to_be_reduced #individual

	-- reduce non-body objects using optimize
	for o in to_be_reduced where ((classof o.baseobject) != Body_Object or o.modifiers[#Edit_Poly] != undefined) do
	(
		select o
		try
		(
			addmodifier $ (ProOptimizer ())
			$.modifiers[#ProOptimizer].VertexPercent = 7
			$.modifiers[#ProOptimizer].MergePoints = on
			$.modifiers[#ProOptimizer].Calculate = on
			ConvertTo $ PolyMeshObject -- collapse immediatly otherwise it gets reset
			
-- 			addmodifier $ (Edit_Normals())
-- 			$.modifiers[#Edit_Normals].EditNormalsMod.SetSelection #{1..$.modifiers[#Edit_Normals].EditNormalsMod.GetNumNormals()}
-- 			$.modifiers[#Edit_Normals].EditNormalsMod.MakeExplicit ()
-- 			addmodifier $ (optimize())
-- 			$.modifiers[#Optimize].preservesmooth1 = on
-- 			$.modifiers[#Optimize].preservemat1 = on
-- 			$.modifiers[#Optimize].facethreshold1 = 9.0
-- 			$.modifiers[#Optimize].edgethreshold1 = 9.0
-- 			$.modifiers[#Optimize].bias1 = 0.22
		)
		catch ()
	)


	-- to_be_reduced = for o in to_be_reduced where isValidNode o collect o -- remove deleted nodes from array

	-- apply reduction settings to body objects
	for o in to_be_reduced where ((classof o.baseobject) == Body_Object and o.modifiers[#Edit_Poly] == undefined) do
	(
		select o
		size_o = length($.max - $.min)
		format "Updating %...\n" o.name	
		$.baseobject.RenderViewportMeshRA = on
		$.baseobject.LowQualityVDS = off
		$.baseobject.MediumQualityVDS = off
		$.baseobject.HighQualityVDS = on
		$.baseobject.IsoAngleDS = 3
		$.baseobject.IsoChordHeightVDS = 0
		$.baseobject.FaceApproxAngleVDS = 50.0--25.0
		$.baseobject.EdgeApproxAngleVDS = 0
		v = 100 - size_o;
		if v < 1 do v = 1;
		$.baseobject.FaceChordHeightVDS = v
		$.baseobject.EdgeChordHeightVDS = 0
		$.baseobject.MaxEdgeLengthPctVDS = 25.0*(100.0/size_o)--25
		
		if (getPolygonCount $)[1]/length ($.max - $.min) > 100 do -- if reduction didn't really get us there, go crazy
		(
			--addmodifier $ (Vertex_Weld ())
			--$.modifiers[#Vertex_Weld].threshold = 2
			
			dim = in coordsys $ (nodeLocalBoundingBox $)[2] - (nodeLocalBoundingBox $)[1]

			c = Cylinder()
			c.radius = dim[1] / 2.0
			c.height = dim[3]
			c.rotation = $.rotation
			c.center = $.center
			c.material = $.material
			delete $
		)
	)

	format "Done!\n"
)

fn attach_all to_attach =
(
	new_object = editable_mesh()
	new_object.name = "merged_mesh"
	convertTo new_object PolyMeshObject
	--resetxform new_object

	parts = for s in to_attach as array where superclassof s == GeometryClass collect s

	for p in parts do
	(
		convertTo p PolyMeshObject
		resetxform p
		polyop.attach new_object p
	)
)

fn ungroup_all to_ungroup =
(
	select to_ungroup
	actionMan.executeAction 0 "40141"  -- Groups: Ungroup
	actionMan.executeAction 0 "40141"  -- Groups: Ungroup
	actionMan.executeAction 0 "40141"  -- Groups: Ungroup
)

fn delete_invisible_faces selected_object =
(
	select selected_object
	
	delete lights
	RadiosityPreferences.displayResetWarning  = false

	format "Load radiosity render preset.\n"
	renderpresets.LoadAll 0 (getFilenamePath( getThisScriptFilename() ) + "radiosity.rps")

	Skylight ()

	($Sky* as array)[1].color = color 255 255 255

	select selected_object

	format "Convert to editable poly.\n"
	macros.run "Modifier Stack" "Convert_to_Poly"

	modPanel.addModToSelection (VertexPaint ()) ui:on

	$.modifiers[#VertexPaint].mapChannel = -1
	$.modifiers[#VertexPaint].lightingModel = 0
	$.modifiers[#VertexPaint].radiosityOption = 1
	$.modifiers[#VertexPaint].colorBy = 1
	$.modifiers[#VertexPaint].layerIsolated = on

	SceneRadiosity.showPanel()

	rootScene[#Environment][#Advanced_Lighting].meshingEnabled = on
	rootScene[#Environment][#Advanced_Lighting].meshingSize = 2000
	rootScene[#Environment][#Advanced_Lighting].shootDirectLights = on
	rootScene[#Environment][#Advanced_Lighting].includeSkylight = on

	child = #();
	format "Calculate AO using radiosity..\n"
	sleep 1; windows.processPostedMessages();
	for c in windows.getChildrenHWND 0 parent:#max where c[5] == "Render Setup: Default Scanline Renderer" do child = c[1];
	sleep 1; windows.processPostedMessages();
	for c in windows.getChildrenHWND child where c[5] == "Start" do UIAccessor.PressButton c[1];

	while (radiosityMeshOps.isRadiosityValid()==false) do (windows.processPostedMessages(););

	completeRedraw();
	forceCompleteRedraw doDisabled:true;
		
	sleep 1;

	format "Assign vertex illumination...\n"
	for c in windows.getChildrenHWND #max where c[5] == "Assign" do UIAccessor.PressButton c[1];

	macros.run "Modifier Stack" "Convert_to_Poly"

	format "Select vertices based on color...\n"
	$.vertSelectBy = 1
	$.vertSelectionColor = color 0 0 0
	$.vertSelectionRedRange = 100
	$.vertSelectionGreenRange = 100
	$.vertSelectionBlueRange = 100

	$.buttonOp #SelectByVertexColor

	subobjectLevel = 1
	$.EditablePoly.ShrinkSelection ()

	$.EditablePoly.ConvertSelection #Vertex #Face

	format "Delete selected faces.\n"
	$.delete #Face

	delete $Sky*
	unhide lights
)
