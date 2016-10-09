-- Philips OSS Unity Asset tool.
-- Version 1.0
-- Based on a script from Jim de Brouwer, jimdebrouwer@live.nl
-- Extended by Deian Mishev , d.s.mishev@gmail.com
--
-- TODO 
-- Mesh optimization
-- We are not gonna use ProOptimze/Optimize within this script.
-- there must be a better way, student always can use it when they really need to. but we wont force it by this script.
--
-- Pivot :
-- Check if pivot is facing right direction for all type of luminairs

global rolloutWidth = 250
global rolloutHeight = 400

global luminairePosition = [0,0,0];
global luminaireLightPosition = [0,0,0];
global luminaireType;
global luminaireName = (substring maxFilename 3 9 );
global luminaire;

global matLib = #()

global INIFile = (getdir #userScripts) + "\\OSS.ini";
global UnityPath;

function CleanScene = ()
function RefreshRollout = ()
function OptimizeMesh = ()
function FindObject = ()
function collapse_byMaterials= ()
function LoadINIFIle = ()
function SaveINIFile = ()
function ExportFBX = ()
function Initialize = ()

rollout OSSAssetTool "AssetTool 1.0" (
   
	edittext txtluminaireName "" text:luminaireName pos:[0, 0] width:(rolloutWidth-50)
	button btnReload "reload" pos:[rolloutWidth-50,0] width:50 height:18
	button btnCleanScene "Clean scene" pos:[0,20] width:rolloutWidth height:30
	button btnOptimize "Optimize" pos:[0,50] width:rolloutWidth height:30

	groupBox materialEditor "Materials" pos:[0,rolloutHeight - 320] width:rolloutWidth height:100 	 
	spinner matsSpinner "ID" range:[0,100,0] type:#integer enabled:true pos:[15, rolloutHeight - 297] width:40
	dropdownlist materialSelector items:matLib enabled:true pos:[60,rolloutHeight - 300] width:(rolloutWidth-100)
	button btnSetMaterial "set" pos:[210,rolloutHeight - 300] width:30 height:20

	button btnExportFBX "Export FBX" pos:[0,rolloutHeight - 30] width:rolloutWidth height:30


	on btnExportFBX pressed do ExportFBX();
	on btnCleanScene pressed do CleanScene();
	on btnOptimize pressed do OptimizeMesh();
	on btnSetMaterial pressed do ApplyMaterial();
	on btnReload pressed do (
		--unregisterRedrawViewsCallback RefreshRollout;
		Initialize();
		--registerRedrawViewsCallback RefreshRollout;
	)
	on txtluminaireName entered txt do(luminaireName = txt);
	

	groupBox infoBox "Info" pos:[0,rolloutHeight - 100] width:rolloutWidth height:70 	 
	local lbl = "Luminaire position " +  (luminairePosition as string);
	label lblPosition lbl width:(rolloutWidth-10) pos:[5,rolloutHeight - 80] 

	label lblUnityfound "Unity found : false" width:(rolloutWidth-10) pos:[5,rolloutHeight - 60] 
	button btnSetUnityPath "set" pos:[rolloutWidth-50,rolloutHeight - 60] width:50 height:16


	on btnSetUnityPath pressed do SaveINIFile();
)

fn FindLuminaireObject = (
	objectsFound = for o in objects where (matchPattern o.name pattern:("*1*")) collect ( o )
	if (objectsFound.Count >= 1) do (
		return objectsFound[1].name;
	)
	return undefined;
)

fn FindObject name = (
	objectsFound = for o in objects where (matchPattern o.name pattern:("*"+name+"*")) collect ( o )
	if (objectsFound.Count >= 1) do (
		return objectsFound[1];
	)
	return undefined;
)

fn selectAllChildren =
(
	for obj in selection do
	(
		if obj.children != undefined do
		(
			selectmore obj.children
		)
	)
)

fn ExportFBX = (
	FBXExporterSetParam "Cameras" False
	FBXExporterSetParam "Animation" False
	FBXExporterSetParam "EmbedTextures" False
	FBXExporterSetParam "Lights" False
	l = FindObject(OSSAssetTool.txtluminaireName.text);
	select l;
	selectAllChildren();
	
	fbxPath = (UnityPath + "\\Assets\\3D Assets\\FBX\\");
	exportPath = (fbxPath + OSSAssetTool.txtluminaireName.text + ".FBX");
	exportFile exportPath #noPrompt selectedOnly:true
)

fn ApplyMaterial = (
	-- updates scenematerials array
	tempFile = GetDir #temp + "\\_temp.max"
	saveNodes #() tempFile
	deleteFile tempFile

	material = undefined;
	for m in scenematerials do(
		if ((classof m) == Multimaterial) do(
		     material = m
		)
	)

	if (material != undefined) then (
		OSSAssetTool.matsSpinner.range.x = 1;
		OSSAssetTool.matsSpinner.range.y = material.materialList.Count;
		if (OSSAssetTool.matsSpinner.value > 0) then (
			if (OSSAssetTool.matsSpinner.value <= material.materialList.Count) then (
				material.materialList[OSSAssetTool.matsSpinner.value].name = OSSAssetTool.materialSelector.text;
			)
		)
	) else (
		scenematerials[1].name = OSSAssetTool.materialSelector.text;
	)

	if (MatEditor.isOpen()) then (
		MatEditor.Close(); 
		MatEditor.Open(); 
	)
)

fn LoadUnityMaterials = (
	my_files = #();
	materialPath = (UnityPath + "\\Assets\\2D Assets\\Materials");
	directories = getDirectories (materialPath+"/*")
	for f in directories do
		join my_files (getFiles (f + "*.mat"))
	
	materials = #();
	for f in my_files do
		append materials (getFilenameFile(f))

	OSSAssetTool.materialSelector.items = materials;
)

fn RefreshRollout = (
	assetLocator = FindObject("asset_locator");
	if (assetLocator != undefined) do (
		luminairePosition = assetLocator.pos;
		luminaireType = assetLocator.name;
	)
	luminaireName = (substring maxFilename 3 9 );
	print "hallo" ;
	if (luminaireName == undefined) then (
		luminaireName = FindLuminaireObject();
		print luminaireName ;
	)
	if (luminaireName == "") then (
		luminaireName = FindLuminaireObject();
		print luminaireName ;
	)
	OSSAssetTool.txtluminaireName.text =luminaireName;
	OSSAssetTool.lblPosition.text =("Luminaire position " + (luminairePosition as string));
)

fn StringContains string searchstring = (
	ltype = findString string searchstring;
	if (ltype == undefined) then return false;
	return true;
)

fn GetLuminaireDirection = (
	direction = [0,-1,0]
	if (StringContains luminaireType "onwall") do direction = [1,0,0];
	if (StringContains luminaireType "inwall") do direction = [1,0,0];
	if (StringContains luminaireType "onceiling") do direction = [0,-1,0];
	if (StringContains luminaireType "inceiling") do direction = [0,-1,0];
	if (StringContains luminaireType "hanging") do direction = [0,-1,0];
	if (StringContains luminaireType "onfloor") do direction = [0,1,0];
	if (StringContains luminaireType "infloor") do direction = [0,1,0];
	if (StringContains luminaireType "floodlight") do direction = [0,1,0];
	if (StringContains luminaireType "pedestal") do direction = [0,1,0];
	local rot = (matrixFromNormal direction) as quat;
    rot
)

fn RotatePivotOnly obj rotation = (
	animate off in coordsys local obj.rotation*=rotation
	obj.objectoffsetrot*=rotation
	obj.objectoffsetpos*=rotation
)

fn get_objsByMaterial mat = (
	the_objs = #()
	for r in (refs.dependents mat) do(
		if (superClassOf r) == GeometryClass do(
			append the_objs r
		)
	)
	the_objs
)

fn Collapse_byMaterials = (
	s_inc = 0
	for s in sceneMaterials do(
		--Create empty mesh
		m = mesh()
		f = m.faces
		meshOp.deleteFaces m f	
		objs = get_objsByMaterial s
		for o in objs do attach m o;
		
		
		if (m.numverts == 0) do delete m;
	)

	objectsInScene = #();
	for object in geometry do
		if (not isDeleted object) then
			append objectsInScene object;

	m = mesh()
	f = m.faces
	meshOp.deleteFaces m f

	for object in objectsInScene do
		object.material = standard();
	
	for object in objectsInScene do
		meshOp.attach m object attachMat:#IDToMat condenseMat:true deleteSourceNode:true;
			
	m.name = luminaireName;
	meshop.applyUVWMap m #box 

	ConvertTo m Editable_poly

	luminaire = m;
)

fn CreateDummy =  (
	in coordsys world
	dummyObject = Dummy name:"bulb" pos:luminaireLightPosition scale:[15,15,15]
	dummyObject.parent = (getNodeByName luminaireName);
)

fn OptimizeMesh = (
	
	RotatePivotOnly (getNodeByName luminaireName) (GetLuminaireDirection())
	CreateDummy();
)

fn GetAllNonGeometry = (
	local nonGeoArray = #();
	append nonGeoArray helpers;
	append nonGeoArray lights;
	return nonGeoArray;
)

fn CalculateLuminairePosition = (
	in coordsys world
	Positions = for o in lights collect o.position
	Xpos = 0;	Ypos = 0;	Zpos = 0
	for i = 1 to Positions.count do
	(
		Xpos += Positions[i].x
		Ypos += Positions[i].y
		Zpos += Positions[i].z
	)
	totalLightPosition = [Xpos ,Ypos ,Zpos] as point3;
	luminaireLightPosition = (totalLightPosition/Positions.count);
)

fn CleanScene = (
	RefreshRollout();
	CalculateLuminairePosition();
	Collapse_byMaterials();
	for obj in GetAllNonGeometry() do (
		delete obj;
	)
	--unregisterRedrawViewsCallback RefreshRollout;
)

fn calcPercent val total = (
	ret = 100.0*val/total
	ret
)

fn LoadINIFIle = (
	UnityPath = getINISetting INIFile "Paths" "Unity" 
	OSSAssetTool.lblUnityfound.text = ("Unity found : " + ((doesFileExist UnityPath) as string));
)

fn SaveINIFile = (
	UnityPath = getSavepath();
	if (UnityPath != undefined) then (
		setINISetting INIFile "Paths" "Unity" UnityPath;
	)
	Initialize();
)

fn Initialize = (
	clearListener();
	units.displaytype = #Metric
	units.MetricType = #Millimeters
	units.SystemType = #Millimeters

	try(destroyDialog OSSAssetTool)catch()
	CreateDialog OSSAssetTool width:rolloutWidth hobjecteight:rolloutHeight;
	LoadINIFIle();
	LoadUnityMaterials();
	RefreshRollout();
)

--unregisterRedrawViewsCallback RefreshRollout;
Initialize();
--registerRedrawViewsCallback RefreshRollout;
