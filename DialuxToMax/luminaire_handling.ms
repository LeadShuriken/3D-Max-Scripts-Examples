struct lum_handling
(
	public LumiDict = dotNetObject "System.Collections.Hashtable",

	public fn place_luminaire_dummies lum_positions roomNrLums roomNrStruct &standart_struct unit_shift:1000=
	(
		if roomNrStruct != 0 then 
		(
			local struct_layer = LayerManager.newLayerFromName ("_StructLuminaireDummies");
			if roomNrStruct == 1 then
			(
				structure_index = get_index ("Struct" + ((roomNrStruct as integer) as string) + "=*") lum_positions;
				struct_name = (((filterString lum_positions[structure_index] "=")[2]) as string);
				structure_array = crop_array_input ("[" + struct_name + "]") "to_end" lum_positions;

				local firstLumIndex = get_index "Lum1=*" structure_array;

				stf_numb_of_lum = 0;
				for obj in structure_array where (matchpattern obj pattern:"Lum*") do stf_numb_of_lum += 1;
				try(stf_numb_of_lum = stf_numb_of_lum/3)catch();
				current_lum = create_dummies firstLumIndex stf_numb_of_lum structure_array
				for n in current_lum do struct_layer.addnode n
				append standart_struct current_lum
			)
			else 
			(
				for i = 1 to roomNrStruct - 1 do
				(
					structure_index_1 = get_index ("Struct" + ((i as integer) as string) + "=*") lum_positions;
					structure_index_2 = get_index ("Struct" + (((i + 1) as integer) as string) + "=*") lum_positions;

					struct_name_1 = (((FilterString lum_positions[structure_index_1] "=")[2]) as string);
					struct_name_2 = (((filterString lum_positions[structure_index_2] "=")[2]) as string);

					structure_array = crop_array_input ("[" + struct_name_1 + "]") ("[" + struct_name_2 + "]") lum_positions;

					local firstLumIndex = get_index "Lum1=*" structure_array;

					stf_numb_of_lum = 0;
					for obj in structure_array where (matchpattern obj pattern:"Lum*") do stf_numb_of_lum += 1;
					try(stf_numb_of_lum = stf_numb_of_lum/3)catch();
					current_lum = create_dummies firstLumIndex stf_numb_of_lum structure_array
					for n in current_lum do struct_layer.addnode n
					append standart_struct current_lum
				)
				structure_index_1 = get_index ("Struct" + ((roomNrStruct as integer) as string) + "=*") lum_positions;
				struct_name = (((filterString lum_positions[structure_index_1] "=")[2]) as string);
				structure_array = crop_array_input ("[" + struct_name + "]") "to_end" lum_positions;

				local firstLumIndex = get_index "Lum1=*" structure_array;

				stf_numb_of_lum = 0;
				for obj in structure_array where (matchpattern obj pattern:"Lum*") do stf_numb_of_lum += 1;
				try(stf_numb_of_lum = stf_numb_of_lum/3)catch();

				current_lum = create_dummies firstLumIndex stf_numb_of_lum structure_array
				for n in current_lum do struct_layer.addnode n
				append standart_struct current_lum
			)
		)
	),

	public fn add_ies_lights ies_array lumi_data &standart_struct unit_shift:1000 ma_factor: = 
	(
		z_shift = 0

		for structure in standart_struct do 
		(
			for n = 1 to structure.count do
			(
				new_name = ""
				lum_descr_start = get_index ("*" + structure[n].name + "*") lumi_data;
				for i = lum_descr_start to lumi_data.count do
				(
					if matchpattern lumi_data[i] pattern:"Name*" then 
					(
						lum_ies_name_array =  filterstring ((filterstring lumi_data[i] "=")[2]) " /"
						for parts in lum_ies_name_array do
						(
							new_name += (parts + "_")
						)
						new_name = substring new_name 1 (new_name.count - 1)
					)
					else if matchpattern lumi_data[i] pattern:"Box*" then 
					(
						lum_bonding_box = ((filterString lumi_data[i] "=")[2]) as string;
						z_shift = ((filterString lum_bonding_box " ")[3] as float)/2
						exit
					)
				)

				for ies in ies_array where (stricmp new_name (getFilenameFile ies)) == 0 do 
				(
					photometric_light = Target_Light Distribution:3 webfile:ies target:(Targetobject transform:structure[n].target.transform)
					photometric_light.name = structure[n].name + "_photometric"
					photometric_light.position = structure[n].position
					rotate photometric_light (structure[n].transform.rotation);

					photometric_light.position.z -= z_shift * unit_shift;
	 				photometric_light.intensityType = 0
	 				photometric_light.intensity = photometric_light.intensity*ma_factor
	 				photometric_light.on = false
	 				structure[n] = photometric_light;

					ies_type_fix photometric_light ies 
				)
			)
		)
	)
)