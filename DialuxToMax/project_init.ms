struct project_builder
(
	private room_names = #(), -- NAME OF THE ROOMS INSIDE THE THE PROJECT
	private numb_rooms = 0,   -- NUMBER OF ROOMS INSIDE THE PROJECT

	public fn project_init file_path func_data_file &file_lum_info &room_data &nr_lum_struct &nr_lums &lum_info =
	(
		room_data = this.initialise_room_info file_path
		
		if (func_data_file != unsupplied) then
		(
			file_lum_info = file_to_array func_data_file
		)
		else
		(
			file_lum_info = room_data
		)

		lum_info = this.initialise_lum_info file_lum_info &nr_lum_struct &nr_lums
	),

	private fn initialise_room_info file_path =
	(
		local project_array = file_to_array file_path
		local string_counter = 0
		local roomStartLineOld

		thisStartLine = (findItem project_array "[this]") as integer -1
		HeaderStArray = for i = 1 to thisStartLine collect project_array[i]
		project_array = diffArray project_array HeaderStArray
		
		this.numb_rooms = (for i = 1 to project_array.count where (matchpattern project_array[i] pattern:"NrRooms*") collect (filterString project_array[i] "=")[2])[1] as integer;
		this.room_names = for i = 1 to project_array.count where (matchpattern project_array[i] pattern:"Room*") collect ((filterString project_array[i] "=")[2]) as string;
		
		roomStartLineOld = (findItem project_array ("[" + this.room_names[1] + "]")) as integer - 1
		projectStArray = for i = 1 to roomStartLineOld collect project_array[i]																
		project_array = diffArray project_array projectStArray

		roomStartLineOld = 1

		for m = 1 to this.numb_rooms - 1 do
		(
			roomStartLineNew = (findItem project_array ("[" + this.room_names[m + 1] + "]")) as integer
			a = for i = roomStartLineOld to (roomStartLineNew - 1) collect project_array[i]
			string_counter = string_counter + a.count
			append room_strArray a
			roomStartLineOld = roomStartLineNew
		)

		room_strArray = ((for i = string_counter + 1 to project_array.count collect project_array[i]) as array)
		room_strArray
	),

	private fn initialise_lum_info file_lum_info &nr_lum_struct &nr_lums =
	(
		nr_lum_struct = get_index "NrStruct=*" file_lum_info
		if (classof nr_lum_struct != OkClass) then (nr_lum_struct = (filterString file_lum_info[nr_lum_struct] "s=")[2] as float) else (nr_lum_struct = 0.0)
		nr_lums = get_index "nr_lums=*" file_lum_info
		if (classof nr_lums != OkClass) then (nr_lums = (filterString file_lum_info[nr_lums] "=")[2] as float) else (nr_lums = 0.0)

		types_of_lum_number = 0
		lumidata_index = get_index "[LUMINAIRE.L1]" file_lum_info ;
		lum_info = (for i = lumidata_index to file_lum_info.count collect file_lum_info[i]) as array;
		for i = 1 to lum_info.count where (matchpattern (lum_info[i]) pattern: "Name=*" ) do (types_of_lum_number += 1)
		file_lum_info = (for i = 1 to (lumidata_index - 1) collect file_lum_info[i])
		lum_info
	)
)