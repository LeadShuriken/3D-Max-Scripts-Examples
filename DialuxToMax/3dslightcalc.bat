rem Example: 3dslightcalc c:\data\room.stf c:\data\positions.stf c:\data\output\ 

set ROOM_FILE=%1
set POS_FILE=%2
set OUTPUT_PATH=%3
set NX=%4
set NY=%5
set WZ=%6
set ALL_LUMINAIRES=%7
set BASE_PATH=%~dp0
rem set MAX_PATH=%ADSK_3DSMAX_x64_2016%
set MAX_PATH=%ADSK_MAXDES_x64_2015%

rem Execute 3dsmax with the automation script and the correct parameters
"%MAX_PATH%\3dsmax.exe" -y -mxs "global base_path = @\"%BASE_PATH%"; global room_file = @\"%ROOM_FILE%\"; global pos_file = @\"%POS_FILE%\"; global save_folder = @\"%OUTPUT_PATH%\"; global all_luminaires = @\"%ALL_LUMINAIRES%\"; global x_size = @\"%NX%\"; global y_size = @\"%NY%\"; global WZ = @\"%WZ%\"; fileIn @\"%BASE_PATH%dialux_convertor.ms\""