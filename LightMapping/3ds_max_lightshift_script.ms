global lightIntMin = 50000
global lightIntMax = 50000
global lightIntStepValue = 500

global lightDistributionMin = 0.35
global lightDistributionMax = 0.35
global lightDistributionStep = 0.05

global curent_renderer = "nlvehvres2dtzzd"

global manager

global cur_lumen

global cur_directionality

global Range_Counter  = 0.1
global step = 0.1

global LightSelected = getnodebyname "VRayLight002" exact:true

global Measure_plane = getnodebyname "ScalePlane" exact:true
	
global MultiplyUnit = 1000

fn InitNetRender =
(
-- setup networkrendering
	try 
	(
		format "Start network initialization...\n"
		manager = netrender.getmanager()
		format "Searching for backburner manager and attempting to connect...\n"
		a = manager.connect #manual curent_renderer
		format "Connection to manager = %\n" a

		manager.QueryControl #wait
		manager.getcontrol()
		manager.wantControl=true
		manager.Lock true
	)
	catch (format "*** % ***\n" (getCurrentException()))
)

fn QueueNetRender renderfile=
(
	jobdone = false;
	while (jobdone == false) do
	(
		jobdone = true;
		format "Submitting job: % .....\n" renderfile
		windows.processPostedMessages();
		
		if (manager.connected == false) do
		(
			format "Trying to reconnect... "
			manager = netrender.getmanager()
			a = manager.connect #manual curent_renderer
			manager.QueryControl #wait
		)
		
		if (manager.connected == true) then
		(
			job = manager.newjob()
			job.name = renderfile
			job.includeMaps = false
			job.fromFrame = 1
			job.toFrame = 57

			job.submit()
			
		) else ( -- we aren't connected
			jobdone = false;
			format "Could not contact manager, retrying in 5 minutes...\n"
			--manager.disconnect();
			sleep 10;
			windows.processPostedMessages();
		)
		
		try -- check if the manager is still alive, to handle manager crashes
		(			
			ns = manager.netstatus;
		)
		catch
		(
			jobdone = false;
			format "Could not get netstatus, manager down, retrying in 5 minutes...\n"
			manager.disconnect();
			sleep 10;
			windows.processPostedMessages();
		)
	)
)

fn Render_job =
(
	makeDir ("O:/_sandbox/IntensityAdjustment/MaxRenders_26_6_2014/MaxExpor_Lumen_" + ((cur_lumen/10) as string) + "_" + ((180 - cur_directionality*180) as string) +"_" ) all:true
	rendOutputFilename = ("O:/_sandbox/IntensityAdjustment/MaxRenders_26_6_2014/MaxExpor_Lumen_" + ((cur_lumen/10) as string) + "_" + ((180 - cur_directionality*180) as string) +"_" + "/" + "MaxLumen_" + ((cur_lumen/10) as string)+ "_" + "Directionality_"  + ((180 - cur_directionality*180) as string) + "_" + "DistanceCounter_" + ".bmp")
	rendSaveFile = true
	format "Saving output as: %\n" rendOutputFilename
	QueueNetRender ("MaxIntensity_" + cur_lumen as string + "_" + (cur_directionality as string))
	saveMaxFile ("O:/_sandbox/IntensityAdjustment/MaxRenders_26_6_2014/MaxExpor_Lumen_" + ((cur_lumen/10) as string) + "_" + ((180 - cur_directionality*180) as string) +"_" + "/" + "MaxLumen_" + ((cur_lumen/10) as string) + "_" + "Directionality_"  + ((180 - cur_directionality*180) as string) + "_" + "DistanceCounter_" + ".max")
	InitNetRender()
)


fn Export_lights =
(	
	InitNetRender()
	for i = lightIntMin to lightIntMax by lightIntStepValue do
		(
		if i >= 10000 do lightIntStepValue = 2000
			for n = lightDistributionMin to lightDistributionMax by lightDistributionStep do
				(
				LightSelected.lightDistribution = n
				LightSelected.multiplier = i
				cur_directionality = n
				cur_lumen = i
				Render_job()
				)
		)
)

Export_lights()