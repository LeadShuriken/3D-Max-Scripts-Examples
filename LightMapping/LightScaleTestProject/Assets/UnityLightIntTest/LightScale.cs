using UnityEngine;
using UnityEditor;
using System.Collections;
using System.IO;

[ExecuteInEditMode]

public class LightScale : MonoBehaviour {
	
	byte[] Tex = null;
	public Camera cam;
	public Light usedLight = new Light();
	public GameObject ScalePlane;

	public double i_min;
	public double i_max;
	public double i_step;

	public double r_min;
	public double r_max;
	public double r_step;

	private float range_step = 0.1f;
	private bool cleared;

	void OnGUI() 
	{
	if (GUI.Button (new Rect (10, 10, 150, 100), "Begin Rendering")) 
		{
		Render_Camera();
		}
	}

	void Render_Camera()
	{	

		for (double i = i_min ; i <= i_max; i += i_step)
			{
			for (double r = r_min; r <= r_max; r += r_step)
				{
					if(usedLight != null)
					{
					usedLight.intensity = (float)i;
					usedLight.range = (float)r;
					}
					range_step = 0.1f;
					for (float Range_Counter = 0.1f; Range_Counter <= 30; Range_Counter = Range_Counter + range_step)
						{
							if(Range_Counter > 2.9)
								{
								range_step = 1.0f;
								}

							ScalePlane.transform.position = new Vector3(0, 0, Range_Counter);
							RenderTexture.active = cam.targetTexture;
							cam.Render ();
							Texture2D image = new Texture2D (cam.targetTexture.width, cam.targetTexture.height);
							image.ReadPixels(new Rect (0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
							
							image.Apply (false);
				
							Tex = image.EncodeToPNG();
							
					System.IO.Directory.CreateDirectory("C://ossact/max_unity_light_mapping/UnityRenderers" + "/Range_" + (r).ToString("0.00") + "_Int_" + (i).ToString("0.00") + "/");

					File.WriteAllBytes (("C://ossact/max_unity_light_mapping/UnityRenderers" + "/Range_" + (r).ToString("0.00") + "_Int_" + (i).ToString("0.00") + "/" + "AtDistance_" + (Range_Counter).ToString("0.00") + "_Range_" + (r).ToString("0.00") + "_Int_" + (i).ToString("0.00") + "_.png"), Tex);

							DestroyImmediate (image);
						}
				if (Input.GetKeyDown("space"))
				{return;}
				}
			if (Input.GetKeyDown("space"))
			{return;}
			}
			ScalePlane.transform.position = new Vector3(0, 0, 0.1f);

			var FileInfo = new FileInfo("C://ossact/max_unity_light_mapping/Batch.bat");
			System.Diagnostics.Process.Start(FileInfo.FullName);
	}
}